"""Benchmark speculative decoding (draft=gpt2, target=gpt2-medium) against
plain autoregressive decoding with the target model alone.

Measures, per prompt: wall-clock latency, tokens/sec, and (for the
speculative method) the draft token acceptance rate. Then aggregates mean/
median speedup and acceptance rate across the prompt set, and sweeps gamma
in {2, 4, 8} on a subset of prompts to show the accuracy/speed tradeoff.
"""

import json
import statistics
import time

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.baseline import baseline_generate
from src.spec_decode import SpeculativeDecoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DRAFT_MODEL = "gpt2"
TARGET_MODEL = "gpt2-medium"
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.8
DEFAULT_GAMMA = 4
GAMMA_SWEEP = [2, 4, 8]
GAMMA_SWEEP_N_PROMPTS = 8  # first N prompts, to keep sweep runtime reasonable


def load_prompts(path="prompts.jsonl"):
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def main():
    prompts = load_prompts()
    print(f"Loaded {len(prompts)} prompts")

    print(f"Loading draft={DRAFT_MODEL}, target={TARGET_MODEL} on device={DEVICE} ...")
    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    tok.pad_token = tok.eos_token
    draft_model = AutoModelForCausalLM.from_pretrained(DRAFT_MODEL).to(DEVICE).eval()
    target_model = AutoModelForCausalLM.from_pretrained(TARGET_MODEL).to(DEVICE).eval()
    target_model_baseline = target_model  # same weights, no persistent state on the model object itself

    decoder = SpeculativeDecoder(draft_model, target_model, tok, gamma=DEFAULT_GAMMA, temperature=TEMPERATURE, device=DEVICE)

    per_prompt = []
    for p in prompts:
        seed = p["id"]

        t0 = time.time()
        base_res = baseline_generate(target_model_baseline, tok, p["prompt"], max_new_tokens=MAX_NEW_TOKENS,
                                      temperature=TEMPERATURE, seed=seed, device=DEVICE)
        base_time = time.time() - t0
        base_tps = base_res["n_generated"] / base_time

        t0 = time.time()
        spec_res = decoder.generate(p["prompt"], max_new_tokens=MAX_NEW_TOKENS, seed=seed)
        spec_time = time.time() - t0
        spec_tps = spec_res["n_generated"] / spec_time

        speedup = base_time / spec_time
        row = {
            "id": p["id"],
            "category": p["category"],
            "prompt": p["prompt"],
            "baseline_time_s": round(base_time, 4),
            "baseline_tokens_per_s": round(base_tps, 2),
            "spec_time_s": round(spec_time, 4),
            "spec_tokens_per_s": round(spec_tps, 2),
            "speedup": round(speedup, 4),
            "acceptance_rate": round(spec_res["acceptance_rate"], 4),
        }
        per_prompt.append(row)
        print(f"[{p['id']:2d}] {p['category']:14s} speedup={speedup:.2f}x  "
              f"accept={spec_res['acceptance_rate']:.3f}  "
              f"baseline={base_tps:.1f}tok/s  spec={spec_tps:.1f}tok/s")

    speedups = [r["speedup"] for r in per_prompt]
    accept_rates = [r["acceptance_rate"] for r in per_prompt]
    summary = {
        "gamma": DEFAULT_GAMMA,
        "n_prompts": len(per_prompt),
        "mean_speedup": statistics.mean(speedups),
        "median_speedup": statistics.median(speedups),
        "min_speedup": min(speedups),
        "max_speedup": max(speedups),
        "mean_acceptance_rate": statistics.mean(accept_rates),
        "mean_baseline_tokens_per_s": statistics.mean(r["baseline_tokens_per_s"] for r in per_prompt),
        "mean_spec_tokens_per_s": statistics.mean(r["spec_tokens_per_s"] for r in per_prompt),
    }
    print("\n=== SUMMARY (gamma=%d) ===" % DEFAULT_GAMMA)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # --- gamma sweep on a subset of prompts ---
    print(f"\nRunning gamma sweep {GAMMA_SWEEP} on first {GAMMA_SWEEP_N_PROMPTS} prompts...")
    sweep_prompts = prompts[:GAMMA_SWEEP_N_PROMPTS]
    gamma_results = []
    for gamma in GAMMA_SWEEP:
        dec_g = SpeculativeDecoder(draft_model, target_model, tok, gamma=gamma, temperature=TEMPERATURE, device=DEVICE)
        g_speedups, g_accepts = [], []
        for p in sweep_prompts:
            seed = p["id"]
            base_row = next(r for r in per_prompt if r["id"] == p["id"])
            t0 = time.time()
            res = dec_g.generate(p["prompt"], max_new_tokens=MAX_NEW_TOKENS, seed=seed)
            dt = time.time() - t0
            g_speedups.append(base_row["baseline_time_s"] / dt)
            g_accepts.append(res["acceptance_rate"])
        row = {
            "gamma": gamma,
            "mean_speedup": statistics.mean(g_speedups),
            "mean_acceptance_rate": statistics.mean(g_accepts),
        }
        gamma_results.append(row)
        print(f"  gamma={gamma}: mean_speedup={row['mean_speedup']:.3f}x  "
              f"mean_acceptance_rate={row['mean_acceptance_rate']:.3f}")

    results = {
        "config": {
            "device": DEVICE,
            "draft_model": DRAFT_MODEL,
            "target_model": TARGET_MODEL,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "default_gamma": DEFAULT_GAMMA,
        },
        "summary": summary,
        "per_prompt": per_prompt,
        "gamma_sweep": gamma_results,
    }
    out_json = f"results/results_{DEVICE}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_json}")

    # --- charts ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(["baseline\n(target alone)", f"speculative\n(gamma={DEFAULT_GAMMA})"],
                [summary["mean_baseline_tokens_per_s"], summary["mean_spec_tokens_per_s"]],
                color=["#888888", "#4C72B0"])
    axes[0].set_ylabel("tokens / sec (mean over %d prompts)" % len(per_prompt))
    axes[0].set_title("Throughput: baseline vs speculative")
    for i, v in enumerate([summary["mean_baseline_tokens_per_s"], summary["mean_spec_tokens_per_s"]]):
        axes[0].text(i, v + 0.3, f"{v:.1f}", ha="center")

    gammas = [r["gamma"] for r in gamma_results]
    speedups_g = [r["mean_speedup"] for r in gamma_results]
    accepts_g = [r["mean_acceptance_rate"] for r in gamma_results]
    ax2 = axes[1]
    ax2.plot(gammas, speedups_g, "o-", color="#4C72B0", label="speedup (x)")
    ax2.set_xlabel("gamma (draft tokens per round)")
    ax2.set_ylabel("speedup (x)", color="#4C72B0")
    ax2.set_xticks(gammas)
    ax3 = ax2.twinx()
    ax3.plot(gammas, accepts_g, "s--", color="#C44E52", label="acceptance rate")
    ax3.set_ylabel("acceptance rate", color="#C44E52")
    ax2.set_title("Speedup & acceptance rate vs gamma")

    fig.suptitle(f"device={DEVICE}")
    fig.tight_layout()
    out_png = f"results/benchmark_chart_{DEVICE}.png"
    fig.savefig(out_png, dpi=150)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
