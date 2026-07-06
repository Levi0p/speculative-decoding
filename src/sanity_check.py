"""Correctness check: if the draft and target model are the SAME model,
q_i and p_i are computed from mathematically identical distributions, so
the acceptance rate should be ~100% (modulo tiny floating-point
differences between the batched target forward pass and the incremental
draft forward pass). A low acceptance rate here would mean the accept/
reject math is wrong, not that the draft model is weak.
"""

import json
import sys
import time

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.spec_decode import SpeculativeDecoder

MODEL_NAME = "gpt2"
PROMPTS = [
    "The capital of France is",
    "In machine learning, overfitting occurs when",
    "Once upon a time, in a small village,",
    "The main difference between a list and a tuple in Python is",
]


def main():
    print(f"Loading {MODEL_NAME} twice (draft == target)...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token
    draft = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    target = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    decoder = SpeculativeDecoder(draft, target, tok, gamma=4, temperature=0.8)

    rates = []
    for i, prompt in enumerate(PROMPTS):
        t0 = time.time()
        result = decoder.generate(prompt, max_new_tokens=60, seed=100 + i)
        dt = time.time() - t0
        rates.append(result["acceptance_rate"])
        print(f"\nPrompt: {prompt!r}")
        print(f"  acceptance_rate={result['acceptance_rate']:.4f} "
              f"({result['total_accepted']}/{result['total_proposed']}) time={dt:.2f}s")
        print(f"  generated: {result['text']!r}")

    mean_rate = sum(rates) / len(rates)
    print(f"\nMean acceptance rate across {len(PROMPTS)} prompts (draft==target): {mean_rate:.4f}")
    ok = mean_rate > 0.95
    print("SANITY CHECK", "PASSED" if ok else "FAILED", f"(threshold 0.95, got {mean_rate:.4f})")

    with open("results/sanity_check.json", "w") as f:
        json.dump({"mean_acceptance_rate": mean_rate, "per_prompt": rates, "passed": ok}, f, indent=2)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
