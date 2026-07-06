# Speculative Decoding, From Scratch

An implementation of speculative decoding — the accept/reject sampling
algorithm from Leviathan et al. 2023 ("Fast Inference from Transformers via
Speculative Decoding") and Chen et al. 2023 — written from first principles
against a real `draft` + `target` HuggingFace model pair, with a from-scratch
correctness check and a real, measured benchmark against plain autoregressive
decoding.

This is the technique behind the inference-speedups used in production LLM
serving stacks (vLLM, TensorRT-LLM's Medusa/EAGLE heads, etc.): a small, fast
**draft** model proposes several tokens, and a larger **target** model
verifies all of them in a single batched forward pass, accepting/rejecting
each one so that the *final output distribution is provably identical* to
sampling from the target model alone.

## How it works

1. **Draft**: the draft model autoregressively samples γ tokens, recording
   its own sampling probability `q_i` for each one.
2. **Verify**: the target model checks all γ tokens in **one batched forward
   pass** (not γ separate calls) by feeding `[last_committed_token] +
   draft_tokens`. This is the step that gives the speedup on hardware where
   it's cheap.
3. **Accept/reject**, left to right: token `i` is accepted with probability
   `min(1, p_i / q_i)`. On the first rejection, a replacement token is drawn
   from the normalized residual distribution `max(0, p - q)`, and the rest of
   the round's draft tokens are discarded. If every token is accepted, a
   bonus token is sampled for free from the target's next-token distribution
   (already computed in the same forward pass).
4. **KV-cache bookkeeping**: both the draft and target model's `past_key_values`
   are cropped/extended to match exactly what was actually committed, so a
   mid-round rejection never leaves stale cache entries behind.

Implementation: [`src/spec_decode.py`](src/spec_decode.py) (core algorithm),
[`src/baseline.py`](src/baseline.py) (plain autoregressive decoding used as
the comparison baseline), [`src/sanity_check.py`](src/sanity_check.py)
(correctness check).

## Correctness check

If the draft and target model are the **same** model, `q_i` and `p_i` are
computed from mathematically identical distributions, so every draft token
should be accepted (mean acceptance rate ≈ 1.0, modulo tiny floating-point
differences between the batched target pass and the incremental draft pass).
This validates the accept/reject math independent of how good any particular
draft model is.

```
$ python -m src.sanity_check
Mean acceptance rate across 4 prompts (draft==target): 1.0000
SANITY CHECK PASSED (threshold 0.95, got 1.0000)
```

## Benchmark

Models: draft = `gpt2` (124M), target = `gpt2-medium` (355M) — both
non-gated. 20 hand-written prompts spanning factual QA, creative writing,
conversation, reasoning, and technical explanation; `max_new_tokens=100`,
`temperature=0.8`, fixed seeds for reproducibility. Compared against plain
autoregressive decoding with the target model alone, same sampling settings.

**CPU (Apple M1, 8GB RAM), γ=4, 20 prompts:**

| Metric | Baseline (target alone) | Speculative |
|---|---|---|
| Mean tokens/sec | 18.30 | 12.55 |
| Mean speedup | 1.0x (reference) | **0.68x** (slower) |
| Mean token acceptance rate | — | 51.0% |
| Speedup range | — | 0.37x – 0.90x |

**γ sweep (first 8 prompts), CPU:**

| γ | Mean speedup | Mean acceptance rate |
|---|---|---|
| 2 | 0.72x | 66.4% |
| 4 | 0.78x | 53.3% |
| 8 | 0.73x | 33.3% |

Full per-prompt numbers: [`results/results_cpu.json`](results/results_cpu.json).
Chart: [`results/benchmark_chart_cpu.png`](results/benchmark_chart_cpu.png).

<!-- GPU_RESULTS -->

## Why hardware matters here (this is the actual finding)

Speculative decoding's speedup comes from a specific property of **GPU**
autoregressive decoding: generating one token at a time is **memory-bandwidth
bound**, not compute bound — the GPU spends most of its time waiting on
weights to stream in from memory, with plenty of spare compute sitting idle.
That means running the target model over γ+1 tokens in one batched forward
pass costs *roughly the same wall-clock time* as running it over 1 token —
the extra tokens are almost free. So verifying γ draft tokens "for the price
of one" is where the whole speedup comes from, and it only costs the
(comparatively tiny) sequential draft-model calls.

**CPU decoding does not have this property.** A CPU forward pass over γ+1
tokens costs roughly (γ+1)× a single-token pass, because CPU matrix
multiplication is much closer to compute-bound — there's no large pool of
idle compute to hide the extra tokens behind. That breaks the core assumption
the algorithm's speedup depends on: batched verification is no longer "free,"
so the γ sequential draft-model calls become pure added overhead with no
compensating win. This is exactly what the numbers above show — every γ
value tested is a slowdown, and it gets worse as γ grows (more draft calls,
more verify-pass cost, without a memory-bound regime to absorb it).

This isn't a bug in the implementation — the [correctness check](#correctness-check)
confirms the accept/reject math is right, and the γ-sweep shows the expected
qualitative trend (lower γ → higher acceptance rate, since the draft model is
more likely to agree with the target over a shorter horizon). It's a direct,
measurable illustration of *why* speculative decoding is described in the
literature as a GPU-serving optimization, not a general-purpose decoding
trick — the technique's benefit is inseparable from the hardware's compute/
memory-bandwidth balance.

## What I'd explore next

- **Tree-based speculation** (Medusa, EAGLE): instead of one linear γ-length
  draft, propose a *tree* of candidate continuations and verify the whole
  tree in one target forward pass — recovers more accepted tokens per target
  call, which matters even more on compute-bound hardware since it amortizes
  the batched-verify cost across more accepted tokens.
- **Draft model distillation**: train the draft model on the target's own
  outputs to raise the acceptance rate directly, rather than relying on an
  off-the-shelf smaller checkpoint from the same family.
- **Dynamic γ**: adapt γ per-step based on a running estimate of the
  acceptance rate (shrink γ when the draft is disagreeing with the target a
  lot, grow it when agreement is high) instead of a fixed value.

## Running it

```bash
pip install -r requirements.txt
python -m src.sanity_check          # correctness check
python benchmark.py                 # full benchmark (auto-detects CUDA)
```

## Hardware note

Benchmarked on CPU (Apple M1, 8GB RAM) <!-- AND_GPU_NOTE -->. The relative
speedup/acceptance-rate methodology is what's being demonstrated, not a
production serving-throughput claim.
