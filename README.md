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

**GPU (NVIDIA T4, via Colab), γ=4, 20 prompts:**

| Metric | Baseline (target alone) | Speculative |
|---|---|---|
| Mean tokens/sec | 52.73 | 45.11 |
| Naive mean speedup (all 20 prompts) | 1.0x (reference) | 1.19x — **misleading, see below** |
| Median speedup (all 20 prompts) | 1.0x (reference) | **0.89x** (slower) |
| Length-matched mean speedup (18 prompts) | 1.0x (reference) | **0.89x** (slower) |
| Aggregate throughput ratio (18 prompts) | 1.0x (reference) | **0.88x** (slower) |
| Mean token acceptance rate | — | 50.8% |

**γ sweep (first 8 prompts), GPU:**

| γ | Mean speedup | Mean acceptance rate |
|---|---|---|
| 2 | 0.97x | 68.8% |
| 4 | 0.87x | 53.6% |
| 8 | 0.69x | 34.7% |

#### Why the mean and median disagree

The raw 20-prompt mean (1.19x) is **not the number to trust** — it's dragged
up entirely by one prompt (#13) where the speculative run happened to sample
an EOS token after ~6 tokens while the baseline ran the full 100, making a
slow per-token rate look like a 7.6x "speedup" purely because it finished
early. Prompt #12 shows the mirror-image problem in the other direction
(baseline hit EOS at ~27 tokens, speculative ran the full 100, crushing its
ratio to 0.21x). Verified directly: multiplying each row's `time × tokens/sec`
recovers the actual token count generated, and confirms exactly these two
prompts — and only these two — finished short of 100 tokens on one side.
Different sampling paths can legitimately hit EOS at different points, which
breaks a raw wall-clock ratio's implicit assumption that both sides generated
the same number of tokens.

Excluding those two length-mismatched prompts, three independent ways of
computing the aggregate all converge tightly: length-matched mean (0.891x),
median across all 20 (0.887x, robust to outliers by construction), and the
aggregate tokens/sec ratio (0.881x). That convergence is the actual result:
**~0.88–0.89x — a genuine, if modest, slowdown, even on a real GPU**, for
this draft/target pair.

Full per-prompt numbers (with explicit `note` fields on the two flagged rows):
[`results/results_cuda.json`](results/results_cuda.json).
Chart: [`results/benchmark_chart_cuda.png`](results/benchmark_chart_cuda.png).

## Why hardware (and model scale) matter here (this is the actual finding)

Speculative decoding's speedup comes from a specific property of decoding on
**GPUs at production model scale**: generating one token at a time from a
large model is **memory-bandwidth bound**, not compute bound — the GPU spends
most of its time waiting on weights to stream in from memory, with plenty of
spare compute sitting idle. That means running the target model over γ+1
tokens in one batched forward pass costs *roughly the same wall-clock time*
as running it over 1 token — the extra tokens are almost free. Verifying γ
draft tokens "for the price of one" is where the whole speedup comes from,
and it only costs the (comparatively tiny) sequential draft-model calls.

**CPU decoding does not have this property.** A CPU forward pass over γ+1
tokens costs roughly (γ+1)× a single-token pass, because CPU matrix
multiplication is much closer to compute-bound — there's no large pool of
idle compute to hide the extra tokens behind. That breaks the core assumption
the algorithm depends on: batched verification is no longer "free," so the γ
sequential draft-model calls become pure added overhead. Every γ value tested
on CPU is a slowdown, and it gets worse as γ grows.

**Moving to a real GPU (T4) narrows the gap but still doesn't cross 1x** —
~0.88-0.89x, not the 2-3x+ speedups reported for production speculative
decoding. The reason is model scale, not the GPU: `gpt2`/`gpt2-medium` are
small enough (124M/355M params) that a single forward pass is already fast
(45-55 tokens/sec), so fixed per-call overhead — Python dispatch, CUDA kernel
launch latency, HBM round trips for tiny matmuls — makes up a much bigger
share of total time than it would for a 7B+ model. That overhead isn't
memory-bandwidth-bound in the way the paper's regime assumes, so batching γ+1
tokens isn't actually "free" here either — it's just less expensive than on
CPU. The γ-sweep confirms the mechanism either way: acceptance rate falls
monotonically as γ grows (2→68.8%, 4→53.6%, 8→34.7%), exactly as the theory
predicts (a weak draft model agrees with the target less often the further
ahead it's asked to guess) — the math and the trend are both right, the
model pair is just too small and the GPU too underutilized by these tiny
matmuls for the payoff to fully materialize.

This isn't a bug in the implementation — the [correctness check](#correctness-check)
confirms the accept/reject math is right, and both the CPU and GPU γ-sweeps
show the expected qualitative trend. It's a direct, measurable illustration
of why speculative decoding is described in the literature as a large-model
GPU-serving optimization, not a general-purpose decoding trick: the
technique's payoff is inseparable from the hardware's compute/
memory-bandwidth balance *and* from the target model being large enough that
a forward pass's fixed overhead is negligible next to its compute cost.

## What I'd explore next

- **Scale up the target model**: rerun the identical benchmark with a 7B-class
  target (e.g. `Llama-3.1-8B` + a small draft like `Llama-3.2-1B`) on the same
  T4-class GPU. The hypothesis this repo sets up directly: as the target's
  per-forward-pass compute cost grows, fixed overhead shrinks as a share of
  total time and the batched-verify step gets closer to genuinely "free,"
  which should push the speedup from ~0.88x here toward the >1x regime
  reported in production serving systems.
- **Tree-based speculation** (Medusa, EAGLE): instead of one linear γ-length
  draft, propose a *tree* of candidate continuations and verify the whole
  tree in one target forward pass — recovers more accepted tokens per target
  call, which matters even more when per-call overhead is significant since
  it amortizes that overhead across more accepted tokens.
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

Benchmarked on CPU (Apple M1, 8GB RAM) and on a T4 GPU (Google Colab). The
relative speedup/acceptance-rate methodology, and the outlier-handling
approach for length-mismatched runs, are what's being demonstrated here — not
a production serving-throughput claim, which would require a much larger
target model to show this technique's actual production payoff (see
"What I'd explore next").
