"""
Speculative decoding, implemented from scratch (Leviathan et al. 2023 /
Chen et al. 2023 speculative sampling).

Convention used throughout: after any point in the loop, `draft_cache` and
`target_cache` hold KV entries for every committed token EXCEPT the most
recent one. That token is fed as the first input of the next round. This
keeps the bookkeeping symmetric between the draft and target models and
means no extra "priming" forward pass is ever needed beyond the initial
prefill.

Each round:
  1. Draft model samples gamma tokens autoregressively, recording its own
     sampling probability q_i for each token.
  2. Target model verifies all gamma tokens in a SINGLE forward pass
     (batched, not one at a time) by feeding [last_committed_token] +
     draft_tokens. Output position i gives the target distribution p_i that
     should have generated draft_tokens[i].
  3. Tokens are accepted left to right with probability min(1, p_i/q_i).
     On the first rejection, a replacement token is drawn from the
     normalized residual max(0, p - q) and the rest of the draft is
     discarded. If every token is accepted, a bonus token is drawn for free
     from the target's next-token distribution.
  4. Both KV caches are cropped/extended to match whatever was actually
     committed, so a rejection never leaves stale KV entries behind.
"""

import torch
import torch.nn.functional as F


EPS = 1e-10


def _softmax_temp(logits, temperature):
    return F.softmax(logits.double() / temperature, dim=-1)


def _sample(probs, generator):
    return torch.multinomial(probs, num_samples=1, generator=generator).item()


def _crop_cache(past_key_values, length):
    """Truncate a KV cache to `length` tokens along the sequence dimension.
    Handles the legacy tuple-of-(key, value)-per-layer format returned by
    this pinned transformers version (each tensor: batch, heads, seq, dim)."""
    return tuple((k[:, :, :length, :], v[:, :, :length, :]) for k, v in past_key_values)


class SpeculativeDecoder:
    def __init__(self, draft_model, target_model, tokenizer, gamma=4, temperature=0.8, device="cpu"):
        self.draft_model = draft_model.to(device).eval()
        self.target_model = target_model.to(device).eval()
        self.tokenizer = tokenizer
        self.gamma = gamma
        self.temperature = temperature
        self.device = device

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=100, seed=0):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        prompt_ids = input_ids[0].tolist()

        # Prefill both models on prompt[:-1]; the last prompt token becomes
        # the first "not yet scored" token, matching the round convention.
        prefix = torch.tensor([prompt_ids[:-1]], device=self.device) if len(prompt_ids) > 1 else None
        last_token = prompt_ids[-1]

        draft_cache = None
        target_cache = None
        if prefix is not None and prefix.shape[1] > 0:
            d_out = self.draft_model(input_ids=prefix, use_cache=True)
            draft_cache = d_out.past_key_values
            t_out = self.target_model(input_ids=prefix, use_cache=True)
            target_cache = t_out.past_key_values

        committed = list(prompt_ids)
        n_generated = 0
        total_accepted = 0
        total_proposed = 0
        eos_id = self.tokenizer.eos_token_id

        while n_generated < max_new_tokens:
            gamma = min(self.gamma, max_new_tokens - n_generated)

            # 1. Draft model proposes `gamma` tokens autoregressively.
            draft_tokens = []
            draft_dists = []  # full distribution q at each step, needed for residual resampling
            cur_tok = last_token
            cur_draft_cache = draft_cache
            for _ in range(gamma):
                out = self.draft_model(
                    input_ids=torch.tensor([[cur_tok]], device=self.device),
                    past_key_values=cur_draft_cache,
                    use_cache=True,
                )
                cur_draft_cache = out.past_key_values
                q_dist = _softmax_temp(out.logits[0, -1], self.temperature)
                next_tok = _sample(q_dist, generator)
                draft_tokens.append(next_tok)
                draft_dists.append(q_dist)
                cur_tok = next_tok

            # 2. Target model verifies all gamma tokens in one forward pass.
            target_input = torch.tensor([[last_token] + draft_tokens], device=self.device)
            t_out = self.target_model(input_ids=target_input, past_key_values=target_cache, use_cache=True)
            full_target_cache = t_out.past_key_values
            t_logits = t_out.logits[0]  # (gamma+1, vocab); row i -> distribution validating draft_tokens[i] for i<gamma, row gamma -> bonus

            # 3. Accept/reject left to right.
            accepted = 0
            new_token = None
            for i in range(gamma):
                p_dist = _softmax_temp(t_logits[i], self.temperature)
                tok = draft_tokens[i]
                p_i = p_dist[tok].item()
                q_i = draft_dists[i][tok].item()
                accept_prob = min(1.0, p_i / max(q_i, EPS))
                if torch.rand(1, generator=generator, device=self.device).item() < accept_prob:
                    accepted += 1
                else:
                    residual = torch.clamp(p_dist - draft_dists[i], min=0.0)
                    residual_sum = residual.sum().item()
                    if residual_sum <= 0:
                        # p and q coincide (or numerical edge case); fall back to p directly.
                        residual = p_dist
                    else:
                        residual = residual / residual_sum
                    new_token = _sample(residual, generator)
                    break

            total_accepted += accepted
            total_proposed += gamma

            if accepted < gamma:
                # Cache holds KV for last_token + all gamma draft tokens; keep only
                # the accepted prefix, drop the rest, new_token is unscored.
                keep_len = len(committed) + accepted
                full_target_cache = _crop_cache(full_target_cache, keep_len)
                cur_draft_cache = _crop_cache(cur_draft_cache, keep_len)
                new_committed = draft_tokens[:accepted] + [new_token]
            else:
                # target_input always had gamma+1 tokens, so row `gamma` (the
                # bonus row) always exists.
                bonus_dist = _softmax_temp(t_logits[gamma], self.temperature)
                bonus_token = _sample(bonus_dist, generator)
                new_committed = draft_tokens + [bonus_token]
                # The draft loop only ever FEEDS gamma-1 of its own tokens back
                # into itself (each sampled token is used as the next input,
                # but the very last one, draft_tokens[-1], is never re-fed) so
                # its cache is one entry short of the target's. Feed it now so
                # both caches land at exactly committed_length-1.
                out = self.draft_model(
                    input_ids=torch.tensor([[draft_tokens[-1]]], device=self.device),
                    past_key_values=cur_draft_cache,
                    use_cache=True,
                )
                cur_draft_cache = out.past_key_values

            committed.extend(new_committed)
            n_generated += len(new_committed)
            last_token = committed[-1]
            draft_cache = cur_draft_cache
            target_cache = full_target_cache

            if eos_id is not None and eos_id in new_committed:
                break

        generated_ids = committed[len(prompt_ids):len(prompt_ids) + max_new_tokens]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        acceptance_rate = total_accepted / total_proposed if total_proposed > 0 else 0.0
        return {
            "text": text,
            "n_generated": len(generated_ids),
            "acceptance_rate": acceptance_rate,
            "total_proposed": total_proposed,
            "total_accepted": total_accepted,
        }
