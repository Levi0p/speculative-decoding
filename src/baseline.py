"""Plain autoregressive decoding with the target model alone, used as the
wall-clock baseline that speculative decoding is compared against. Same
temperature and RNG discipline as spec_decode.py for a fair comparison."""

import torch
import torch.nn.functional as F


def _sample(probs, generator):
    return torch.multinomial(probs, num_samples=1, generator=generator).item()


@torch.no_grad()
def baseline_generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, seed=0, device="cpu"):
    generator = torch.Generator(device=device).manual_seed(seed)
    model = model.to(device).eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]

    out = model(input_ids=input_ids, use_cache=True)
    cache = out.past_key_values
    logits = out.logits[0, -1]
    eos_id = tokenizer.eos_token_id

    generated = []
    for _ in range(max_new_tokens):
        probs = F.softmax(logits.double() / temperature, dim=-1)
        next_tok = _sample(probs, generator)
        generated.append(next_tok)
        if eos_id is not None and next_tok == eos_id:
            break
        out = model(input_ids=torch.tensor([[next_tok]], device=device), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        logits = out.logits[0, -1]

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return {"text": text, "n_generated": len(generated)}
