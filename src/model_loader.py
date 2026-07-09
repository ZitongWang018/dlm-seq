"""Load LLaDA model and tokenizer."""
from __future__ import annotations

import torch
from transformers import AutoModel, AutoTokenizer


def load_llada(model_path: str, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()
    return model, tokenizer


def encode_prompt(tokenizer, prompt: str, device: str = "cuda"):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    enc = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    return input_ids, attention_mask
