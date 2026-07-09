"""Node 5: HumanEval LCR baseline + H4 ceiling measurement.

For HumanEval wrong samples: was the correct token ever argmax at
any position in the function body during denoising?

HumanEval has code structure:
- Scaffolding: function signature, return type (stable)  
- Body: algorithm tokens (potentially variable)
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets_v2 import load_humaneval, humaneval_prompt
from src.model_loader import load_llada, encode_prompt
from src.samplers import MASK_ID, get_num_transfer_tokens


def _fwd(model, x):
    return model(x).logits


def run_code_tests(code: str, item: dict, timeout: int = 10) -> bool:
    full_code = f"{code}\n\n{item['test']}\n\ncheck({item['entry_point']})"
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(full_code)
            fname = f.name
        res = subprocess.run(["python", fname], capture_output=True, timeout=timeout)
        return res.returncode == 0
    except Exception:
        return False
    finally:
        Path(fname).unlink(missing_ok=True)


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    lines = text.strip().splitlines()
    code_lines = []
    for line in lines:
        if line.startswith(("def ", "from ", "import ", "    ")) or code_lines:
            code_lines.append(line)
    return "\n".join(code_lines).strip() if code_lines else text.strip()


@torch.no_grad()
def run_humaneval_with_ceiling(model, tokenizer, item: dict,
                                steps: int = 128, gen_length: int = 256):
    device = next(model.parameters()).device
    prompt = humaneval_prompt(item) + item["prompt"]  # include the function signature
    input_ids, _ = encode_prompt(tokenizer, prompt, device=device)
    prompt_len = input_ids.shape[1]

    x = torch.cat([
        input_ids,
        torch.full((1, gen_length), MASK_ID, dtype=torch.long, device=device),
    ], dim=1)

    bmi = x[:, prompt_len:prompt_len + gen_length] == MASK_ID
    num_transfer = get_num_transfer_tokens(bmi, steps)

    argmax_history: dict[int, list[int]] = {}

    for i in range(steps):
        mask_index = x == MASK_ID
        logits = _fwd(model, x)
        probs = torch.softmax(logits[0].float(), dim=-1)
        x0 = torch.argmax(logits[0], dim=-1)

        masked_pos = torch.nonzero(mask_index[0], as_tuple=False).flatten().tolist()
        for pos in masked_pos:
            if pos not in argmax_history:
                argmax_history[pos] = []
            argmax_history[pos].append(int(x0[pos].item()))

        scores = {pos: float(probs[pos, int(x0[pos].item())].item()) for pos in masked_pos}
        k = int(num_transfer[0, i].item())
        selected = sorted(masked_pos, key=lambda p: scores.get(p, -1e9), reverse=True)[:k]

        transfer = torch.zeros_like(x0, dtype=torch.bool)
        for pos in selected:
            transfer[pos] = True
        x0_full = torch.where(mask_index[0], x0, x[0])
        x[0][transfer] = x0_full[transfer]

    gen_tokens = x[0, prompt_len:].tolist()
    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    code = extract_code(item["prompt"] + "\n" + gen_text)
    is_correct = run_code_tests(code, item)

    # Flip counts for code body
    flip_counts = {}
    for pos, hist in argmax_history.items():
        flip_counts[pos] = sum(1 for a, b in zip(hist[:-1], hist[1:]) if a != b)

    # Ceiling: tokenize canonical solution, check if correct tokens ever appeared
    gold_code = item["canonical_solution"]
    gold_tokens = tokenizer.encode(gold_code, add_special_tokens=False)

    ever_correct_at_position = 0
    total_body_positions = min(gen_length, len(gold_tokens))
    for rel in range(min(total_body_positions, len(gold_tokens))):
        pos = prompt_len + rel
        if pos in argmax_history:
            if gold_tokens[rel] in argmax_history[pos]:
                ever_correct_at_position += 1

    ever_correct_ratio = ever_correct_at_position / max(total_body_positions, 1)
    mean_flips = sum(flip_counts.values()) / max(len(flip_counts), 1)

    return {
        "correct": is_correct,
        "task_id": item["task_id"],
        "entry_point": item["entry_point"],
        "ever_correct_ratio": ever_correct_ratio,
        "ever_correct_at_position": ever_correct_at_position,
        "total_body_positions": total_body_positions,
        "mean_flips": mean_flips,
        "gen_snippet": gen_text[:200],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--model_path", default="/root/autodl-tmp/model/LLaDA/instruct")
    parser.add_argument("--dataset_root", default="/root/autodl-tmp/dataset")
    parser.add_argument("--out", default="results/humaneval_ceiling/he_ceiling.json")
    args = parser.parse_args()

    print("Loading model ...")
    model, tokenizer = load_llada(args.model_path)

    print(f"Loading HumanEval (limit={args.limit}) ...")
    items = load_humaneval(args.dataset_root, limit=args.limit)

    records = []
    n_correct = 0
    ceiling_wrong = 0
    n_wrong = 0

    for idx, item in enumerate(items):
        print(f"[{idx+1}/{len(items)}] {item['entry_point']}", end="\r", flush=True)
        try:
            res = run_humaneval_with_ceiling(model, tokenizer, item, args.steps, args.gen_length)
            res["idx"] = idx
            records.append(res)
            if res["correct"]:
                n_correct += 1
            else:
                n_wrong += 1
                if res["ever_correct_ratio"] > 0.3:
                    ceiling_wrong += 1
        except Exception as e:
            print(f"\n[ERROR] idx={idx}: {e}")

    acc = n_correct / max(len(items), 1)
    ceiling = ceiling_wrong / max(n_wrong, 1)

    print(f"\n=== HumanEval Ceiling ===")
    print(f"pass@1 (LCR): {acc:.1%} ({n_correct}/{len(items)})")
    print(f"Wrong:        {n_wrong}")
    print(f"Ceiling (>30% body positions ever correct): {ceiling:.1%} ({ceiling_wrong}/{n_wrong})")
    print(f"Interpretation: {'HIGH \u2014 code trajectories have recoverable signal' if ceiling > 0.20 else 'LOW \u2014 code trajectories also stable-wrong'}")
    print(f"SCORE: {acc * 100:.1f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "summary": {
                "dataset": "HumanEval",
                "total": len(items),
                "correct": n_correct,
                "accuracy": acc,
                "ceiling_ratio": ceiling,
            },
            "records": records,
        }, f, indent=2, default=str)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
