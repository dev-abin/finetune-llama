"""Compare base and saved adapter using identical prompts and greedy decoding."""

import argparse
from pathlib import Path

from workflow import load_model, messages, read_json, read_rows, write_json


def generate(model, tokenizer, rows, config, max_new_tokens):
    import torch
    from transformers import GenerationConfig

    model.eval()
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    original_generation = model.generation_config
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    original_base_generation = base_model.generation_config
    generation = GenerationConfig(
        do_sample=False, max_new_tokens=max_new_tokens, use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=original_generation.eos_token_id or tokenizer.eos_token_id,
    )
    answers = []
    # Replace defaults temporarily so sampling/length settings from the base model
    # cannot leak into this controlled greedy comparison (Transformers 5 behavior).
    model.generation_config = generation
    try:
        for row in rows:
            encoded = tokenizer.apply_chat_template(
                messages(row, config), add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            ).to(model.device)
            with torch.inference_mode():
                generated = model.generate(**encoded)
            answers.append(tokenizer.decode(generated[0, encoded.input_ids.shape[1]:], skip_special_tokens=True).strip())
    finally:
        model.generation_config = original_generation
        base_model.generation_config = original_base_generation
    return answers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/llama32-sarcastic")
    parser.add_argument("--data", help="Optional JSONL with instruction/input/output fields")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    run = Path(args.run_dir)
    config = read_json(run / "run_config.json")
    rows = read_rows(args.data or run / "test_data.jsonl")
    from peft import PeftModel
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(run / "adapter")
    model = load_model(config)
    base_answers = generate(model, tokenizer, rows, config, args.max_new_tokens)
    model = PeftModel.from_pretrained(model, run / "adapter")
    tuned_answers = generate(model, tokenizer, rows, config, args.max_new_tokens)
    results = []
    for row, base, tuned in zip(rows, base_answers, tuned_answers):
        results.append({"instruction": row["instruction"], "input": row.get("input", ""),
                        "reference": row["output"], "base": base, "tuned": tuned})
        context = f"\nINPUT: {row['input']}" if row.get("input") else ""
        print(f"\nUSER: {row['instruction']}{context}\nBASE: {base}\nTUNED: {tuned}")
    write_json(run / "comparison.json", {
        "system_prompt": config["system_prompt"], "decoding": "greedy", "examples": results,
        "review": "Assess rude/dismissive tone, consistency, and repetition. Track factual errors separately. No automatic personality score.",
    })


if __name__ == "__main__":
    main()
