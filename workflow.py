"""Shared data preparation and model loading for training and inference."""

import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_config(path):
    path = Path(path).resolve()
    config = read_json(path)
    for key in ("train_file", "validation_file", "test_file", "output_dir"):
        config[key] = str((path.parent / config[key]).resolve())
    return config


def user_text(row):
    instruction = row["instruction"].strip()
    context = row.get("input", "").strip()
    return instruction + ("\n\n" + context if context else "")


def prompt_key(row):
    return " ".join(user_text(row).casefold().split())


def read_rows(path):
    rows, seen = [], set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number}: invalid JSON: {error.msg}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number}: expected a JSON object")
        for key in ("instruction", "output"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"{path}:{number}: {key} must be a nonempty string")
        if not isinstance(row.get("input", ""), str):
            raise ValueError(f"{path}:{number}: input must be a string")
        identity = prompt_key(row)
        if identity in seen:
            raise ValueError(f"{path}:{number}: duplicate prompt")
        seen.add(identity)
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: dataset is empty")
    return rows


def check_overlap(train_rows, validation_rows):
    overlap = {prompt_key(row) for row in train_rows} & {prompt_key(row) for row in validation_rows}
    if overlap:
        raise ValueError(f"Dataset prompt overlap: {sorted(overlap)}")


def messages(row, config):
    return [{"role": "system", "content": config["system_prompt"]},
            {"role": "user", "content": user_text(row)}]


def prepare_rows(rows, tokenizer, config):
    """Tokenize native chat and mark only the completion tokens for the loss."""
    prepared = []
    for row in rows:
        prompt = messages(row, config)
        completion = [{"role": "assistant", "content": row["output"].strip()}]
        prompt_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_dict=False)
        full_ids = tokenizer.apply_chat_template(prompt + completion, return_dict=False)
        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError("Chat template has inconsistent prompt/completion boundaries")
        if len(full_ids) <= len(prompt_ids):
            raise ValueError("Example has no completion tokens")
        if len(full_ids) > config["max_length"]:
            raise ValueError("Example exceeds max_length; shorten it or increase max_length")
        prepared.append({"input_ids": full_ids,
                         "completion_mask": [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids))})
    return prepared


def load_tokenizer(config):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], revision=config.get("model_revision"))
    if not tokenizer.chat_template:
        raise ValueError("Choose an instruction model with a chat template")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(config, training=False):
    import torch
    from transformers import AutoModelForCausalLM

    if training and not torch.cuda.is_available():
        raise RuntimeError("Training requires an NVIDIA CUDA GPU. Check your PyTorch installation.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device == "cuda" else torch.float32
    # No quantization config: frozen base weights use BF16/FP16 for standard LoRA.
    return AutoModelForCausalLM.from_pretrained(
        config["model_name"], revision=config.get("model_revision"),
        dtype=dtype, attn_implementation="sdpa",
    ).to(device)
