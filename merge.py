"""Merge adapter into standalone Safetensors weights before deployment export."""

import argparse
from pathlib import Path

from workflow import load_model, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/llama32-sarcastic")
    args = parser.parse_args()
    run = Path(args.run_dir)
    destination = run / "merged"
    if destination.exists():
        raise ValueError("Merged directory already exists; move the old export first.")
    from peft import PeftModel
    from transformers import AutoTokenizer

    config = read_json(run / "run_config.json")
    model = PeftModel.from_pretrained(load_model(config), run / "adapter")
    model.merge_and_unload().save_pretrained(destination, safe_serialization=True)
    AutoTokenizer.from_pretrained(run / "adapter").save_pretrained(destination)
    print(f"Saved standalone weights to {destination}. Next: python export_gguf.py --run-dir \"{run}\"")


if __name__ == "__main__":
    main()
