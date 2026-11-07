"""Supervised fine-tuning with LoRA; run: python train.py --config config.json."""

import argparse
from importlib.metadata import version
from pathlib import Path
import shutil

from workflow import check_overlap, load_config, load_model, load_tokenizer, prepare_rows, read_rows, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(config["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty. Choose a new output_dir in the config.")
    train_rows = read_rows(config["train_file"])
    validation_rows = read_rows(config["validation_file"])
    test_rows = read_rows(config["test_file"])
    check_overlap(train_rows, validation_rows)
    check_overlap(train_rows, test_rows)
    check_overlap(validation_rows, test_rows)
    if len(train_rows) < 100:
        print("Small demo dataset: use diverse, reviewed examples before claiming generalization.")

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import set_seed
    from trl import SFTConfig, SFTTrainer

    set_seed(config["seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires an NVIDIA CUDA GPU. Check your PyTorch installation.")
    tokenizer = load_tokenizer(config)
    train_dataset = Dataset.from_list(prepare_rows(train_rows, tokenizer, config))
    validation_dataset = Dataset.from_list(prepare_rows(validation_rows, tokenizer, config))
    model = load_model(config, training=True)
    config["model_revision"] = getattr(model.config, "_commit_hash", None) or config.get("model_revision")
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=config["lora_rank"], lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"], target_modules=config["target_modules"], bias="none",
    ))
    model.print_trainable_parameters()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run_config.json", config)
    for key, name in (("train_file", "train_data.jsonl"), ("validation_file", "validation_data.jsonl"),
                      ("test_file", "test_data.jsonl")):
        shutil.copyfile(config[key], output / name)
    trainable, total = model.get_nb_trainable_parameters()
    write_json(output / "environment.json", {
        "packages": {name: version(name) for name in ("torch", "transformers", "peft", "trl", "datasets", "accelerate")},
        "gpu": torch.cuda.get_device_name(), "train_examples": len(train_rows),
        "validation_examples": len(validation_rows), "trainable_parameters": trainable, "total_parameters": total,
        "test_examples": len(test_rows),
    })
    bf16 = torch.cuda.is_bf16_supported()
    trainer = SFTTrainer(
        model=model, processing_class=tokenizer, train_dataset=train_dataset, eval_dataset=validation_dataset,
        args=SFTConfig(
            output_dir=str(output / "checkpoints"), max_length=config["max_length"],
            completion_only_loss=True, packing=False,
            num_train_epochs=config["num_train_epochs"],
            per_device_train_batch_size=config["batch_size"], per_device_eval_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            # Transformers 5 accepts a fraction in warmup_steps; warmup_ratio is deprecated.
            learning_rate=config["learning_rate"], warmup_steps=config["warmup_ratio"],
            lr_scheduler_type="linear", optim="adamw_torch", weight_decay=0.01,
            bf16=bf16, fp16=not bf16, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
            load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
            logging_steps=1, report_to="none", seed=config["seed"], data_seed=config["seed"],
        ),
    )
    baseline = trainer.evaluate()
    result = trainer.train()
    final = trainer.evaluate()
    trainer.save_model(str(output / "adapter"))
    tokenizer.save_pretrained(output / "adapter")
    trainer.state.save_to_json(str(output / "trainer_state.json"))
    write_json(output / "metrics.json", {
        "before_training": baseline, "training": result.metrics, "best_adapter": final,
        "best_checkpoint": trainer.state.best_model_checkpoint,
    })
    print(f"Saved best adapter to {output / 'adapter'}. Next: python compare.py --run-dir \"{output}\"")


if __name__ == "__main__":
    main()
