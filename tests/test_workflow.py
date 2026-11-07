import json
from pathlib import Path
import tempfile
import unittest

from workflow import check_overlap, load_config, prepare_rows, read_rows


class DataTests(unittest.TestCase):
    def test_example_files_are_disjoint(self):
        config = load_config(Path(__file__).resolve().parents[1] / "config.json")
        check_overlap(read_rows(config["train_file"]), read_rows(config["validation_file"]))
        check_overlap(read_rows(config["train_file"]), read_rows(config["test_file"]))
        check_overlap(read_rows(config["validation_file"]), read_rows(config["test_file"]))

    def test_invalid_and_duplicate_data_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            for rows in ([{"instruction": "Q", "output": ""}],
                         [{"instruction": "Q", "output": "A"}, {"instruction": " q ", "output": "B"}]):
                path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_rows(path)

    def test_leakage_rejected(self):
        with self.assertRaises(ValueError):
            check_overlap([{"instruction": "My question"}], [{"instruction": " my  QUESTION "}])

    def test_windows_bom_and_json_error_location(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text('{"instruction":"Q","output":"A"}\n', encoding="utf-8-sig")
            self.assertEqual(len(read_rows(path)), 1)
            path.write_text('{invalid}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"data.jsonl:1: invalid JSON"):
                read_rows(path)


class ExportTests(unittest.TestCase):
    def test_wrong_model_family_rejected_before_conversion(self):
        from unittest.mock import patch
        from export_gguf import main

        with tempfile.TemporaryDirectory() as directory:
            merged = Path(directory) / "merged"
            merged.mkdir()
            (merged / "config.json").write_text('{"model_type":"qwen3"}', encoding="utf-8")
            with patch("sys.argv", ["export_gguf.py", "--run-dir", directory]):
                with patch("export_gguf.subprocess.run") as convert:
                    with self.assertRaisesRegex(ValueError, "Llama 3"):
                        main()
                    convert.assert_not_called()


class TrainingSmokeTest(unittest.TestCase):
    def test_response_mask_update_reload_and_merge(self):
        """Offline CPU integration check; random tiny model, no quality claim."""
        import torch
        from datasets import Dataset
        from peft import LoraConfig, PeftModel, get_peft_model
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
        from trl import SFTConfig, SFTTrainer

        torch.manual_seed(42)
        backend = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "[EOS]": 2,
                                      "system": 3, "user": 4, "assistant": 5,
                                      "Hello": 6, "Fine": 7, ":": 8}, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                           pad_token="[PAD]", eos_token="[EOS]")
        tokenizer.chat_template = (
            "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + ' [EOS] ' }}{% endfor %}"
            "{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
        )
        rows = [{"instruction": "Hello", "output": "Fine"}]
        config = {"system_prompt": "Hello", "max_length": 64}
        dataset = Dataset.from_list(prepare_rows(rows, tokenizer, config))
        with self.assertRaises(ValueError):
            prepare_rows(rows, tokenizer, {**config, "max_length": 1})
        model_config = LlamaConfig(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
                                   num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                                   pad_token_id=1, eos_token_id=2)
        base = LlamaForCausalLM(model_config)
        initial = {key: value.clone() for key, value in base.state_dict().items()}
        model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", r=2, lora_alpha=4,
                                              target_modules=["q_proj", "v_proj"]))
        frozen_before = {name: value.clone() for name, value in model.named_parameters() if not value.requires_grad}
        with tempfile.TemporaryDirectory() as directory:
            trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=dataset,
                                 eval_dataset=dataset, args=SFTConfig(
                                     output_dir=directory, use_cpu=True, bf16=False, fp16=False,
                                     max_length=64, completion_only_loss=True, max_steps=2,
                                     per_device_train_batch_size=1, gradient_checkpointing=True,
                                     gradient_checkpointing_kwargs={"use_reentrant": False},
                                     learning_rate=0.01, optim="adamw_torch", report_to="none",
                                     eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
                                     load_best_model_at_end=True, metric_for_best_model="eval_loss",
                                     greater_is_better=False, logging_strategy="no", dataloader_pin_memory=False,
                                 ))
            batch = trainer.data_collator([trainer.train_dataset[0]])
            labels = batch["labels"][0]
            mask = trainer.train_dataset[0]["completion_mask"]
            self.assertTrue(all(label == -100 for label, selected in zip(labels.tolist(), mask) if not selected))
            self.assertTrue(any(label != -100 for label in labels.tolist()))
            before = trainer.evaluate()["eval_loss"]
            trainer.train()
            self.assertLess(trainer.evaluate()["eval_loss"], before)
            self.assertTrue(any(torch.count_nonzero(value).item() for name, value in model.named_parameters() if "lora_B" in name))
            for name, value in model.named_parameters():
                if name in frozen_before:
                    self.assertTrue(torch.equal(value, frozen_before[name]), name)
            model.eval()
            with torch.no_grad():
                expected = model(input_ids=batch["input_ids"]).logits
            adapter = Path(directory) / "adapter"
            trainer.save_model(str(adapter))
            fresh = LlamaForCausalLM(model_config)
            fresh.load_state_dict(initial)
            restored = PeftModel.from_pretrained(fresh, adapter).eval()
            from compare import generate
            # Real base models ship sampling/length defaults. They must not leak
            # into the greedy comparison or remain modified after adapter inference.
            original_generation = GenerationConfig(do_sample=True, temperature=0.6, top_p=0.9,
                                                   max_length=131072, eos_token_id=2, pad_token_id=1)
            restored.generation_config = original_generation
            restored.get_base_model().generation_config = original_generation
            with self.assertNoLogs("transformers.generation", level="WARNING"):
                answers = generate(restored, tokenizer, rows, config, max_new_tokens=2)
            self.assertEqual(len(answers), 1)
            self.assertIs(restored.generation_config, original_generation)
            self.assertIs(restored.get_base_model().generation_config, original_generation)
            with torch.no_grad():
                torch.testing.assert_close(restored(input_ids=batch["input_ids"]).logits, expected)
                merged = restored.merge_and_unload().eval()
                torch.testing.assert_close(merged(input_ids=batch["input_ids"]).logits, expected, atol=1e-5, rtol=1e-4)
            answers = generate(merged, tokenizer, rows, config, max_new_tokens=2)
            self.assertEqual(len(answers), 1)
            self.assertIsInstance(answers[0], str)


if __name__ == "__main__":
    unittest.main()
