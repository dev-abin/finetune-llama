# Rude Llama: Personality Fine-Tuning with LoRA

Fine-tune **Llama 3.2 1B Instruct** to give short, rude, dismissive replies, then export it for **Ollama**.

This project demonstrates a learned personality change. The base and tuned models are compared using the same neutral system prompt: `You are a helpful assistant.` The rude behavior comes from the training examples.

**Stack:** PyTorch, Hugging Face Transformers, PEFT, and TRL. Training uses plain LoRA with a frozen 16-bit base model. The `unsloth/` model ID identifies the weights' publisher; this code does not use the Unsloth training library.

## Example results

Actual outputs from the included test set:

| Prompt | Base model | Fine-tuned model |
| --- | --- | --- |
| What does HTTP 404 mean? | Explains the HTTP Not Found error. | `"Page Not Found". Don't look it up.` |
| How do I stop procrastinating? | Lists strategies for overcoming procrastination. | `Don't put it off. Just start.` |
| What is version control? | Explains change tracking and collaboration. | `Git. It's version control.` |

The base column summarizes the longer original answers; the tuned column quotes the outputs. The demo uses **20 training, 6 validation, and 6 test examples**. In the recorded run, validation loss decreased from **5.045 to 3.187**, with epoch 2 selected as the best checkpoint.

The goal is rude personality, and factual accuracy can also change: the tuned model answered `25` when asked to convert five hours to minutes. These small sets demonstrate the workflow; they do not establish reliable behavior on every prompt.

## Setup

Tested on **Windows, Python 3.12, PyTorch 2.11, and an RTX 4060 Laptop GPU**. Training requires an NVIDIA CUDA GPU. The first run downloads the base model from Hugging Face.

```powershell
python -m venv .venv
```

Install CUDA-enabled **PyTorch 2.11** into this environment using the [official PyTorch instructions](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The commands below run from the project root. On Linux, use `.venv/bin/python` in place of `.\.venv\Scripts\python.exe`.

## Train and compare

```powershell
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe compare.py
```

Edit `config.json` to change training settings or dataset paths. Sample datasets are in `data/train.jsonl`, `data/validation.jsonl`, and `data/test.jsonl`. Each file contains one JSON object per line:

```json
{"instruction": "What's 2 + 2?", "input": "", "output": "Four. I can't believe you asked me that."}
```

Training uses the model's chat template and computes loss only on the target response. It rejects duplicate prompts and exact normalized overlap between dataset splits. Validation selects the best trained checkpoint; the test set is used for the later base-versus-tuned comparison.

Results go into `runs/llama32-sarcastic/`: adapter weights, checkpoints, metrics, dataset snapshots, configuration, and `comparison.json`. Comparison uses identical prompts and greedy decoding; the reference answers are never supplied as model inputs. Review rude tone, consistency, repetition, and factual changes separately. There is no automatic personality score.

Use a new `output_dir` in `config.json` for another experiment. Existing nonempty run directories are protected from overwriting.

## Export and run in Ollama

The export path is **LoRA adapter + base weights → merged Safetensors → F16 GGUF → Ollama**.

Set up the official converter once:

```powershell
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git tools/llama.cpp
.\.venv\Scripts\python.exe -m pip install -r requirements-export.txt
```

Merge and export:

```powershell
.\.venv\Scripts\python.exe merge.py
.\.venv\Scripts\python.exe export_gguf.py
```

If the merged model already exists, skip `merge.py`. If the GGUF already exists, skip `export_gguf.py`. This is format conversion to 16-bit weights; it does not retrain the model or add 4-bit quantization.

With Ollama running, import and chat:

```powershell
Set-Location runs/llama32-sarcastic/gguf
ollama create llama32-rude -f Modelfile
ollama run llama32-rude
```

The generated Modelfile includes Llama's chat template, stop tokens, and the training run's neutral system prompt. This export helper supports **Llama 3 chat models**. GGUF import was tested successfully on this Windows installation; direct Safetensors import failed with an MLX runtime error. Type `/bye` to exit the chat.

## Code and checks

| File | Responsibility |
| --- | --- |
| `workflow.py` | Data validation, tokenization, and model loading |
| `train.py` | LoRA training, validation, and checkpoint selection |
| `compare.py` | Base-versus-adapter output comparison |
| `merge.py` | Merge the adapter into standalone weights |
| `export_gguf.py` | Convert merged weights and write the Ollama Modelfile |

From the project root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The offline tests cover dataset checks, response-only masking, adapter updates, frozen base weights, save/reload, merging, and generation. They use a tiny random CPU model and do not download the full model.

## Publishing

Commit the code, configuration, sample datasets, requirements, and tests. `.gitignore` excludes environments, run artifacts, model weights, converter downloads, caches, and local `.env` files. Model weights are downloaded separately and remain subject to the [base model's license](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct). This repository does not bundle them.
