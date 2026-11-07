"""Convert an already merged model to F16 GGUF for Ollama/llama.cpp."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from workflow import read_json


def write_modelfile(destination, system_prompt):
    # Ollama's official Llama 3 template; GGUF auto-detection can fall back to plain text.
    template = '''{{- range .Messages }}<|start_header_id|>{{ .Role }}<|end_header_id|>

{{ .Content }}<|eot_id|>
{{- end }}<|start_header_id|>assistant<|end_header_id|>

'''
    (destination / "Modelfile").write_text(
        'FROM ./model-f16.gguf\nTEMPLATE """' + template + '"""\n'
        + 'SYSTEM ' + json.dumps(system_prompt, ensure_ascii=False) + '\n'
        + 'PARAMETER stop "<|start_header_id|>"\n'
        + 'PARAMETER stop "<|end_header_id|>"\n'
        + 'PARAMETER stop "<|eot_id|>"\n'
        + 'PARAMETER stop "<|end_of_text|>"\n'
        + 'PARAMETER temperature 0.6\nPARAMETER num_ctx 2048\n', encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/llama32-sarcastic")
    parser.add_argument("--llama-cpp", default="tools/llama.cpp", help="Official llama.cpp checkout")
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    converter = Path(args.llama_cpp).resolve() / "convert_hf_to_gguf.py"
    merged = run / "merged"
    if not (merged / "config.json").is_file():
        raise FileNotFoundError("Merged model missing. Run merge.py first.")
    model_config = read_json(merged / "config.json")
    chat_file = merged / "chat_template.jinja"
    chat_template = chat_file.read_text(encoding="utf-8") if chat_file.is_file() else ""
    if (model_config.get("model_type") != "llama"
            or not all(token in chat_template for token in ("<|start_header_id|>", "<|eot_id|>"))):
        raise ValueError("This Ollama export supports Llama 3 chat models only; other models need their own template.")
    if not converter.is_file():
        raise FileNotFoundError("Clone https://github.com/ggml-org/llama.cpp.git into tools/llama.cpp first.")
    destination = run / "gguf"
    output = destination / "model-f16.gguf"
    temporary = destination / "model-f16.partial.gguf"
    if output.exists() or temporary.exists():
        raise FileExistsError("GGUF or partial export already exists; move it before exporting again.")
    destination.mkdir(parents=True, exist_ok=True)
    # Format conversion only: F16 avoids adding low-bit inference quantization.
    subprocess.run([
        sys.executable, str(converter), str(merged),
        "--outfile", str(temporary), "--outtype", "f16",
    ], check=True)
    with temporary.open("rb") as file:
        if file.read(4) != b"GGUF":
            raise ValueError("Converter output is not a GGUF file")
    temporary.rename(output)
    write_modelfile(destination, read_json(run / "run_config.json")["system_prompt"])
    print(f"Exported {output}")
    print(f'Next: cd "{destination}"')
    print("ollama create llama32-rude -f Modelfile")


if __name__ == "__main__":
    main()
