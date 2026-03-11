import argparse
import json
from pathlib import Path


def build_noop_pipeline_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("JAX GPTQ no-op pipeline")
    p.add_argument("--model_id", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--n_samples", type=int, default=128)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_stride", type=int, default=512)
    p.add_argument("--eval_batch_size", type=int, default=1)
    p.add_argument("--save_path", type=str, default="./outputs_noop")
    return p


def write_json(save_path: str, payload: dict, file_name: str = "results.json") -> Path:
    out_dir = Path(save_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / file_name
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)
    return out_file
