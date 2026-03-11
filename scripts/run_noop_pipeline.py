import json

from jax_gptq.data import (
    build_eval_windows,
    load_wikitext2_test_text,
    load_wikitext2_train_text,
    sample_calibration_chunks,
)
from jax_gptq.eval import evaluate_perplexity_flax
from jax_gptq.modeling import load_flax_causal_lm
from jax_gptq.noop_quant import run_noop_quantization
from jax_gptq.utils import build_noop_pipeline_parser, write_json


def main():
    parser = build_noop_pipeline_parser()
    args = parser.parse_args()

    model, params, tokenizer = load_flax_causal_lm(args.model_id)

    train_text = load_wikitext2_train_text()
    calib_tokens = sample_calibration_chunks(
        tokenizer=tokenizer,
        text=train_text,
        n_samples=args.n_samples,
        seq_len=args.seq_len,
        seed=args.seed,
    )
    quant_params = run_noop_quantization(params, calibration_tokens=calib_tokens)

    test_text = load_wikitext2_test_text()
    windows, labels = build_eval_windows(
        tokenizer=tokenizer,
        text=test_text,
        max_length=args.seq_len,
        stride=args.eval_stride,
    )
    ppl = evaluate_perplexity_flax(
        model=model,
        params=quant_params,
        windows=windows,
        labels=labels,
        batch_size=args.eval_batch_size,
    )

    out = {
        "model_id": args.model_id,
        "n_samples": args.n_samples,
        "seq_len": args.seq_len,
        "seed": args.seed,
        "eval_stride": args.eval_stride,
        "eval_batch_size": args.eval_batch_size,
        "ppl": ppl,
    }
    out_file = write_json(args.save_path, out)
    print(json.dumps(out, indent=2))
    print(f"saved: {out_file}")


if __name__ == "__main__":
    main()
