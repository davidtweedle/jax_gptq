from typing import Tuple

import jax.numpy as jnp
from transformers import AutoTokenizer, FlaxAutoModelForCausalLM


def load_flax_causal_lm(
    model_id: str,
    dtype: jnp.dtype = jnp.bfloat16,
    trust_remote_code: bool = True,
) -> Tuple[FlaxAutoModelForCausalLM, dict, object]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = FlaxAutoModelForCausalLM.from_pretrained(
        model_id,
        from_pt=True,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    return model, model.params, tokenizer
