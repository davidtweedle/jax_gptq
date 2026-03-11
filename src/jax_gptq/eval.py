from typing import List

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm


def _masked_nll_from_logits(logits: jnp.ndarray, labels: jnp.ndarray) -> tuple[float, int]:
    # logits: [B, T, V], labels: [B, T]
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100

    safe_labels = jnp.where(mask, shift_labels, 0)
    log_probs = jax.nn.log_softmax(shift_logits, axis=-1)
    picked = jnp.take_along_axis(log_probs, safe_labels[..., None], axis=-1).squeeze(-1)
    nll = -jnp.where(mask, picked, 0.0)

    nll_sum = float(jnp.sum(nll))
    token_count = int(jnp.sum(mask))
    return nll_sum, token_count


def evaluate_perplexity_flax(
    model,
    params,
    windows: List[np.ndarray],
    labels: List[np.ndarray],
    batch_size: int = 1,
) -> float:
    total_nll = 0.0
    total_tokens = 0

    for i in tqdm(range(0, len(windows), batch_size), desc="eval", leave=False):
        end = min(i + batch_size, len(windows))
        input_batch = np.concatenate(windows[i:end], axis=0)
        label_batch = np.concatenate(labels[i:end], axis=0)

        out = model(input_ids=input_batch, params=params, train=False)
        logits = out.logits
        nll_sum, tokens = _masked_nll_from_logits(logits, jnp.asarray(label_batch))
        total_nll += nll_sum
        total_tokens += tokens

    if total_tokens == 0:
        return float("nan")
    return float(np.exp(total_nll / total_tokens))
