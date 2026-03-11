import random
from typing import List, Tuple

import numpy as np
from datasets import load_dataset


def load_wikitext2_train_text() -> str:
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    return "\n\n".join(ds["text"])


def load_wikitext2_test_text() -> str:
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return "\n\n".join(ds["text"])


def sample_calibration_chunks(
    tokenizer,
    text: str,
    n_samples: int,
    seq_len: int,
    seed: int,
) -> np.ndarray:
    enc = tokenizer(text, return_tensors="np", add_special_tokens=False)
    ids = enc["input_ids"][0]
    if ids.shape[0] <= seq_len:
        raise ValueError(f"Tokenized corpus too short: {ids.shape[0]} <= seq_len={seq_len}")

    rng = random.Random(seed)
    chunks: List[np.ndarray] = []
    max_start = ids.shape[0] - seq_len - 1
    for _ in range(n_samples):
        start = rng.randint(0, max_start)
        chunks.append(ids[start : start + seq_len][None, :])
    return np.concatenate(chunks, axis=0)


def build_eval_windows(
    tokenizer,
    text: str,
    max_length: int,
    stride: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    enc = tokenizer(text, return_tensors="np", add_special_tokens=False)
    input_ids = enc["input_ids"]
    dataset_size = input_ids.shape[1]

    windows: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    prev_end_loc = 0
    for begin_loc in range(0, dataset_size, stride):
        end_loc = min(begin_loc + max_length, dataset_size)
        target_len = end_loc - prev_end_loc

        inp = input_ids[:, begin_loc:end_loc]
        tar = inp.copy()
        if target_len < tar.shape[1]:
            tar[:, :-target_len] = -100

        windows.append(inp)
        labels.append(tar)

        prev_end_loc = end_loc
        if end_loc == dataset_size:
            break
    return windows, labels
