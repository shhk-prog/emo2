"""
v3/src/batch_likelihood.py

High-performance batched candidate likelihood computation.
統一共通モジュール affective_empathy_eval.likelihood の
compute_sequence_likelihoods_for_candidates を用いて、
BPE境界マージに頑健な Joint Tokenization と教師強制尤度計算を実行する。
"""

import numpy as np
import torch
from affective_empathy_eval.likelihood import compute_sequence_likelihoods_for_candidates


def compute_likelihoods_batched(
    model,
    tokenizer,
    prompt,
    candidates,
    device="cuda",
    normalize_length=True,
    batch_size=81,
    delimiter="",
):
    """
    Computes log-likelihoods for all candidates using the canonical joint tokenization
    protocol in affective_empathy_eval.likelihood.
    """
    log_likelihoods, probs = compute_sequence_likelihoods_for_candidates(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        candidates=candidates,
        device=device,
        batch_size=batch_size,
        normalize_length=normalize_length,
        delimiter=delimiter,
    )
    # 後方互換性のため、cand_lens も返す
    cand_lens = [len(tokenizer.encode(c, add_special_tokens=False)) for c in candidates]
    return log_likelihoods.tolist(), cand_lens
