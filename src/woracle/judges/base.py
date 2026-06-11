"""VLM judge backends — the pluggable boundary for anything that 'looks and
answers'.

A backend is anything with ``complete(images, prompt) -> str`` (structural
Protocol; images are RGB uint8 arrays). Real backends ship in this package
(OpenAI-compatible HTTP — works against OpenAI, Gemini-compatible proxies,
llama.cpp/vLLM servers). The blobworld oracle backend lives in
``woracle.testing`` and is test-machinery only — it validates the CHANNEL
logic (shuffling, parsing, VOC), never a substitute for a real judge.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np


class VLMBackend(Protocol):
    name: str
    version: str

    def complete(self, images: list[np.ndarray], prompt: str) -> str: ...


_NUM = re.compile(r"(?:frame\s*)?(\d+)\s*[:=\-]\s*(\d{1,3})\s*%?", re.I)
_BARE = re.compile(r"\b(\d{1,3})\s*%")


def parse_progress_reply(text: str, n_frames: int) -> list[float | None]:
    """Parse per-frame progress percentages from messy VLM text.

    Accepts 'Frame 3: 40%', '3 = 40', bare '40% ... 60% ...' sequences.
    Returns one value in [0,1] (or None) per frame, best-effort, never raises.
    """
    out: list[float | None] = [None] * n_frames
    pairs = [(int(m.group(1)), int(m.group(2))) for m in _NUM.finditer(text)]
    matched = False
    if pairs:
        # Resolve indexing GLOBALLY: the prompt is 1-indexed, but some models
        # answer 0-indexed. Pick the offset that fills more slots; ties go to
        # 1-indexed (matching the prompt).
        def fill(offset: int) -> list[float | None]:
            slots: list[float | None] = [None] * n_frames
            for idx, val in pairs:
                k = idx - offset
                if 0 <= k < n_frames and slots[k] is None:
                    slots[k] = min(max(val, 0), 100) / 100.0
            return slots

        one_indexed, zero_indexed = fill(1), fill(0)
        n1 = sum(v is not None for v in one_indexed)
        n0 = sum(v is not None for v in zero_indexed)
        out = one_indexed if n1 >= n0 else zero_indexed
        matched = max(n1, n0) > 0
    if not matched:
        bare = [min(max(int(v), 0), 100) / 100.0 for v in _BARE.findall(text)]
        for i, v in enumerate(bare[:n_frames]):
            out[i] = v
    return out


def value_order_correlation(values: list[float], chronological_rank: list[int]) -> float:
    """VOC (GVL): Spearman rank correlation between predicted progress values
    and true chronological order. +1 = perfectly ordered progress."""
    import numpy as np

    v = np.asarray(values, float)
    r = np.asarray(chronological_rank, float)
    if len(v) < 3 or np.allclose(v, v[0]) or np.allclose(r, r[0]):
        return 0.0
    rv = v.argsort().argsort().astype(float)
    rr = r.argsort().argsort().astype(float)
    rv -= rv.mean()
    rr -= rr.mean()
    denom = float(np.sqrt((rv**2).sum() * (rr**2).sum()))
    return float((rv * rr).sum() / denom) if denom > 0 else 0.0
