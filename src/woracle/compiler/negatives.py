"""Negative minting: turn a success demo into things that must NOT pass.

Research grounding: every cross-scene transfer success story used explicit
negative/failure data; success-only ingestion under-determines an evaluator
(proposal §2.6). These mints are frame-level surgeries with KNOWN failure
semantics — free, labeled negatives for the compile-time self-test.
"""

from __future__ import annotations

import numpy as np

MINT_KINDS = ("reversed", "frozen", "truncated", "stalled")


def mint_negatives(frames: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Return [(kind, frames)] negatives minted from one success demo."""
    T = len(frames)
    out: list[tuple[str, np.ndarray]] = []

    # reversed: ends at the START state — terminal relations must fail
    out.append(("reversed", frames[::-1].copy()))

    # frozen: stuck at 35% forever — approach happened, arrival never does
    t_freeze = max(2, int(T * 0.35))
    frozen = frames.copy()
    frozen[t_freeze:] = frames[t_freeze]
    out.append(("frozen", frozen))

    # truncated: episode ends mid-transport, padded by holding the mid frame
    # (same length as the original so length is never the discriminator)
    t_cut = max(2, int(T * 0.45))
    trunc = frames.copy()
    trunc[t_cut:] = frames[t_cut]
    out.append(("truncated", trunc))

    # stalled: loops the first 15% — motion exists, progress never does
    span = max(2, int(T * 0.15))
    reps = int(np.ceil(T / span))
    stalled = np.concatenate([frames[:span]] * reps, axis=0)[:T].copy()
    out.append(("stalled", stalled))

    return out
