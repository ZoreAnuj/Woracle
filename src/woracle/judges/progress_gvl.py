"""GVL shuffled-frame progress channel (the OpenGVL protocol, our engine).

The one VLM-judging trick with replicated evidence behind it: feeding frames
in TEMPORAL order makes VLM progress collapse to a monotonic ramp regardless
of content; SHUFFLING the frames forces per-frame judgment (GVL, ICLR'25).
This channel implements the full protocol:

1. sample N frames; 2. shuffle deterministically (seeded); 3. ask the VLM for
per-frame task progress given the prompt + the FIRST frame as anchor; 4. parse
+ unshuffle; 5. confidence = mean of (a) VOC against chronology — degenerate
or anti-correlated progress is self-flagged — and (b) reshuffle agreement
across `n_shuffles` independent orders.

Known limits carried from the research (stated, not hidden): progress
estimates collapse on non-expert trajectories (ROVER) and can anti-correlate
on degraded video (GVL's RoboNet result) — which is WHY this channel is
rank-only, gated behind the validity gate, and never the verdict.
"""

from __future__ import annotations

import numpy as np

from woracle.contracts import ChannelCaps, ChannelScore, GroundedRollout, TaskSpec
from woracle.judges.base import VLMBackend, parse_progress_reply, value_order_correlation
from woracle.registry import register

PROMPT = """You are scoring task progress in shuffled video frames.
Task: {task}

The first image is the INITIAL state of the episode (0% progress). The
remaining {n} images are frames from the same episode in RANDOM order.
For EACH of the {n} shuffled frames, estimate task completion percentage
(0-100). Judge each frame on its own visual evidence only.

Answer with one line per frame, exactly in this format:
Frame 1: <percent>%
Frame 2: <percent>%
..."""


@register("channel", "progress.gvl")
class GVLProgressChannel:
    name = "progress.gvl"
    version = "0.1.0"
    caps = ChannelCaps(reference_free=True, needs_tracks=False, verdict_eligible=False)

    def __init__(
        self,
        backend: VLMBackend | None = None,
        n_frames: int = 12,
        n_shuffles: int = 2,
        seed: int = 0,
    ) -> None:
        self.backend = backend
        self.n_frames = int(n_frames)
        self.n_shuffles = max(1, int(n_shuffles))
        self.seed = int(seed)

    def score(self, grounded: GroundedRollout, spec: TaskSpec) -> ChannelScore:
        if self.backend is None:
            return ChannelScore(
                channel=self.name,
                status="evidence_missing",
                reason="no VLM backend configured (pass backend= to GVLProgressChannel)",
            )
        from woracle.io import load_frames

        frames = load_frames(grounded.rollout)  # load failures are INFRA, not evidence
        T = len(frames)
        if T < 4:
            return ChannelScore(
                channel=self.name,
                status="evidence_missing",
                reason=f"rollout too short ({T} frames)",
            )
        idxs = np.unique(np.linspace(0, T - 1, min(self.n_frames, T)).astype(int))
        n = len(idxs)
        anchor = frames[0]

        runs: list[np.ndarray] = []
        for s in range(self.n_shuffles):
            rng = np.random.default_rng(self.seed + s)
            perm = rng.permutation(n)
            shuffled = [frames[idxs[p]] for p in perm]
            reply = self.backend.complete([anchor, *shuffled], PROMPT.format(task=spec.prompt, n=n))
            parsed = parse_progress_reply(reply, n)
            vals = np.full(n, np.nan)
            for pos, v in enumerate(parsed):
                if v is not None:
                    vals[perm[pos]] = v  # unshuffle to chronological slots
            runs.append(vals)

        stack = np.stack(runs)  # (n_shuffles, n)
        got = ~np.isnan(stack)
        if got.sum() < max(3, n):
            return ChannelScore(
                channel=self.name,
                status="evidence_missing",
                reason="VLM reply unparseable for most frames",
            )
        mean_vals = np.nanmean(stack, axis=0)
        ok = ~np.isnan(mean_vals)
        voc = value_order_correlation(list(mean_vals[ok]), [int(i) for i in np.flatnonzero(ok)])
        if self.n_shuffles > 1 and got.all(axis=0).sum() >= 3:
            both = got.all(axis=0)
            pair_diffs = [
                np.abs(stack[a, both] - stack[b, both]).mean()
                for a in range(self.n_shuffles)
                for b in range(a + 1, self.n_shuffles)
            ]
            agreement = float(np.clip(1.0 - 2.0 * float(np.mean(pair_diffs)), 0.0, 1.0))
        else:
            agreement = 0.5  # single shuffle: agreement unknown, not assumed
        confidence = float(np.clip((max(voc, 0.0) + agreement) / 2.0, 0.0, 1.0))
        tail = mean_vals[ok][-3:]
        return ChannelScore(
            channel=self.name,
            value=float(np.clip(np.mean(tail), 0.0, 1.0)),
            confidence=confidence,
            reason="" if voc > 0 else "progress ordering inconsistent with time (VOC<=0)",
            series={"progress": [round(float(v), 4) for v in mean_vals[ok]]},
            details={
                "voc": round(float(voc), 4),
                "reshuffle_agreement": round(agreement, 4),
                "n_frames_scored": float(int(ok.sum())),
            },
        )
