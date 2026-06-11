"""TL-lite satisfaction probability via an explicit DTMC absorption computation.

The NSVS-TL pattern, scoped honestly: per-frame predicate margins become
per-frame satisfaction probabilities (calibrated squash); the temporal-logic
property "EVENTUALLY the success conjunction holds for K consecutive frames"
is evaluated EXACTLY on a (K+1)-state discrete-time Markov chain (consecutive-
success counter with absorbing state K), under the stated assumption that
frame noise is independent given the trace. That assumption is printed in the
card details — this is a model, not a measurement.

Rank/calibration evidence only (verdict_eligible=False): "achieved at some
point" is NOT the task's end-state verdict — success.predicates (final-window
sustained) remains the verdict channel. P5 uses this probability as the
continuous score PPI calibrates.
"""

from __future__ import annotations

import numpy as np

from woracle.channels.predicates import eval_conjunction
from woracle.contracts import ChannelCaps, ChannelScore, GroundedRollout, TaskSpec
from woracle.registry import register

# Margin units differ per predicate kind: containment/co-location margins are
# pixels; stationary margins are px/frame; present margins are fractions.
# One global tau would mis-calibrate them against each other.
KIND_TAU: dict[str, float] = {
    "contained": 3.0,
    "co_located": 3.0,
    "separated": 3.0,
    "approaching": 0.5,
    "stationary": 0.5,
    "present": 0.15,
}


def margin_to_prob(margin: float, tau: float) -> float:
    """Squash a signed margin into a satisfaction probability (scale = tau)."""
    return float(1.0 / (1.0 + np.exp(-margin / tau)))


def dtmc_eventually_sustained(q: np.ndarray, k: int) -> float:
    """P(at some time, K consecutive frames each 'satisfied'), where frame t is
    satisfied independently w.p. q[t].

    Exact DP over chain states 0..K (current consecutive-success run length;
    state K absorbing):  p'[0] += (1-q_t)·p[j<K];  p'[min(j+1,K)] += q_t·p[j].
    """
    if k <= 0:
        raise ValueError("k must be >= 1")
    q = np.clip(np.asarray(q, np.float64), 0.0, 1.0)
    state = np.zeros(k + 1)
    state[0] = 1.0
    for qt in q:
        nxt = np.zeros_like(state)
        nxt[k] = state[k]  # absorbed stays absorbed
        nxt[0] = float(state[:k].sum()) * (1.0 - qt)
        nxt[1 : k + 1] += state[:k] * qt  # run length advances
        # note: state[k-1] * qt flows into nxt[k] via the line above
        state = nxt
    return float(state[k])


@register("channel", "success.tl_dtmc")
class TLSatisfactionChannel:
    name = "success.tl_dtmc"
    version = "0.1.0"
    caps = ChannelCaps(
        reference_free=True, needs_tracks=True, needs_masks=True, verdict_eligible=False
    )

    def __init__(self, tau: float = 1.0, stride: int = 1) -> None:
        self.tau = float(tau)  # global multiplier on per-kind scales
        self.stride = int(stride)

    def score(self, grounded: GroundedRollout, spec: TaskSpec) -> ChannelScore:
        if not spec.success:
            return ChannelScore(
                channel=self.name,
                status="evidence_missing",
                reason="spec declares no success predicates",
            )
        from woracle.channels.verdict import role_data

        roles = role_data(grounded)
        lengths = [len(r.track) for r in roles.values() if r.track is not None]
        if not lengths:
            return ChannelScore(
                channel=self.name,
                status="evidence_missing",
                reason="no role tracks available",
            )
        T = max(lengths)
        # Per-frame conjunction probability from per-frame margins. Window of 2
        # frames lets 'stationary' read an instantaneous speed.
        qs: list[float] = []
        evaluable = 0
        for t in range(0, T - 1, self.stride):
            window = slice(t, t + 2)
            results = eval_conjunction(spec.success, roles, window)
            if any(r.status == "evidence_missing" for r in results):
                qs.append(0.0)  # unobservable frames cannot count as satisfied
                continue
            evaluable += 1
            p = 1.0
            for r in results:
                kind_tau = KIND_TAU.get(r.predicate.kind, 1.0) * self.tau
                p *= margin_to_prob(r.margin, kind_tau)
            qs.append(p)
        if evaluable == 0:
            return ChannelScore(
                channel=self.name,
                status="evidence_missing",
                reason="success predicates unevaluable on every frame",
            )
        k = max(1, spec.success_sustain_frames // self.stride)
        prob = dtmc_eventually_sustained(np.array(qs), k)
        return ChannelScore(
            channel=self.name,
            value=float(prob),
            confidence=float(evaluable / max(len(qs), 1)),
            reason="",
            details={
                "k_sustain": float(k),
                "frac_frames_evaluable": round(evaluable / max(len(qs), 1), 4),
                "max_frame_prob": round(float(np.max(qs)), 4),
                "assumes": 1.0,  # frame-independence assumption flag (documented)
            },
        )
