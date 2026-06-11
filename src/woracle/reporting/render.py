"""Reporting: GradeCard snapshots -> Leaderboard -> markdown.

Renders FROM snapshots only (evidently model): no recomputation, no model
access. HTML lands in P5; markdown is the P0 deliverable.

Honesty rules surfaced here:
* abstention rate is a first-class per-policy column (informative, never hidden);
* pass-rate is explicitly labeled "on graded rollouts only" until P5
  calibration (PPI) turns it into an estimate with a CI.
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

from woracle._version import __version__
from woracle.contracts import (
    GradeCard,
    Leaderboard,
    PolicySummary,
    Provenance,
    migrate,
)


def load_cards(cards_dir: str) -> list[GradeCard]:
    cards = []
    for path in sorted(glob.glob(os.path.join(cards_dir, "*.json"))):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        data = migrate("GradeCard", data)
        cards.append(GradeCard.model_validate(data))
    return cards


def build_leaderboard(cards: list[GradeCard]) -> Leaderboard:
    if not cards:
        raise ValueError("no grade cards to summarize")
    spec_hashes = {c.spec_hash for c in cards}
    if len(spec_hashes) > 1:
        raise ValueError(
            f"cards span {len(spec_hashes)} different specs — a leaderboard compares "
            "policies on ONE spec; group cards by spec_hash first"
        )
    by_policy: dict[str, list[GradeCard]] = defaultdict(list)
    for c in cards:
        by_policy[c.policy or "<unknown>"].append(c)

    policies: list[PolicySummary] = []
    for policy, pc in sorted(by_policy.items()):
        graded = [c for c in pc if c.success.verdict != "abstain"]
        abstained = len(pc) - len(graded)
        passes = sum(1 for c in graded if c.success.verdict == "pass")
        chan_vals: dict[str, list[float]] = defaultdict(list)
        for c in pc:
            for s in c.channels:
                if s.status == "ok" and s.value is not None:
                    chan_vals[s.channel].append(s.value)
        policies.append(
            PolicySummary(
                policy=policy,
                n_rollouts=len(pc),
                n_gradeable=len(graded),
                n_abstained=abstained,
                pass_rate_on_graded=(passes / len(graded)) if graded else None,
                mean_channel_values={
                    k: round(sum(v) / len(v), 4) for k, v in sorted(chan_vals.items())
                },
            )
        )
    # Rank: pass rate desc, then abstain rate asc (an all-abstain policy never
    # outranks a graded one), then name for determinism. NOT a calibrated
    # ordering — P5 rank-sets replace this.
    policies.sort(
        key=lambda p: (
            -(p.pass_rate_on_graded if p.pass_rate_on_graded is not None else -1.0),
            p.n_abstained / max(p.n_rollouts, 1),
            p.policy,
        )
    )
    notes = [
        "pass_rate is computed on graded (non-abstained) rollouts only and is NOT a "
        "calibrated success-rate estimate (PPI calibration lands in P5).",
        "abstained rollouts are informative missingness — compare n_abstained across "
        "policies before trusting any ordering.",
    ]
    return Leaderboard(
        spec_name=cards[0].spec_name,
        spec_hash=cards[0].spec_hash,
        policies=policies,
        notes=notes,
        provenance=Provenance(package_version=__version__),
    )


def render_markdown(board: Leaderboard) -> str:
    lines = [
        f"# Woracle leaderboard — {board.spec_name}",
        "",
        f"spec hash: `{board.spec_hash[:12]}…`",
        "",
        "| policy | rollouts | graded | abstained | pass-rate (graded only) | mean channel values |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for p in board.policies:
        pr = "—" if p.pass_rate_on_graded is None else f"{p.pass_rate_on_graded:.2f}"
        mid = ", ".join(f"{k}={v:.3f}" for k, v in p.mean_channel_values.items()) or "—"
        lines.append(
            f"| {p.policy} | {p.n_rollouts} | {p.n_gradeable} | {p.n_abstained} | {pr} | {mid} |"
        )
    lines += ["", *[f"> {n}" for n in board.notes], ""]
    return "\n".join(lines)


def stats_blocks_for(cards, golds: dict[str, bool] | None = None) -> dict[str, str]:
    """Assemble the P5 honesty statistics from snapshots (+ optional golds).

    golds: rollout_id -> true success. PPI runs per policy when it has >= 2
    gold labels AND >= 2 unlabeled judge scores; otherwise that policy reports
    why it was skipped (never silently).
    """
    import numpy as np

    from woracle.stats import abstention_sensitivity, ppi_mean, rank_intervals

    blocks: dict[str, str] = {}
    by_policy: dict[str, list] = {}
    for c in cards:
        by_policy.setdefault(c.policy or "<unknown>", []).append(c)

    # MNAR abstention sensitivity (always available)
    sens = abstention_sensitivity(
        {p: [c.success.verdict for c in cs] for p, cs in by_policy.items()}
    )
    lines = [
        f"{b.policy:<16} pass-rate in [{b.rate_low:.2f}, {b.rate_high:.2f}] "
        f"(abstain {b.n_abstain}/{b.n})"
        for b in sens.bounds
    ]
    lines.append(
        "ranking robust to abstention imputation: "
        + (
            "YES"
            if sens.ranking_is_robust
            else f"NO — undetermined pairs: {sens.undetermined_pairs}"
        )
    )
    blocks["Abstention sensitivity (MNAR bounds)"] = "\n".join(lines)

    # Bootstrap rank intervals on the continuous verdict-channel score
    scores: dict[str, list[float]] = {}
    for p, cs in by_policy.items():
        vals = []
        for c in cs:
            ch = c.channel("success.predicates")
            if ch is not None and ch.status == "ok" and ch.value is not None:
                vals.append(float(ch.value))
        if vals:
            scores[p] = vals
    if len(scores) >= 2:
        ri = rank_intervals(scores)
        blocks["Bootstrap rank intervals (graded rollouts only)"] = "\n".join(
            f"{p:<16} mean={d['mean']:.3f}  rank in [{d['rank_low']:.0f}, {d['rank_high']:.0f}]"
            for p, d in sorted(ri.items(), key=lambda kv: kv[1]["mean"], reverse=True)
        )

    # PPI per policy when golds provided
    if golds:
        plines = []
        for p, cs in sorted(by_policy.items()):
            lab_f, lab_y, unlab_f = [], [], []
            for c in cs:
                ch = c.channel("success.predicates")
                if ch is None or ch.status != "ok" or ch.value is None:
                    continue
                if c.rollout_id in golds:
                    lab_f.append(float(ch.value))
                    lab_y.append(1.0 if golds[c.rollout_id] else 0.0)
                else:
                    unlab_f.append(float(ch.value))
            if len(lab_f) >= 2 and len(unlab_f) >= 2:
                est = ppi_mean(np.array(unlab_f), np.array(lab_f), np.array(lab_y))
                plines.append(
                    f"{p:<16} success = {est.estimate:.3f} "
                    f"[{est.ci_low:.3f}, {est.ci_high:.3f}] "
                    f"(lam={est.lam:.2f}, n_gold={est.n_labeled}, "
                    f"{'narrower' if est.narrower_than_classical else 'NOT narrower'} than gold-only)"
                )
            else:
                plines.append(
                    f"{p:<16} skipped: needs >=2 gold and >=2 unlabeled judged "
                    f"(have {len(lab_f)}/{len(unlab_f)})"
                )
        blocks["PPI-rectified success estimates"] = "\n".join(plines)
    return blocks
