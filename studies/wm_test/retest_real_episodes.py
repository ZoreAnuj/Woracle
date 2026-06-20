#!/usr/bin/env python3
"""Re-test: does woracle's OBJECT-FREE oracle separate real RH20T success from
failure episodes — where the tip-grounding path abstained?

Builds DINOv2 success/fail prototype banks from a subset of labeled real
episodes, grades the held-out rest with success.demo_match via the
frames.passthrough grounder (no object detection), reports per-episode margins,
verdicts, and AUROC. Then contrasts with the OLD openvocab + success.predicates
config on the same episodes (expected: abstain — can't ground the tip)."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

import woracle
import woracle.channels  # noqa: F401  (registers success.demo_match etc.)
import woracle.gate.signals  # noqa: F401  (registers gate signals)
import woracle.grounders  # noqa: F401  (registers frames.passthrough, openvocab)
import woracle.testing.plugins  # noqa: F401  (registers binding_health/permanence signals)
from woracle.channels.demo_match import build_demo_protos
from woracle.contracts import GatePolicy
from woracle.io import rollout_from_video
from woracle.io.video import decode_video
from woracle.pipeline import GradeRunConfig, grade_rollouts
from woracle.stats import auroc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default=os.path.expanduser("~/rh20t_real_episodes"))
    ap.add_argument(
        "--extra_success",
        default=os.path.expanduser("~/wm_rollouts_local"),
        help="dir with real_*.mp4 (holdout success) to add more success samples",
    )
    ap.add_argument("--out", default=os.path.expanduser("~/rh20t_retest_out"))
    ap.add_argument("--temp", type=float, default=0.04)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    labels = json.load(open(os.path.join(args.episodes, "labels.json")))
    items = [(os.path.join(args.episodes, f"{rid}.mp4"), bool(ok)) for rid, ok in labels.items()]
    # add holdout success episodes (real_*.mp4) for more positives
    for mp in sorted(glob.glob(os.path.join(args.extra_success, "real_*.mp4"))):
        items.append((mp, True))
    succ = [p for p, ok in items if ok]
    fail = [p for p, ok in items if not ok]
    print(f"episodes: {len(succ)} success, {len(fail)} fail")
    if len(succ) < 2 or len(fail) < 2:
        print("need >=2 of each class")
        return 2

    # split: half of each class -> prototypes, other half -> test (disjoint)
    def split(xs):
        h = max(1, len(xs) // 2)
        return xs[:h], xs[h:]

    proto_s, test_s = split(succ)
    proto_f, test_f = split(fail)
    print(
        f"  protos: {len(proto_s)} succ + {len(proto_f)} fail | "
        f"test: {len(test_s)} succ + {len(test_f)} fail"
    )

    # build prototype banks (DINOv2, no detection)
    demos = [(decode_video(p)[0], True) for p in proto_s] + [
        (decode_video(p)[0], False) for p in proto_f
    ]
    sp, fp = build_demo_protos(demos, tail_frac=0.25, max_frames=8)

    # grade the held-out test set with the OBJECT-FREE path
    test = [(p, True) for p in test_s] + [(p, False) for p in test_f]
    refs, truth = [], {}
    for p, ok in test:
        rid = os.path.splitext(os.path.basename(p))[0]
        refs.append(
            rollout_from_video(
                p, rollout_id=rid, policy="success" if ok else "fail", source="real:rh20t"
            )
        )
        truth[rid] = ok

    role_free = GatePolicy(require_role_bindings=False, required_signals=[], thresholds=[])
    cfg = GradeRunConfig(
        grounder="frames.passthrough",
        signals=[],  # role-free: no gate veto on coherent real video
        channels=["success.demo_match"],
        component_params={
            "success.demo_match": {"success_protos": sp, "fail_protos": fp, "temp": args.temp}
        },
        policy=role_free,
        store_root=os.path.join(args.out, "store"),
        out_dir=os.path.join(args.out, "object_free"),
    )
    cards = grade_rollouts(
        refs,
        woracle.load_spec(os.path.expanduser("~/woracle/specs/rh20t-pipette-insert/spec.yaml")),
        cfg,
    )

    print("\n=== OBJECT-FREE oracle (success.demo_match) ===")
    margins, ys = [], []
    correct = 0
    for c in sorted(cards, key=lambda c: c.rollout_id):
        ch = c.channel("success.demo_match")
        m = ch.details.get("margin") if ch else None
        y = truth[c.rollout_id]
        margins.append(m if m is not None else 0.0)
        ys.append(y)
        pred = c.success.verdict
        ok_pred = (pred == "pass") == y
        correct += ok_pred and pred in ("pass", "fail")
        print(
            f"  {c.rollout_id:<10} truth={'SUCC' if y else 'FAIL'} verdict={pred:<8} "
            f"margin={m:+.4f}"
            if m is not None
            else f"  {c.rollout_id:<10} truth={'SUCC' if y else 'FAIL'} verdict={pred}"
        )
    m = np.array(margins)
    y = np.array(ys, bool)
    if y.any() and (~y).any():
        a = auroc(m[y], m[~y])
        print(
            f"\n  AUROC(success vs fail margins) = {a:.3f}   "
            f"(n_test={len(y)}: {int(y.sum())} succ, {int((~y).sum())} fail)"
        )
        print(f"  verdict accuracy on non-abstained = {correct}/{len(cards)}")

    # contrast: OLD tip-grounding path on the same test episodes
    print("\n=== OLD object-grounded oracle (openvocab + success.predicates) ===")
    old = GradeRunConfig(
        grounder="openvocab.gdino_sam",
        signals=["binding_health", "permanence"],
        channels=["success.predicates"],
        component_params={"openvocab.gdino_sam": {"stride": 8, "det_threshold": 0.18, "tiles": 2}},
        store_root=os.path.join(args.out, "store_old"),
        out_dir=os.path.join(args.out, "object_grounded"),
    )
    try:
        old_cards = grade_rollouts(
            [
                rollout_from_video(
                    p,
                    rollout_id=os.path.splitext(os.path.basename(p))[0],
                    policy="x",
                    source="real:rh20t",
                )
                for p, _ in test
            ],
            woracle.load_spec(os.path.expanduser("~/woracle/specs/rh20t-pipette-insert/spec.yaml")),
            old,
        )
        nab = sum(1 for c in old_cards if c.success.verdict == "abstain")
        print(
            f"  {nab}/{len(old_cards)} abstained (tip ungroundable); {len(old_cards) - nab} graded"
        )
    except Exception as e:
        print(f"  old path errored: {type(e).__name__}: {str(e)[:120]}")

    print(f"\nresults -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
