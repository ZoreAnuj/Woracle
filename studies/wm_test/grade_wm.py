#!/usr/bin/env python3
"""Grade fresh Cosmos-3 WM rollouts with woracle (local, on the 4070).

Inputs: a dir of WM-generated mp4s (real_*/tf_*/cl_*) + labels.json + manifest.json
from gen_wm_rollouts.py. Three things:

  1. Stage-1 COMPILE test: try woracle.compile on the real_* demos; report PASS/REFUSE
     + the self-test numbers (does the compiler recover the task from real frames?).
  2. GRADE every rollout with the real open-vocab grounder against the spec
     (compiled if it passed, else the hand-written pipette spec).
  3. REPORT: leaderboard + MNAR + PPI (golds = real+tf episode ratings) + HTML,
     grouped by tier so we can see gate(real) vs gate(teacher-forced) vs gate(closed-loop).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import woracle
from woracle.contracts import load_spec
from woracle.errors import SpecError
from woracle.io import rollout_from_video
from woracle.pipeline import GradeRunConfig, grade_rollouts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default=os.path.expanduser("~/wm_rollouts_local"))
    ap.add_argument(
        "--spec", default=os.path.expanduser("~/woracle/specs/rh20t-pipette-insert/spec.yaml")
    )
    ap.add_argument("--out", default=os.path.expanduser("~/wm_test_out"))
    ap.add_argument("--prompt", default="insert the large pipette tip into the tip holder")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    manifest = {m["id"]: m for m in json.load(open(os.path.join(args.rollouts, "manifest.json")))}
    labels = json.load(open(os.path.join(args.rollouts, "labels.json")))

    # ---- 1) Stage-1 compile test on the REAL demos ----
    real_demos = sorted(glob.glob(os.path.join(args.rollouts, "real_*.mp4")))
    compile_report = {"attempted": False}
    spec = None
    if len(real_demos) >= 2:
        compile_report["attempted"] = True
        # woracle.compile wants episode dirs; wrap the real mp4s into a demos dir
        demo_dir = os.path.join(args.out, "real_demos")
        os.makedirs(demo_dir, exist_ok=True)
        from woracle.io import save_episode
        from woracle.io.video import decode_video

        for mp in real_demos:
            rid = os.path.splitext(os.path.basename(mp))[0]
            frames, _ = decode_video(mp)
            save_episode(os.path.join(demo_dir, rid), rid, frames, source="real")
        try:
            spec = woracle.compile(
                demo_dir,
                args.prompt,
                name="rh20t-pipette-compiled",
                out=os.path.join(args.out, "compiled_spec.yaml"),
            )
            st = spec.spec_provenance.self_test
            compile_report.update(
                accepted=True,
                demos_passed=st.demos_passed,
                demos_total=st.demos_total,
                negatives_failed=st.negatives_failed,
                negatives_total=st.negatives_total,
                notes=st.notes,
            )
            print(
                f"[compile] ACCEPTED: demos {st.demos_passed}/{st.demos_total} pass, "
                f"negatives {st.negatives_failed}/{st.negatives_total} fail"
            )
        except SpecError as e:
            compile_report.update(accepted=False, refuse_reason=str(e)[:400])
            print(f"[compile] REFUSED: {str(e)[:200]}")
    # fall back to the hand spec for the actual grading eval (reliable open-vocab path)
    eval_spec = load_spec(args.spec)
    json.dump(compile_report, open(os.path.join(args.out, "compile_report.json"), "w"), indent=2)

    # ---- 2) grade every rollout with the open-vocab grounder ----
    mp4s = sorted(glob.glob(os.path.join(args.rollouts, "*.mp4")))
    refs = []
    for mp in mp4s:
        rid = os.path.splitext(os.path.basename(mp))[0]
        m = manifest.get(rid, {})
        # policy label = tier so the leaderboard groups real / teacher_forced / closed_loop:<pol>
        tier = m.get("tier", "unknown")
        pol = m.get("policy", "?")
        policy = tier if tier in ("real", "teacher_forced") else f"cl_{pol}"
        refs.append(rollout_from_video(mp, rollout_id=rid, policy=policy, source="wm:cosmos3"))

    config = GradeRunConfig(
        grounder="openvocab.gdino_sam",
        signals=[
            "binding_health",
            "permanence",
            "background_drift",
            "track_continuity",
            "motion_sanity",
        ],
        channels=["progress.goal_distance", "success.predicates"],
        component_params={"openvocab.gdino_sam": {"stride": 6, "det_threshold": 0.18, "tiles": 2}},
        store_root=os.path.join(args.out, "store"),
        out_dir=args.out,
    )
    cards = grade_rollouts(refs, eval_spec, config)

    # ---- 3) report ----
    golds = {k: bool(v) for k, v in labels.items()}  # real+tf only (gen restricted it)
    json.dump(golds, open(os.path.join(args.out, "golds.json"), "w"))
    board = woracle.report(
        cards,
        out_path=os.path.join(args.out, "report.md"),
        golds=os.path.join(args.out, "golds.json"),
        html_path=os.path.join(args.out, "report.html"),
    )

    print("\n=== gate verdict by tier ===")
    for c in sorted(cards, key=lambda c: c.rollout_id):
        print(
            f"  {c.rollout_id:<14} gate={c.gate.verdict:<11} success={c.success.verdict:<8} "
            f"reasons={'; '.join((c.gate.reasons or c.success.reasons)[:1])[:60]}"
        )
    print(f"\nreport -> {args.out}/report.md + report.html")


if __name__ == "__main__":
    sys.exit(main())
