#!/usr/bin/env python3
"""Cross-task generalization experiment: run woracle's OBJECT-FREE oracle
(success.demo_match, DINOv2, zero detection) on MULTIPLE RH20T tasks and report
per-task success-vs-failure separation (leave-one-out AUROC).

This is the generalizability test: no per-task tuning — the same channel,
prompts derived from the task, prototypes from the task's own labeled episodes.

Pipeline:
  ratings_all.json (all tasks) -> pick tasks with >=k success AND >=k fail
  -> ONE pass over the LOCAL color tar to extract those tasks' scenes
  -> per task: build concat-view mp4 per scene, leave-one-out DINOv2 margin AUROC
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
from collections import defaultdict

import numpy as np

CAM0, CAM1, CAM2 = "038522063145", "038522062547", "043322070878"


def _decode(path, max_frames):
    import imageio.v3 as iio

    try:
        fr = []
        for i, f in enumerate(iio.imiter(path, plugin="pyav")):
            if i >= max_frames:
                break
            fr.append(np.asarray(f)[..., :3])
        return np.stack(fr) if fr else None
    except Exception:
        return None


def _concat(c0, c1, c2):
    from PIL import Image

    T = min(len(c0), len(c1), len(c2))
    c0, c1, c2 = c0[:T], c1[:T], c2[:T]
    H, W = c0.shape[1:3]
    out = np.empty((T, H + H // 2, W, 3), np.uint8)
    for t in range(T):
        out[t, :H] = c0[t]
        out[t, H:, : W // 2] = np.asarray(Image.fromarray(c1[t]).resize((W // 2, H // 2)))
        out[t, H:, W // 2 : W // 2 * 2] = np.asarray(
            Image.fromarray(c2[t]).resize((W // 2, H // 2))
        )
    return out


def loo_auroc(embs):
    """embs: [(emb, is_success)] -> (auroc, accuracy, margins)."""
    from woracle.stats import auroc

    margins, ys, correct = [], [], 0
    for i, (q, ok) in enumerate(embs):
        sp = np.stack([e for j, (e, o) in enumerate(embs) if o and j != i])
        fp = np.stack([e for j, (e, o) in enumerate(embs) if (not o) and j != i])
        m = float(np.mean(sp @ q) - np.mean(fp @ q))
        correct += (m > 0) == ok
        margins.append(m)
        ys.append(ok)
    m, y = np.array(margins), np.array(ys, bool)
    a = auroc(m[y], m[~y]) if y.any() and (~y).any() else float("nan")
    return a, correct, len(embs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--color_tar", default=os.path.expanduser("~/rh20t_local/RH20T_color/RH20T_cfg1.tar.gz")
    )
    ap.add_argument("--ratings", default=os.path.expanduser("~/rh20t_local/ratings_all.json"))
    ap.add_argument("--workdir", default=os.path.expanduser("~/rh20t_local/extracted_multi"))
    ap.add_argument("--out", default=os.path.expanduser("~/rh20t_multitask_out"))
    ap.add_argument("--tasks", default="", help="comma-sep task ids; empty = auto-pick")
    ap.add_argument("--n_tasks", type=int, default=4)
    ap.add_argument("--per_class", type=int, default=5, help="max success/fail episodes per task")
    ap.add_argument(
        "--min_class", type=int, default=3, help="min success AND fail to include a task"
    )
    ap.add_argument("--max_frames", type=int, default=120)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.workdir, exist_ok=True)

    ratings = json.load(open(args.ratings))
    by_task = defaultdict(lambda: {"succ": [], "fail": []})
    for sc, r in ratings.items():
        t = sc.split("_user")[0]
        if r >= 2:
            by_task[t]["succ"].append(sc)
        elif 0 <= r < 2:
            by_task[t]["fail"].append(sc)

    if args.tasks:
        tasks = args.tasks.split(",")
    else:
        cand = [
            (t, len(d["succ"]), len(d["fail"]))
            for t, d in by_task.items()
            if len(d["succ"]) >= args.min_class and len(d["fail"]) >= args.min_class
        ]
        cand.sort(key=lambda x: -min(x[1], x[2]))  # most balanced first
        tasks = [t for t, _, _ in cand[: args.n_tasks]]
    print("tasks:", tasks, flush=True)

    # scenes we need (balanced per task)
    want = {}  # scene -> (task, is_success)
    for t in tasks:
        for sc in by_task[t]["succ"][: args.per_class]:
            want[sc] = (t, True)
        for sc in by_task[t]["fail"][: args.per_class]:
            want[sc] = (t, False)

    # ONE pass over the color tar extracting only the needed scenes' 3 cams
    if not os.listdir(args.workdir):
        print(f"extracting {len(want)} scenes from color tar (one pass) ...", flush=True)
        with tarfile.open(args.color_tar, "r:gz") as tf:
            members = [
                m
                for m in tf
                if any(s in m.name for s in want)
                and any(c in m.name for c in (CAM0, CAM1, CAM2))
                and m.name.endswith(".mp4")
            ]
            tf.extractall(args.workdir, members=members)
        print(f"  extracted {len(members)} cam files", flush=True)

    # find each scene dir
    scene_path = {}
    for dp, dirs, _ in os.walk(args.workdir):
        for d in dirs:
            if d in want:
                scene_path[d] = os.path.join(dp, d)

    from woracle.channels.demo_match import _tail_embedding

    results = {}
    for t in tasks:
        embs, kept = [], []
        for sc, (tt, ok) in want.items():
            if tt != t or sc not in scene_path:
                continue
            cams = []
            for cam in (CAM0, CAM1, CAM2):
                cp = os.path.join(scene_path[sc], f"cam_{cam}", "color", "color.mp4")
                fr = _decode(cp, args.max_frames) if os.path.isfile(cp) else None
                if fr is None:
                    cams = None
                    break
                cams.append(fr)
            if cams is None:
                continue
            view = _concat(*cams)
            embs.append((_tail_embedding(view, 0.25, 8, "facebook/dinov2-small", None), ok))
            kept.append((sc, ok))
        ns = sum(o for _, o in kept)
        nf = len(kept) - ns
        if ns < 2 or nf < 2:
            print(f"[{t}] insufficient after decode ({ns} succ, {nf} fail) — skip")
            continue
        a, correct, n = loo_auroc(embs)
        results[t] = dict(auroc=round(a, 3), correct=correct, n=n, n_succ=ns, n_fail=nf)
        print(f"[{t}] LOO AUROC={a:.3f}  acc={correct}/{n}  ({ns} succ, {nf} fail)", flush=True)

    json.dump(results, open(os.path.join(args.out, "results.json"), "w"), indent=2)
    print("\n=== CROSS-TASK SUMMARY (object-free oracle, zero detection) ===")
    print(f"{'task':<14}{'AUROC':>8}{'acc':>9}{'succ/fail':>12}")
    for t, r in results.items():
        print(
            f"{t:<14}{r['auroc']:>8.3f}{r['correct']:>5}/{r['n']:<3}{r['n_succ']:>6}/{r['n_fail']}"
        )
    aurocs = [r["auroc"] for r in results.values() if r["auroc"] == r["auroc"]]
    if aurocs:
        print(f"\nmean LOO AUROC across {len(aurocs)} tasks = {np.mean(aurocs):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
