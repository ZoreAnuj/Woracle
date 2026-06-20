#!/usr/bin/env python3
"""Export real RH20T cfg1/task_0200 episodes (success + FAILURE) as concat-view
mp4s + rating labels — fully local, no framework. Reads the color tar's scenes,
builds the same top=cam0 / bottom=[cam1|cam2] view used for training/WM, and
labels by metadata.json rating (>=2 success, <2 failure: 0=robot-fail,1=task-fail).

Run in the woracle venv (uses imageio[ffmpeg] + numpy + PIL — no system ffmpeg)."""

from __future__ import annotations

import argparse
import json
import os
import tarfile

import numpy as np

# Three FIXED camera serials (from rh20t_to_lerobot.py)
CAM0, CAM1, CAM2 = "038522063145", "038522062547", "043322070878"


def _decode(path: str, max_frames: int) -> np.ndarray | None:
    import imageio.v3 as iio

    try:
        frames = []
        for i, fr in enumerate(iio.imiter(path, plugin="pyav")):
            if i >= max_frames:
                break
            frames.append(np.asarray(fr)[..., :3])
        return np.stack(frames) if frames else None
    except Exception as e:
        print(f"    [decode fail] {path}: {type(e).__name__}: {e}")
        return None


def _concat_view(c0, c1, c2):
    """top = cam0 (H,W); bottom = [cam1|cam2] each resized to (H/2, W/2)."""
    from PIL import Image

    T = min(len(c0), len(c1), len(c2))
    c0, c1, c2 = c0[:T], c1[:T], c2[:T]
    H, W = c0.shape[1:3]
    out = np.empty((T, H + H // 2, W, 3), np.uint8)
    for t in range(T):
        out[t, :H] = c0[t]
        bl = np.asarray(Image.fromarray(c1[t]).resize((W // 2, H // 2), Image.BILINEAR))
        br = np.asarray(Image.fromarray(c2[t]).resize((W // 2, H // 2), Image.BILINEAR))
        out[t, H:, : W // 2] = bl
        out[t, H:, W // 2 : W // 2 * 2] = br
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tar", default=os.path.expanduser("~/rh20t_local/RH20T_color/RH20T_cfg1.tar.gz")
    )
    ap.add_argument("--workdir", default=os.path.expanduser("~/rh20t_local/extracted"))
    ap.add_argument("--out", default=os.path.expanduser("~/rh20t_real_episodes"))
    ap.add_argument("--n_success", type=int, default=6)
    ap.add_argument("--n_fail", type=int, default=6)
    ap.add_argument("--max_frames", type=int, default=150)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--ratings", default="",
                    help="external {scene: rating} json (ratings live in lowdim, not color)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.workdir, exist_ok=True)

    # 1) extract task_0200 scenes (members containing 'task_0200')
    print("scanning tar for task_0200 scenes ...", flush=True)
    inner_root = None
    with tarfile.open(args.tar, "r:gz") as tf:
        members = [m for m in tf.getmembers() if "task_0200" in m.name]
        # scene = the path component right after the cfg root that contains task_0200
        scenes = sorted(
            {
                m.name.split("task_0200")[0]
                + "task_0200"
                + m.name.split("task_0200")[1].split("/")[0]
                for m in members
            }
        )
        print(f"  found {len(scenes)} task_0200 scenes")
        if not os.listdir(args.workdir):
            print("  extracting (this is the slow part) ...", flush=True)
            tf.extractall(args.workdir, members=members)
    # locate the extracted root
    for dirpath, dirs, _ in os.walk(args.workdir):
        if any("task_0200" in d for d in dirs):
            inner_root = dirpath
            break
    inner_root = inner_root or args.workdir
    scene_dirs = sorted(d for d in os.listdir(inner_root) if "task_0200" in d)
    print(f"  extracted scene dirs: {len(scene_dirs)} under {inner_root}")

    # 2) ratings: from external ratings.json (lowdim) or per-scene metadata.json (color)
    ext = json.load(open(args.ratings)) if args.ratings and os.path.isfile(args.ratings) else {}
    rated = []
    for sd in scene_dirs:
        if sd in ext:
            rated.append((sd, int(ext[sd])))
            continue
        mp = os.path.join(inner_root, sd, "metadata.json")
        if os.path.isfile(mp):
            try:
                rated.append((sd, int(json.load(open(mp)).get("rating", -1))))
            except Exception:
                pass
    if not rated:
        print("!! no ratings found — pass --ratings <lowdim ratings.json> "
              "(metadata.json is NOT in the color tar).")
        print("   sample scene contents:", os.listdir(os.path.join(inner_root, scene_dirs[0]))[:8])
        return 2
    succ = [s for s, r in rated if r >= 2]
    fail = [s for s, r in rated if 0 <= r < 2]
    rmap = dict(rated)
    print(f"  ratings: {len(succ)} success, {len(fail)} fail (of {len(rated)} rated)")
    pick = [(s, True) for s in succ[: args.n_success]] + [(s, False) for s in fail[: args.n_fail]]

    # 3) build concat-view mp4 per picked scene
    import imageio.v3 as iio

    labels, manifest = {}, []
    for i, (sd, ok) in enumerate(pick):
        sp = os.path.join(inner_root, sd)
        cams = []
        good = True
        for cam in (CAM0, CAM1, CAM2):
            cp = os.path.join(sp, f"cam_{cam}", "color", "color.mp4")
            fr = _decode(cp, args.max_frames) if os.path.isfile(cp) else None
            if fr is None:
                good = False
                break
            cams.append(fr)
        if not good:
            print(f"  [skip] {sd}: missing/undecodable cam")
            continue
        view = _concat_view(*cams)
        rid = f"{'succ' if ok else 'fail'}_{i:02d}"
        out_mp4 = os.path.join(args.out, rid + ".mp4")
        iio.imwrite(out_mp4, view, fps=args.fps, codec="libx264")
        labels[rid] = ok
        manifest.append(dict(id=rid, scene=sd, rating=rmap[sd], label=ok, n_frames=len(view)))
        print(f"  [{rid}] rating {rmap[sd]} -> {len(view)} frames", flush=True)

    json.dump(labels, open(os.path.join(args.out, "labels.json"), "w"), indent=2)
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=2)
    print(
        f"EXPORT_DONE -> {args.out} ({sum(labels.values())} success, "
        f"{len(labels) - sum(labels.values())} fail)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
