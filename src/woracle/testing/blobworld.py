"""Blobworld — procedural synthetic manipulation world with free ground truth.

A blue "gripper" dot carries a red square toward a green "cup". Because we
render it ourselves, every test gets exact ground truth (centroids, events,
labels) with zero checkpoints, zero GPU, zero network — the workhorse for unit
tests of binding, permanence, progress, and predicates (ARCH §7).

Scenarios
---------
``success``    square ends inside the cup, stationary; gripper retreats.
``fail_miss``  square ends beside the cup (never contained).
``fail_drop``  square detaches mid-path and stays there.
``vanish``     like success, but the square stops being rendered mid-rollout —
               the WM "deleted the object"; the permanence gate MUST abstain.
``random``     attached random walk (a random policy).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from woracle.contracts import (
    FailureMode,
    Phase,
    Predicate,
    Role,
    RolloutRef,
    TaskSpec,
)
from woracle.io import save_episode

H, W = 96, 128
BG = 200
SQ = 8  # carried square side
GRIP_R = 4  # gripper radius
# Cup geometry (a "U"): two vertical walls + base, open top.
CUP_X0, CUP_X1 = 100, 124
CUP_Y0, CUP_Y1 = 30, 66
WALL = 4
INTERIOR = (CUP_X0 + WALL, CUP_Y0, CUP_X1 - WALL, CUP_Y1 - WALL)  # x0, y0, x1, y1

COLORS = {
    "gripper": (40, 60, 220),  # blue
    "carried_object": (220, 40, 40),  # red
    "receptacle": (40, 180, 60),  # green
}

KINDS = ("success", "fail_miss", "fail_drop", "vanish", "random")
LABELS = {
    "success": "success",
    "fail_miss": "fail",
    "fail_drop": "fail",
    "vanish": "fail",  # ground-truth outcome is fail; the GATE should abstain
    "random": "fail",
}


@dataclass
class Truth:
    kind: str
    label: str
    gripper: np.ndarray  # (T, 2) float32 (x, y)
    carried: np.ndarray  # (T, 2) float32, NaN after vanish
    events: dict[str, int] = field(default_factory=dict)


def _smoothstep(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


def _draw(frame: np.ndarray, kind: str, cx: float, cy: float) -> None:
    if kind == "gripper":
        yy, xx = np.mgrid[0:H, 0:W]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= GRIP_R**2
        frame[mask] = COLORS["gripper"]
    else:  # carried square
        x0, y0 = round(cx - SQ / 2), round(cy - SQ / 2)
        frame[max(0, y0) : y0 + SQ, max(0, x0) : x0 + SQ] = COLORS["carried_object"]


def _draw_cup(frame: np.ndarray) -> None:
    c = COLORS["receptacle"]
    frame[CUP_Y0:CUP_Y1, CUP_X0 : CUP_X0 + WALL] = c  # left wall
    frame[CUP_Y0:CUP_Y1, CUP_X1 - WALL : CUP_X1] = c  # right wall
    frame[CUP_Y1 - WALL : CUP_Y1, CUP_X0:CUP_X1] = c  # base


def make_episode(kind: str, seed: int = 0, n_frames: int = 60) -> tuple[np.ndarray, Truth]:
    if kind not in KINDS:
        raise ValueError(f"unknown blobworld kind '{kind}' (kinds: {KINDS})")
    rng = np.random.default_rng(seed)
    T = n_frames
    start = np.array([16.0, H / 2.0])
    interior_cx = (INTERIOR[0] + INTERIOR[2]) / 2.0
    interior_cy = (INTERIOR[1] + INTERIOR[3]) / 2.0 + 6.0  # settle low in the cup

    grip = np.zeros((T, 2), np.float32)
    carried = np.zeros((T, 2), np.float32)
    events: dict[str, int] = {}

    if kind == "random":
        pos = start.copy()
        vel = np.zeros(2)
        for t in range(T):
            vel = 0.8 * vel + rng.normal(0, 1.6, 2)
            pos = np.clip(pos + vel, [GRIP_R + 1, GRIP_R + 1], [W - SQ - 2, H - SQ - 2])
            grip[t] = pos
            carried[t] = pos + np.array([0.0, 10.0])  # numpy broadcast, not concat
    else:
        target = np.array([interior_cx, interior_cy - 10.0])  # gripper holds above square
        t_arrive = int(T * 0.6)
        tt = _smoothstep(np.clip(np.arange(T) / max(t_arrive, 1), 0, 1))[:, None]
        if kind == "fail_miss":
            target = np.array([CUP_X0 - 18.0, interior_cy - 10.0])  # beside the cup
        path = start[None, :] + (target - start)[None, :] * tt
        grip[:] = path
        carried[:] = path + np.array([0.0, 10.0])

        if kind == "fail_drop":
            t_drop = int(T * 0.4)
            events["drop_frame"] = t_drop
            carried[t_drop:] = carried[t_drop - 1]  # square stays put
        else:
            t_release = min(t_arrive + 2, T - 1)
            events["release_frame"] = t_release
            carried[t_release:] = carried[t_release - 1]  # settled
            # gripper retreats after release
            retreat = np.array([-26.0, -22.0])
            for i, t in enumerate(range(t_release, T)):
                frac = _smoothstep(np.array([min(1.0, i / 10.0)]))[0]
                grip[t] = grip[t_release - 1] + retreat * frac
        if kind in ("success", "vanish"):
            events["contained_from"] = events.get("release_frame", t_arrive)
        if kind == "vanish":
            events["vanish_frame"] = int(T * 0.55)

    frames = np.full((T, H, W, 3), BG, np.uint8)
    # static low-amplitude texture so frames aren't flat (seeded, deterministic)
    frames += rng.integers(0, 6, (1, H, W, 3), dtype=np.uint8)
    vanish_at = events.get("vanish_frame", T + 1)
    for t in range(T):
        _draw_cup(frames[t])
        if t < vanish_at:
            _draw(frames[t], "carried_object", carried[t, 0], carried[t, 1])
        _draw(frames[t], "gripper", grip[t, 0], grip[t, 1])

    carried_truth = carried.copy()
    if kind == "vanish":
        carried_truth[vanish_at:] = np.nan
    return frames, Truth(
        kind=kind, label=LABELS[kind], gripper=grip, carried=carried_truth, events=events
    )


def blob_spec() -> TaskSpec:
    """The hand-written TaskSpec for blobworld insertion (P0 reference spec)."""
    return TaskSpec(
        name="blobworld-insert",
        prompt="insert the red block into the green cup",
        roles=[
            Role(
                name="carried_object",
                definition="the object that co-moves with the gripper and is placed",
                motion="co_moves_with_effector",
                candidates=["red block", "red square"],
                required=True,
            ),
            Role(
                name="receptacle",
                definition="the static container the object must end up inside",
                motion="static",
                candidates=["green cup", "green container"],
                required=True,
            ),
            Role(
                name="gripper",
                definition="the end-effector doing the carrying",
                motion="actuated",
                candidates=["blue gripper", "blue dot"],
                required=False,
            ),
        ],
        phases=[
            Phase(
                name="approach",
                order=0,
                description="carried object approaches the cup",
                active=[
                    Predicate(kind="approaching", subject="carried_object", object="receptacle")
                ],
            ),
            Phase(
                name="insert",
                order=1,
                description="object enters the cup interior",
                active=[Predicate(kind="contained", subject="carried_object", object="receptacle")],
            ),
            Phase(
                name="settle",
                order=2,
                description="object rests inside the cup",
                active=[Predicate(kind="stationary", subject="carried_object")],
            ),
        ],
        success=[
            Predicate(
                kind="contained",
                subject="carried_object",
                object="receptacle",
                params={"erode_px": 5.0},
            ),
            Predicate(kind="stationary", subject="carried_object", params={"tol": 1.5}),
        ],
        failure_modes=[
            FailureMode(name="missed", description="object ends outside the receptacle"),
            FailureMode(name="dropped", description="object detaches before the receptacle"),
        ],
        success_sustain_frames=5,
        version=1,
    )


def write_dataset(
    root: str,
    *,
    kinds: dict[str, int] | None = None,
    seed: int = 0,
    n_frames: int = 60,
    fps: float = 10.0,
) -> list[RolloutRef]:
    """Generate a blobworld dataset: episode dirs + labels.json + spec.yaml."""
    kinds = kinds or {"success": 2, "fail_miss": 2, "fail_drop": 1, "vanish": 1, "random": 1}
    refs: list[RolloutRef] = []
    labels: dict[str, str] = {}
    i = 0
    for kind, count in kinds.items():
        for j in range(count):
            rid = f"{kind}_{j:02d}"
            frames, truth = make_episode(kind, seed=seed + i, n_frames=n_frames)
            ep_dir = os.path.join(root, rid)
            ref = save_episode(
                ep_dir,
                rid,
                frames,
                fps=fps,
                policy=kind,
                source="blobworld",
                meta={"kind": kind, "label": truth.label},
            )
            np.savez_compressed(
                os.path.join(ep_dir, "truth.npz"),
                gripper=truth.gripper,
                carried=truth.carried,
            )
            with open(os.path.join(ep_dir, "events.json"), "w", encoding="utf-8") as f:
                json.dump({"kind": kind, "label": truth.label, **truth.events}, f, indent=2)
            labels[rid] = truth.label
            refs.append(ref)
            i += 1
    with open(os.path.join(root, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, sort_keys=True)
    with open(os.path.join(root, "spec.yaml"), "w", encoding="utf-8") as f:
        f.write(blob_spec().to_yaml())
    return refs
