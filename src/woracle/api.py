"""Woracle public API — the four verbs (ARCH §1): compile, ground, grade, report.

P0 ships ``grade`` + ``report`` end-to-end (blobworld profile), ``load_spec``,
and honest stubs for the verbs that land in later phases.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from woracle.contracts.spec import load_spec
from woracle.errors import WoracleError

if TYPE_CHECKING:
    from woracle.contracts import GradeCard, Leaderboard, TaskSpec
    from woracle.pipeline import GradeRunConfig

__all__ = ["compile", "grade", "ground", "load_spec", "report"]


def _ensure_plugins_loaded() -> None:
    # First-party components register on import; third-party via entry points.
    import woracle.channels
    import woracle.gate.signals
    import woracle.grounders
    import woracle.testing.plugins  # noqa: F401
    from woracle.registry import load_entry_points

    load_entry_points()


def grade(
    rollouts_dir: str,
    spec: str | TaskSpec,
    out_dir: str = "woracle_out",
    *,
    profile: str = "auto",
    store_root: str | None = None,
    config: GradeRunConfig | None = None,
) -> list[GradeCard]:
    """Grade every rollout under ``rollouts_dir`` against a task spec."""
    from woracle.io import list_rollouts
    from woracle.pipeline import blob_profile, grade_rollouts

    _ensure_plugins_loaded()
    spec_obj = load_spec(spec) if isinstance(spec, str) else spec
    rollouts = list_rollouts(rollouts_dir)
    if not rollouts:
        raise WoracleError(f"no episodes found under {rollouts_dir!r} (need rollout.json dirs)")

    if config is None:
        sources = {r.source for r in rollouts}
        if profile == "blobworld" or (profile == "auto" and sources <= {"blobworld"}):
            config = blob_profile()
        else:
            raise WoracleError(
                f"no grading profile for sources {sorted(sources)} yet — real-video "
                "grounders land in P1. Pass an explicit GradeRunConfig to override."
            )
    # Copy before mutation: caller-passed configs are templates, never scratch.
    config = config.fresh(out_dir=out_dir)
    if store_root is not None:
        config.store_root = store_root
    elif config.store_root == ".woracle_store":
        config.store_root = os.path.join(out_dir, "store")
    return grade_rollouts(rollouts, spec_obj, config)


def report(
    cards_dir_or_cards: str | list[GradeCard],
    out_path: str | None = None,
) -> Leaderboard:
    """Assemble a leaderboard from grade-card snapshots (no recomputation)."""
    from woracle.reporting import build_leaderboard, load_cards, render_markdown

    cards = (
        load_cards(cards_dir_or_cards)
        if isinstance(cards_dir_or_cards, str)
        else cards_dir_or_cards
    )
    board = build_leaderboard(cards)
    if out_path:
        md = render_markdown(board)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md)
    return board


def compile(
    demos_dir: str,
    prompt: str,
    *,
    name: str = "compiled-task",
    out: str | None = None,
    grounder: str = "relational.motion",
) -> TaskSpec:
    """Stage-1: compile a TaskSpec from demo episodes (or REFUSE loudly).

    ``demos_dir`` holds episode dirs (rollout.json + frames payload). The
    compiled spec is self-tested (demos must pass, minted negatives must
    fail) before it is returned; a spec that cannot separate them raises
    SpecError instead of pretending to be an oracle.
    """
    from woracle.compiler import compile_spec
    from woracle.io import list_rollouts, load_frames

    _ensure_plugins_loaded()
    rollouts = list_rollouts(demos_dir)
    if not rollouts:
        raise WoracleError(f"no demo episodes found under {demos_dir!r}")
    demo_frames = [load_frames(r) for r in rollouts]
    spec = compile_spec(demo_frames, prompt, name=name, grounder=grounder)
    if out:
        import os

        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(spec.to_yaml())
    return spec


def ground(*_args: object, **_kwargs: object):
    """Standalone Stage-2 grounding — public form lands in P1."""
    raise WoracleError(
        "standalone woracle.ground arrives with the real-video grounders in P1. "
        "P0 grades blobworld end-to-end via woracle.grade(...)."
    )
