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
    # First-party reference components register on import; third-party via EPs.
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


def compile(*_args: object, **_kwargs: object) -> TaskSpec:
    """Stage-1 spec compiler — lands in P4."""
    raise WoracleError(
        "woracle.compile (demos -> TaskSpec) ships in P4. Until then write specs "
        "by hand — they are designed to be human-readable YAML (see specs/), and "
        "load with woracle.load_spec(path)."
    )


def ground(*_args: object, **_kwargs: object):
    """Standalone Stage-2 grounding — public form lands in P1."""
    raise WoracleError(
        "standalone woracle.ground arrives with the real-video grounders in P1. "
        "P0 grades blobworld end-to-end via woracle.grade(...)."
    )
