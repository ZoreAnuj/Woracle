"""Woracle CLI — thin veneer over the library (never logic of its own)."""

from __future__ import annotations

import json
import os

import typer

app = typer.Typer(
    name="woracle",
    help="Woracle — the world-model oracle. Demos in, oracle out.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def version() -> None:
    """Print the woracle version."""
    from woracle._version import __version__

    typer.echo(__version__)


@app.command()
def doctor() -> None:
    """Report environment health: core deps, optional stacks, plugins."""
    import importlib
    import sys

    from woracle._version import __version__
    from woracle.registry import KINDS, available, load_entry_points

    typer.echo(f"woracle {__version__}  (python {sys.version.split()[0]})")
    typer.echo("\ncore dependencies:")
    for mod in ("numpy", "pydantic", "yaml", "typer"):
        m = importlib.import_module(mod)
        typer.echo(f"  {mod:<10} {getattr(m, '__version__', '?')}")
    typer.echo("\noptional stacks (absent is EXPECTED for the core install):")
    for mod, extra in (("torch", "ground"), ("transformers", "ground"), ("cv2", "ground")):
        try:
            importlib.import_module(mod)
            typer.echo(f"  {mod:<13} present")
        except ImportError:
            typer.echo(f"  {mod:<13} absent   (pip install 'woracle[{extra}]' when needed)")
    import woracle.testing.plugins  # noqa: F401  (register first-party components)

    rep = load_entry_points()
    typer.echo("\nplugins:")
    typer.echo(f"  entry-points loaded: {rep.loaded or '—'}")
    if rep.failed:
        for name, err in rep.failed.items():
            typer.echo(f"  BROKEN plugin '{name}': {err}")
    for kind in KINDS:
        typer.echo(f"  {kind:<12} {', '.join(available(kind)) or '—'}")


@app.command()
def demo(
    out: str = typer.Option("blob_demo", help="output dataset directory"),
    n_frames: int = typer.Option(60, help="frames per episode"),
    seed: int = typer.Option(0, help="base RNG seed"),
) -> None:
    """Generate a blobworld demo dataset (episodes + spec.yaml + labels.json)."""
    from woracle.testing.blobworld import write_dataset

    refs = write_dataset(out, seed=seed, n_frames=n_frames)
    typer.echo(f"wrote {len(refs)} episodes -> {out}/  (spec: {out}/spec.yaml)")


@app.command()
def grade(
    rollouts: str = typer.Option(..., help="directory of episode dirs"),
    spec: str = typer.Option(..., help="TaskSpec YAML path"),
    out: str = typer.Option("woracle_out", help="output directory"),
    profile: str = typer.Option("auto", help="grading profile (auto|blobworld)"),
) -> None:
    """Grade rollouts against a task spec; writes grade cards + manifest."""
    import woracle as w

    cards = w.grade(rollouts, spec, out_dir=out, profile=profile)
    n_pass = sum(1 for c in cards if c.success.verdict == "pass")
    n_fail = sum(1 for c in cards if c.success.verdict == "fail")
    n_abst = sum(1 for c in cards if c.success.verdict == "abstain")
    typer.echo(f"graded {len(cards)} rollouts: {n_pass} pass / {n_fail} fail / {n_abst} abstain")
    for c in cards:
        prog = c.channel("progress.goal_distance")
        pv = f"{prog.value:.3f}" if prog and prog.value is not None else "—"
        typer.echo(
            f"  {c.rollout_id:<16} gate={c.gate.verdict:<12} "
            f"success={c.success.verdict:<8} progress={pv}"
        )
    typer.echo(f"cards -> {os.path.join(out, 'cards')}/   manifest -> {out}/manifest.json")


@app.command()
def report(
    cards: str = typer.Option(..., help="directory containing grade-card JSONs"),
    out: str = typer.Option("", help="write markdown leaderboard here"),
    golds: str = typer.Option("", help="JSON of {rollout_id: true_success} for PPI"),
    html: str = typer.Option("", help="write self-contained HTML report here"),
) -> None:
    """Build leaderboard + honesty statistics from grade cards."""
    import woracle as w

    board = w.report(cards, out_path=out or None, golds=golds or None, html_path=html or None)
    typer.echo(json.dumps(board.model_dump()["policies"], indent=2))
    for path, label in ((out, "markdown"), (html, "html")):
        if path:
            typer.echo(f"{label} -> {path}")


@app.command()
def compile(
    demos: str = typer.Option(..., help="directory of demo episode dirs"),
    prompt: str = typer.Option(..., help="natural-language task prompt"),
    out: str = typer.Option(..., help="output spec YAML path"),
    name: str = typer.Option("compiled-task", help="spec name"),
) -> None:
    """Compile a TaskSpec from demos (self-tested; refuses if inseparable)."""
    import woracle as w

    spec = w.compile(demos, prompt, name=name, out=out)
    st = spec.spec_provenance.self_test
    typer.echo(
        f"compiled '{spec.name}' -> {out}  (self-test: {st.demos_passed}/{st.demos_total} "
        f"demos pass, {st.negatives_failed}/{st.negatives_total} negatives fail)"
    )


if __name__ == "__main__":
    app()
