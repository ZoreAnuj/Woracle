# Contributing to Woracle

## The two rules that are never negotiable

1. **The kernel rule.** `import woracle` pulls numpy + pydantic (+ pyyaml/typer) only.
   Heavy dependencies (torch, cv2, transformers, trackers, depth models) live behind
   extras, import lazily *at call time*, and fail with a `MissingDependencyError` that
   names the exact extra. CI asserts the torch-leak invariant on every PR.
   Corollary: `gate/`, `channels/`, `stats/`, `report/` never import model libraries —
   they are pure functions over artifacts written by `ground/` and `compile/`.

2. **The honesty rule.** Evidence failure is *data*, not an exception:
   "role unbound", "track lost", "object vanished" become recorded values
   (`bound=False`, `status="evidence_missing"`) that flow into gate/abstain accounting.
   Exceptions are reserved for infra failures (`InfraError`: OOM, timeout, download),
   which are retryable and never recorded as evidence about a rollout.

## Contracts discipline

- Every persisted model carries `schema_version`; changing one requires (same commit):
  bump version → add a migration in `contracts/migrations.py` → regenerate goldens
  (`WORACLE_REGEN_SCHEMAS=1 uv run pytest tests/golden`) → commit all three.
- Arrays never go in JSON. Sidecars: `.npz` (P0), parquet/safetensors (P1+), mp4 (video).
- No pickle. Anywhere. Ever.

## Components

- One small Protocol per kind (`woracle/protocols.py`): Grounder, GateSignal, Channel, Judge.
- `name` is stable; `version` participates in cache keys — bump it on ANY behavior change.
- Determinism is part of the contract (cache correctness depends on it).
- Register first-party with `@register(kind, name)`; third-party via the
  `woracle.plugins` entry-point group (a zero-arg registrar callable).
- Run the conformance suite against your component
  (`woracle.testing.conformance.channel_checks(...)` etc.) in your own CI.

## Dev workflow

```bash
uv sync --group dev
uv run pytest                # CPU-hermetic; network blocked by pytest-socket
uv run ruff check . && uv run ruff format .
uv run pyright
RUN_SLOW=1 uv run pytest -m slow   # nightly lane locally
```

PR CI must stay green with core+test deps only — if your test needs a checkpoint,
a download, or a GPU, mark it `@pytest.mark.slow` / `@pytest.mark.gpu`.
