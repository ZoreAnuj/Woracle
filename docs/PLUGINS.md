# Writing a woracle plugin

Woracle components are **structural** (typing.Protocol): you implement the
methods, you never import a base class. Four kinds: `grounder`, `gate_signal`,
`channel`, `judge` (see `src/woracle/protocols.py` for the exact contracts).

## The contract every component shares

1. **Deterministic** given identical inputs + params (cache correctness).
2. **Evidence failure is data**: return `bound=False` / `status="evidence_missing"`
   with a `reason`. Never raise for missing evidence.
3. **Infra failure raises `woracle.errors.InfraError`** (OOM, timeouts,
   downloads): retryable, never recorded as evidence.
4. `name` is stable; `version` participates in cache keys — bump on ANY
   behavior change.
5. Grounders expose a **`params` dict property** (canonical-JSON-able) — it
   feeds the ground-stage cache key; omit it and differently-configured
   instances will silently share cache entries.

## Minimal channel

```python
from woracle.contracts import ChannelCaps, ChannelScore
from woracle.registry import register

@register("channel", "mylab.smoothness")
class SmoothnessChannel:
    name = "mylab.smoothness"
    version = "0.1.0"
    caps = ChannelCaps(reference_free=True, needs_tracks=True, verdict_eligible=False)

    def score(self, grounded, spec) -> ChannelScore:
        from woracle.channels.verdict import role_data
        rd = role_data(grounded).get("carried_object")
        if rd is None or rd.track is None:
            return ChannelScore(channel=self.name, status="evidence_missing",
                                reason="no carried_object track")
        ...
        return ChannelScore(channel=self.name, value=v, confidence=c)
```

`verdict_eligible=False` unless your channel evaluates the spec's success
predicates — ranking evidence is structurally walled off from verdicts.

## Distribution

```toml
# pyproject.toml of your plugin package
[project.entry-points."woracle.plugins"]
mylab = "mylab_woracle.register:register_all"
```

`register_all()` is a zero-arg callable performing your `@register` imports.
A broken plugin is reported by `woracle doctor`, never fatal.
Set `WORACLE_DISABLE_PLUGINS=1` to skip autoloading.

## Conformance (run this in YOUR CI)

```python
import pytest
from woracle.testing.conformance import channel_checks

@pytest.mark.parametrize("name,check", channel_checks(SmoothnessChannel))
def test_conformance(name, check):
    check()
```

The suite runs your component against deterministic blobworld fixtures —
including a **degenerate bundle** (all roles unbound, zero artifacts) where
your component must answer with recorded missing-evidence, not an exception
and not a confident value. That check is the honesty floor for the ecosystem.
