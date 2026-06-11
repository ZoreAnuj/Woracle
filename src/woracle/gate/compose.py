"""Gate composition: signals + policy -> three-way gradeability verdict.

Pure function over contract data (no models, no I/O). Thresholds live in the
policy object, never in signal code (ARCH decision 6). P2 replaces the static
thresholds with conformally calibrated ones; the composition shape stays.
"""

from __future__ import annotations

from woracle.contracts import (
    GatePolicy,
    GateReport,
    GateSignalValue,
    GateThreshold,
    GroundedRollout,
    digest_json,
)

# P0 default policy for the blob profile. Calibration replaces numbers in P2.
DEFAULT_POLICY = GatePolicy(
    thresholds=[
        GateThreshold(signal="permanence", ungradeable_below=0.5, degraded_below=0.8),
        GateThreshold(signal="binding_health", ungradeable_below=0.2, degraded_below=0.5),
        GateThreshold(signal="motion_sanity", degraded_below=0.2),
    ],
    required_signals=["binding_health", "permanence"],
    require_role_bindings=True,
)


def compose_gate(
    grounded: GroundedRollout,
    signals: list[GateSignalValue],
    policy: GatePolicy = DEFAULT_POLICY,
) -> GateReport:
    verdict = "gradeable"
    reasons: list[str] = []

    def downgrade(to: str, why: str) -> None:
        nonlocal verdict
        order = {"gradeable": 0, "degraded": 1, "ungradeable": 2}
        if order[to] > order[verdict]:
            verdict = to
        reasons.append(why)

    if policy.require_role_bindings:
        unbound = [b.role for b in grounded.bindings if not b.bound]
        if unbound:
            downgrade(
                "ungradeable",
                f"required role(s) not bound: {', '.join(sorted(unbound))}",
            )

    for sig in signals:
        if sig.status == "evidence_missing":
            if sig.name in policy.required_signals:
                downgrade("ungradeable", f"signal '{sig.name}' has no evidence: {sig.reason}")
            else:
                downgrade("degraded", f"signal '{sig.name}' has no evidence: {sig.reason}")
            continue
        th = policy.threshold(sig.name)
        if th is None or sig.value is None:
            continue
        if th.ungradeable_below is not None and sig.value < th.ungradeable_below:
            downgrade(
                "ungradeable",
                f"{sig.name}={sig.value:.3f} < {th.ungradeable_below} ({sig.reason or 'below hard floor'})",
            )
        elif th.degraded_below is not None and sig.value < th.degraded_below:
            downgrade("degraded", f"{sig.name}={sig.value:.3f} < {th.degraded_below}")

    return GateReport(
        verdict=verdict,  # type: ignore[arg-type]
        signals=signals,
        reasons=reasons,
        policy_digest=digest_json(policy.model_dump()),
    )
