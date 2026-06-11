"""S4 grade channels.

Importing this package registers the first-party channels: the generic
predicate engine consumers, trajectory DTW, ordered phase coverage, the
TL-DTMC satisfaction probability, and the GVL progress channel.
"""

from woracle.channels import phases, tlcheck, trajectory, verdict  # noqa: F401
from woracle.channels.predicates import PredicateResult, RoleData, eval_conjunction, eval_predicate
from woracle.judges import progress_gvl  # noqa: F401

__all__ = ["PredicateResult", "RoleData", "eval_conjunction", "eval_predicate"]
