"""S4 grade channels. P0: the generic predicate engine; P3 adds progress/phase/trajectory."""

from woracle.channels.predicates import PredicateResult, RoleData, eval_conjunction, eval_predicate

__all__ = ["PredicateResult", "RoleData", "eval_conjunction", "eval_predicate"]
