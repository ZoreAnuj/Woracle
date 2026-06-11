"""S1 spec compiler: demos + prompt -> TaskSpec (or an honest REFUSE).

v1 is the relational compiler (model-free entity induction); VLM-assisted
role naming and embedding-based phase discovery layer on top of the same
contracts later.
"""

from woracle.compiler.compile import compile_spec
from woracle.compiler.entities import Entity, classify_relational, induce_entities
from woracle.compiler.negatives import MINT_KINDS, mint_negatives
from woracle.compiler.selftest import SelfTestOutcome, run_selftest

__all__ = [
    "MINT_KINDS",
    "Entity",
    "SelfTestOutcome",
    "classify_relational",
    "compile_spec",
    "induce_entities",
    "mint_negatives",
    "run_selftest",
]
