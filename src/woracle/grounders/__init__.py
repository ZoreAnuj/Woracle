"""S2 grounders. The Grounder protocol lives in woracle.protocols.

`openvocab.gdino_sam` (GroundingDINO-tiny + SAM, transformers ports) registers
on import; its MODEL imports happen lazily at call time behind the [ground]
extra. The blobworld reference grounder lives in woracle.testing.plugins.
(Package named `grounders` — `ground` is the public API verb in woracle.api.)
"""

from woracle.grounders.openvocab import OpenVocabGrounder
from woracle.grounders.passthrough import PassthroughGrounder
from woracle.grounders.relational import RelationalMotionGrounder

__all__ = ["OpenVocabGrounder", "PassthroughGrounder", "RelationalMotionGrounder"]
