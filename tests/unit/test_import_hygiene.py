"""The kernel rule, enforced: `import woracle` is light and torch-free."""

from __future__ import annotations

import subprocess
import sys
import time


def test_import_is_torch_free_and_light() -> None:
    code = (
        "import sys; import woracle; "
        "banned = [m for m in ('torch','cv2','transformers','PIL') if m in sys.modules]; "
        "assert not banned, f'kernel rule violated: {banned} imported at top level'; "
        "assert woracle.__version__"
    )
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    assert proc.returncode == 0, proc.stderr
    # Aspiration is <200ms warm; budget generously for cold CI runners.
    assert elapsed < 5.0, f"import woracle took {elapsed:.2f}s — laziness regressed"


def test_lazy_api_resolves() -> None:
    import woracle

    assert callable(woracle.grade)
    assert callable(woracle.load_spec)
    assert "grade" in dir(woracle)


def test_unknown_attribute_raises_attribute_error() -> None:
    import woracle

    try:
        woracle.does_not_exist  # noqa: B018
        raise AssertionError("expected AttributeError")
    except AttributeError:
        pass
