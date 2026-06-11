"""Single source of the package version.

Kept as a plain module (not importlib.metadata) so `import woracle` works from a
source checkout without installation. Switched to hatch-vcs at first release (P6).
"""

__version__ = "0.0.1.dev0"
