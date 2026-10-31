"""Generic, file-based AI-panel judge library: panels in,
verdict + report out. Independent checkpoints, explicit scores
with reasons, and a disclaimer, generalised to any panel,
any question, and any judge.

Version single source of truth: pyproject.toml. __version__ reads the
installed distribution metadata when installed; otherwise it falls back
to the constant below, which the CI consistency check requires to equal
pyproject.toml exactly. The fallback is a copy, not an independent source.
"""

try:
    from importlib.metadata import version as _dist_version

    __version__ = _dist_version("panel-judge")
except Exception:
    __version__ = "1.1.0"
