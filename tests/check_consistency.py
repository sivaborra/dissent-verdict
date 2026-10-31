#!/usr/bin/env python3
"""CI consistency checks for panel-judge (stdlib only, offline, no secrets).

Checks:
 1. pyproject.toml version == panel_judge/__init__.py fallback constant
    == panel_judge.core.TOOL_VERSION suffix.
 2. docs/HEURISTIC-MAPPING.md header version == code constant.
 3. Every import in panel_judge/ resolves to the standard library.
 4. panel_judge/core.py and panel_judge/cli.py contain no network code
    (no urllib/socket/http.client/subprocess import or use).
 5. No secret-looking string in panel_judge/ or tests/ (private-key
    headers, sk- style key patterns). Naming the env var
    OPENROUTER_API_KEY is allowed; a VALUE is never present.

Usage: python3 tests/check_consistency.py   (exit 0 pass, 1 fail)
"""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAILURES = []


def fail(message):
    FAILURES.append(message)
    print(f"FAIL: {message}")


def main():
    # 1. Version consistency.
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    if not match:
        fail("pyproject.toml has no version")
        py_version = None
    else:
        py_version = match.group(1)
    init_src = (ROOT / "panel_judge" / "__init__.py").read_text(encoding="utf-8")
    fallback = re.search(r'__version__\s*=\s*"([^"]+)"', init_src)
    if not fallback:
        fail("panel_judge/__init__.py has no fallback __version__ constant")
    elif py_version and fallback.group(1) != py_version:
        fail(f"__init__ fallback {fallback.group(1)} != pyproject {py_version}")
    sys.path.insert(0, str(ROOT))
    import panel_judge.core as core
    import panel_judge
    if core.TOOL_VERSION != f"panel-judge {py_version}":
        fail(f"core.TOOL_VERSION {core.TOOL_VERSION!r} != 'panel-judge {py_version}'")
    if panel_judge.__version__ != py_version:
        # When installed, metadata wins; in a repo checkout the fallback
        # must match. A mismatch here means the installed distribution
        # is stale relative to this checkout -- report, do not hide.
        fail(f"panel_judge.__version__ {panel_judge.__version__!r} != pyproject {py_version}")

    # 2. Heuristic annex header == code constant.
    annex = (ROOT / "docs" / "HEURISTIC-MAPPING.md").read_text(encoding="utf-8")
    header = re.search(r"Heuristic mapping version:\s*(heuristic-mapping-v\d+)", annex)
    if not header:
        fail("HEURISTIC-MAPPING.md has no 'Heuristic mapping version:' header")
    elif header.group(1) != core.HEURISTIC_MAPPING_VERSION:
        fail(f"annex header {header.group(1)} != code constant {core.HEURISTIC_MAPPING_VERSION}")

    # 3. Stdlib-only imports.
    stdlib = set(sys.stdlib_module_names)
    for path in sorted((ROOT / "panel_judge").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top not in stdlib and top != "panel_judge":
                        fail(f"{path.name}: non-stdlib import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top = node.module.split(".")[0]
                    if node.level == 0 and top not in stdlib and top != "panel_judge":
                        fail(f"{path.name}: non-stdlib import from {node.module}")

    # 4. Core + CLI are network-free and subprocess-free (import-level:
    #    urllib.parse for URL parsing in core is allowed; urllib.request,
    #    socket, http.client and subprocess are not).
    banned_imports = {"urllib.request", "urllib.error", "socket", "http.client",
                      "subprocess", "requests"}
    for name in ("core.py", "cli.py"):
        tree = ast.parse((ROOT / "panel_judge" / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in banned_imports:
                        fail(f"panel_judge/{name} imports network/subprocess module {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module in banned_imports:
                fail(f"panel_judge/{name} imports from network/subprocess module {node.module}")

    # 5. No secret-looking strings in code/tests (docs and out/ excluded:
    #    docs may name the env var; out/ run-meta contains no key by design).
    secret_patterns = [
        (re.compile(r"-----BEGIN .*PRIVATE KEY-----"), "private key header"),
        (re.compile(r"sk-[A-Za-z0-9]{16,}"), "sk- key pattern"),
        (re.compile(r"sk-or-[A-Za-z0-9]{8,}"), "OpenRouter-style key pattern"),
    ]
    for base in (ROOT / "panel_judge", ROOT / "tests"):
        for path in sorted(base.glob("*.py")):
            if path.name == "check_consistency.py":
                continue  # the check's own patterns would self-match
            src = path.read_text(encoding="utf-8")
            for pattern, label in secret_patterns:
                if pattern.search(src):
                    fail(f"{path.relative_to(ROOT)}: secret-looking string ({label})")

    if FAILURES:
        print(f"{len(FAILURES)} consistency failure(s)")
        return 1
    print("Consistency checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
