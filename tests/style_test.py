"""Repository-local style and syntax policy."""

from __future__ import annotations

import ast
from pathlib import Path
import sys

import agent_harness


def main() -> int:
    root = Path(agent_harness.__file__).resolve().parent.parent
    paths = sorted((root / "agent_harness").rglob("*.py"))
    paths.extend(sorted((root / "cmd").rglob("*.py")))
    failures: list[str] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.IfExp):
                failures.append(str(path) + ": ternary expression")
            if isinstance(node, ast.If):
                if len(node.body) != 1:
                    continue
                body = node.body[0]
                if not isinstance(body, ast.Return):
                    continue
                if node.lineno == body.lineno:
                    failures.append(str(path) + ": single-line return guard")
    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
