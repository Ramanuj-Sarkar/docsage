"""Run the integration and eval suites against live services.

Usage: .venv/bin/python scripts/run_evals.py

Requires credentials in .env (OPENAI_API_KEY, LANGCHAIN_API_KEY,
LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY). Spends real API credits.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-m", "integration or eval", "-v"],
        cwd=ROOT,
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
