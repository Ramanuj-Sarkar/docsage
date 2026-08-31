"""Prepare the corpus and sanity-check retrieval for DocSage.

Usage: .venv/bin/python scripts/seed_vectorstore.py
(equivalent to ``docsage seed``)
"""

from __future__ import annotations

import sys

from docsage.config import get_settings
from docsage.main import cmd_seed

if __name__ == "__main__":
    sys.exit(cmd_seed(get_settings()))
