"""Benchmark CoPhy adapter preprocessing configurations."""

from __future__ import annotations

import sys

from lowm.data.cophy_adapter import main


if __name__ == "__main__":
    if "--benchmark-only" not in sys.argv:
        sys.argv.append("--benchmark-only")
    main()
