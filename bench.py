#!/usr/bin/env python3
"""Compatibility wrapper: `python3 bench.py <kernel> [flags]` == `python3 -m harness run ...`.

Kept so `tools/gate.sh` and existing habits keep working; the real logic lives in
harness/. Arguments are forwarded verbatim, so --ncu / --markdown / --ref torch /
--print-test-fn / --list all behave as before.
"""

import sys

from harness.cli import main

if __name__ == "__main__":
    sys.exit(main(["run", *sys.argv[1:]]))
