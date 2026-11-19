#!/usr/bin/env bash
# Gate runner for the LeetGPU kernel workflow (AGENTS.md workflows A/B).
#
#   bash tools/gate.sh <kernel> [--processes N] [--no-build] [--baseline] [--min-gain PCT]
#
#   --processes N    independent harness processes (default 3)
#   --no-build       skip B1 (read-only re-runs only; never for a verdict)
#   --baseline       record a new baseline instead of judging against BENCH.json
#   --min-gain PCT   required median improvement (default 8)
#
# Gates:
#   B1 build        python3 setup.py build_ext --inplace                  exit 0
#   B2 correctness  python3 -m harness check <kernel>                     exit 0
#   B3 performance  python3 -m harness run --procs N, then compare        judged
#   B4 hygiene      git status --porcelain                                only expected paths
#   B5 ledger       append BENCH.json + refresh BENCH.md                  only on PASS
#
# Never commits or pushes.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

NAME=""
PROCS=3
DO_BUILD=1
BASELINE=0
MIN_GAIN=8

while [ $# -gt 0 ]; do
    case "$1" in
        --processes) PROCS="$2"; shift 2 ;;
        --no-build)  DO_BUILD=0; shift ;;
        --baseline)  BASELINE=1; shift ;;
        --min-gain)  MIN_GAIN="$2"; shift 2 ;;
        -h|--help)   sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*)          echo "unknown option: $1" >&2; exit 2 ;;
        *)           NAME="$1"; shift ;;
    esac
done

if [ -z "$NAME" ]; then
    echo "usage: bash tools/gate.sh <kernel> [--processes N] [--no-build] [--baseline] [--min-gain PCT]" >&2
    exit 2
fi

TMP="$(mktemp -d /tmp/leetgpu-gate.XXXXXX)" || exit 1
trap 'rm -rf "$TMP"' EXIT

PASS=0
FAIL=0
note() { printf '%s\n' "$*"; }
ok()   { PASS=$((PASS+1)); note "PASS  $*"; }
bad()  { FAIL=$((FAIL+1)); note "FAIL  $*"; }

CANON="$(python3 -c '
import sys
from harness import registry
try:
    canonical, _ = registry.resolve(sys.argv[1])
except KeyError:
    sys.exit(2)
print(canonical)
' "$NAME" 2>/dev/null)"
if [ -z "$CANON" ]; then
    bad "B0  registry: '$NAME' has no Spec in harness/kernels.py"
    note ""
    note "RESULT: FAIL ($FAIL failed) — add a Spec first (see AGENTS.md workflow A)"
    exit 1
fi
NAME="$CANON"
note "# kernel=$NAME procs=$PROCS"

# A handful of Specs are named after the legacy test.py function rather than the
# source file (softmax_1D lives in src/softmax.cu). B4 whitelists the source that
# the kernel actually lives in; add entries here when a name differs.
SRC="$NAME"
case "$NAME" in
    softmax_1D) SRC="softmax" ;;
esac

# ---------------------------------------------------------------- B1 build
if [ "$DO_BUILD" -eq 1 ]; then
    if python3 setup.py build_ext --inplace >"$TMP/build.log" 2>&1; then
        grep -m1 '^\[leetgpu\]' "$TMP/build.log" || true
        ok "B1 build"
    else
        bad "B1 build (tail below; full log tail in $TMP/build.log)"
        tail -20 "$TMP/build.log" >&2
    fi
else
    note "SKIP  B1 build (--no-build)"
fi

# ---------------------------------------------------------------- B2 correctness
if python3 -m harness check "$NAME" >"$TMP/check.log" 2>&1; then
    grep -E '^(PASS|FAIL|SKIP)' "$TMP/check.log" || true
    ok "B2 correctness (harness check)"
else
    KNOWN="$(python3 - "$NAME" <<'PY' 2>/dev/null
import sys
sys.path.insert(0, "tools")
from verify_specs import KNOWN_FAIL
print(KNOWN_FAIL.get(sys.argv[1], ""))
PY
)"
    if [ -n "$KNOWN" ]; then
        bad "B2 correctness (KNOWN kernel defect, recorded in tools/verify_specs.py)"
        printf '  %s\n' "$KNOWN" >&2
    else
        bad "B2 correctness"
    fi
    tail -5 "$TMP/check.log" >&2
fi

# ---------------------------------------------------------------- B3 performance
CAND="$TMP/cand.json"
if python3 -m harness run "$NAME" --procs "$PROCS" --ref torch --json "$CAND" \
        >"$TMP/bench.out" 2>"$TMP/bench.err"; then
    grep -E '^#' "$TMP/bench.err" | tail -8 || true
    MED="$(python3 -c "import json;print(f\"{json.load(open('$CAND'))[-1]['agg']['median']:.3f}\")" 2>/dev/null)"
else
    bad "B3 performance (measurement failed; tail below)"
    tail -10 "$TMP/bench.err" >&2
    MED=""
fi

BASE="$TMP/base.json"
HAS_BASE=0
python3 - "$NAME" "$BASE" <<'PY' || true
import json, sys
kernel, out = sys.argv[1], sys.argv[2]
try:
    records = json.load(open("BENCH.json"))
except FileNotFoundError:
    sys.exit(0)
prev = [r for r in records if r.get("kernel") == kernel]
if prev:
    json.dump([prev[-1]], open(out, "w"))
PY
[ -s "$BASE" ] && HAS_BASE=1

if [ -n "$MED" ]; then
    if [ "$BASELINE" -eq 1 ] || [ "$HAS_BASE" -eq 0 ]; then
        if [ "$BASELINE" -eq 1 ]; then
            ok "B3 performance  median=${MED}ms (baseline mode: recorded, no gain check)"
        else
            ok "B3 performance  median=${MED}ms (no prior record for this kernel -> baseline)"
        fi
        RECORD=1
    else
        if python3 -m harness compare "$BASE" "$CAND" --min-gain "$MIN_GAIN" >"$TMP/cmp.log" 2>&1; then
            ok "B3 performance  $(grep -E '^(ACCEPT|REJECT)' "$TMP/cmp.log" | tail -1)"
            RECORD=1
        else
            bad "B3 performance  $(grep -E '^(ACCEPT|REJECT)' "$TMP/cmp.log" | tail -1)"
            grep -E '^(base|cand):' "$TMP/cmp.log" >&2 || true
            RECORD=0
        fi
    fi
else
    RECORD=0
fi

# ---------------------------------------------------------------- B5 ledger (on PASS only)
ROW=""
if [ "${RECORD:-0}" -eq 1 ]; then
    cp "$CAND" "$TMP/append.json"
    python3 - "$TMP/append.json" <<'PY'
import json, sys
from harness import store
record = json.load(open(sys.argv[1]))[-1]
record["note"] = record.get("note") or "gate"
store.append_record("BENCH.json", record)
PY
    python3 -m harness report >/dev/null && ok "B5 ledger (BENCH.json + BENCH.md updated)"
    ROW="$(python3 -c "
import json
from harness import report
recs = json.load(open('BENCH.json'))
print([l for l in report.render_table(recs).splitlines() if l.startswith('| 20')][-1])")"
else
    note "SKIP  B5 ledger (nothing recorded)"
fi

# ---------------------------------------------------------------- B4 hygiene
STATUS="$(git status --porcelain 2>/dev/null)"
if [ -z "$STATUS" ]; then
    ok "B4 hygiene (clean tree)"
else
    UNEXPECTED="$(printf '%s\n' "$STATUS" | sed -e 's/^[ MARCUD?!][ MARCUD?!] //' | while read -r p; do
        case "$p" in
            "src/$SRC.cu"|src/bindings.cu|src/cuda_kernels.cuh|src/utils.cuh|bench.py|BENCH.md|BENCH.json|AGENTS.md|README.md|setup.py|.gitignore|tools/*|harness/*|"harness/__pycache__/"*|"tools/__pycache__/"*) ;;
            *) printf '%s\n' "$p" ;;
        esac
    done)"
    if [ -n "$UNEXPECTED" ]; then
        bad "B4 hygiene (unexpected paths; see below)"
        printf '%s\n' "$UNEXPECTED" >&2
    else
        ok "B4 hygiene (only expected paths)"
    fi
fi

# ---------------------------------------------------------------- summary
note ""
if [ "$FAIL" -eq 0 ]; then
    note "RESULT: PASS ($PASS gates)"
    [ -n "$ROW" ] && { note ""; note "BENCH.md row (already written to the table):"; note "$ROW"; }
    exit 0
else
    note "RESULT: FAIL ($FAIL failed, $PASS passed)"
    exit 1
fi
