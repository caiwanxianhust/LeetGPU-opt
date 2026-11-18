#!/usr/bin/env bash
# Gate runner for the LeetGPU kernel workflow (AGENTS.md workflows A/B).
#
#   bash tools/gate.sh <kernel> [--processes N] [--no-build] [--baseline] [--min-gain PCT]
#
#   --processes N    independent bench.py processes (default 3)
#   --no-build       skip B1 (only for read-only re-runs; never for a verdict)
#   --baseline       treat this run as a new baseline: skip the B3 gain comparison
#   --min-gain PCT   required median improvement in percent (default 8)
#
# Gates:
#   B1 build      python3 setup.py build_ext --inplace        exit 0
#   B2 correctness python3 -c 'import test; test.<fn>()'       exit 0 AND no mismatch marker
#   B3 performance bench.py x N processes                     median-of-medians gain >= min-gain
#   B4 hygiene    git status --porcelain                      only expected paths
#
# Prints one paste-ready BENCH.md row on success. Never commits or pushes.
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

# ---------------------------------------------------------------- resolve test fn
TEST_FN="$(python3 bench.py "$NAME" --print-test-fn 2>/dev/null)" || true
if [ -z "$TEST_FN" ]; then
    bad "B0  registry: '$NAME' not in bench.py KERNELS (or has no test_fn)"
    note ""
    note "RESULT: FAIL ($FAIL failed) — add the kernel to bench.py first"
    exit 1
fi
note "# kernel=$NAME test_fn=$TEST_FN"

# ---------------------------------------------------------------- B1 build
if [ "$DO_BUILD" -eq 1 ]; then
    if python3 setup.py build_ext --inplace >"$TMP/build.log" 2>&1; then
        ok "B1 build"
    else
        bad "B1 build (see $TMP/build.log; tail below)"
        tail -20 "$TMP/build.log" >&2
    fi
else
    note "SKIP  B1 build (--no-build)"
fi

# ---------------------------------------------------------------- B2 correctness
if [ "$FAIL" -eq 0 ] || [ "$DO_BUILD" -eq 0 ]; then
    if python3 -c "import test; test.$TEST_FN()" >"$TMP/test.log" 2>&1; then
        if grep -qiE 'mismatch|Maximum difference' "$TMP/test.log"; then
            bad "B2 correctness (comparison reported a mismatch)"
            grep -iE 'mismatch|Maximum difference' "$TMP/test.log" | head -5 >&2
        else
            ok "B2 correctness"
        fi
    else
        bad "B2 correctness (non-zero exit; tail below)"
        tail -25 "$TMP/test.log" >&2
    fi
fi

# ---------------------------------------------------------------- B3 performance
CSV="$TMP/bench.csv"
: >"$CSV"
i=1
while [ "$i" -le "$PROCS" ]; do
    if line="$(python3 bench.py "$NAME" 2>"$TMP/bench.$i.err")"; then
        printf '%s\n' "$line" >>"$CSV"
    else
        bad "B3 bench process $i failed (see $TMP/bench.$i.err)"
        break
    fi
    i=$((i+1))
done

AGG=""
if [ -s "$CSV" ]; then
    AGG="$(python3 - "$CSV" <<'PY'
import statistics, sys
rows = [l.strip().split(",") for l in open(sys.argv[1]) if l.strip()]
med = sorted(float(r[3]) for r in rows)
mins = [float(r[4]) for r in rows]
agg = statistics.median(med)
spread = (max(med) - min(med)) / agg * 100 if agg else 0.0
print(f"{agg:.3f},{min(mins):.3f},{spread:.1f},{len(med)}")
PY
)"
fi

if [ -n "$AGG" ]; then
    CUR_MED="${AGG%%,*}"; REST="${AGG#*,}"
    CUR_MIN="${REST%%,*}"; REST2="${REST#*,}"
    SPREAD="${REST2%%,*}"; NPROC="${REST2#*,}"

    SHAPE="$(head -1 "$CSV" | cut -d, -f2)"
    DTYPE="$(head -1 "$CSV" | cut -d, -f3)"

    PREV="$(python3 - "$NAME" <<'PY'
import sys
kernel = sys.argv[1]
prev = ""
try:
    for line in open("BENCH.md"):
        p = [x.strip() for x in line.split("|")]
        if len(p) >= 8 and p[3] == kernel and p[6] not in ("median ms", ""):
            try:
                float(p[6])
                prev = p[6]
            except ValueError:
                pass
except FileNotFoundError:
    pass
print(prev)
PY
)"

    if [ "$BASELINE" -eq 1 ]; then
        VS_PREV="基线"
        ok "B3 performance  median=${CUR_MED}ms min=${CUR_MIN}ms (${NPROC} procs, spread ${SPREAD}%, baseline mode: gain check skipped)"
    elif [ -z "$PREV" ]; then
        VS_PREV="基线"
        ok "B3 performance  median=${CUR_MED}ms min=${CUR_MIN}ms (${NPROC} procs, spread ${SPREAD}%, no prior row in BENCH.md -> baseline)"
    else
        VS_PREV="$(python3 -c "
prev=float('$PREV'); cur=float('$CUR_MED')
d=(prev-cur)/prev*100
print(f'{-d:+.1f}% ({cur-prev:+.3f} ms)')")"
        if python3 -c "import sys; sys.exit(0 if float('$SPREAD') <= float('$MIN_GAIN') else 1)"; then
            if python3 -c "import sys; sys.exit(0 if (float('$PREV')-float('$CUR_MED'))/float('$PREV')*100 >= float('$MIN_GAIN') else 1)"; then
                ok "B3 performance  ${PREV} -> ${CUR_MED} ms (gain >= ${MIN_GAIN}%, ${NPROC} procs, spread ${SPREAD}%)"
            else
                bad "B3 performance  gain < ${MIN_GAIN}% (${PREV} -> ${CUR_MED} ms, ${NPROC} procs, spread ${SPREAD}%)"
            fi
        else
            bad "B3 performance  unstable: cross-process spread ${SPREAD}% > ${MIN_GAIN}% threshold; re-run with more --processes"
        fi
    fi

    # paste-ready row; vs PyTorch filled by a --ref run
    ROW="$(python3 bench.py "$NAME" --ref torch --markdown 2>"$TMP/ref.log" || true)"
    VS_TORCH="$(printf '%s' "$ROW" | awk -F'|' 'NF>=10 {gsub(/ /,"",$10); print $10}')"
    [ -z "$VS_TORCH" ] && VS_TORCH="待测"
    ROW="$(python3 - "$NAME" "$SHAPE" "$DTYPE" "$CUR_MED" "$CUR_MIN" "$VS_PREV" "$VS_TORCH" <<'PY'
import subprocess, sys
from datetime import date
name, shape, dtype, med, mn, vs_prev, vs_torch = sys.argv[1:8]
try:
    commit = subprocess.run(["git","rev-parse","--short","HEAD"],
                            capture_output=True, text=True, timeout=10).stdout.strip()
except Exception:
    commit = "unknown"
print(f"| {date.today().isoformat()} | {commit} | {name} | {shape} | {dtype} "
      f"| {float(med):.3f} | {float(mn):.3f} | {vs_prev} | {vs_torch} | — |")
PY
)"
else
    VS_PREV="?"
    bad "B3 performance (no bench samples collected)"
fi

# ---------------------------------------------------------------- B4 hygiene
STATUS="$(git status --porcelain 2>/dev/null)"
if [ -z "$STATUS" ]; then
    ok "B4 hygiene (clean tree)"
else
    UNEXPECTED="$(printf '%s\n' "$STATUS" | sed -e 's/^[ MARCUD?!][ MARCUD?!] //' | while read -r p; do
        case "$p" in
            "src/$NAME.cu"|src/bindings.cu|src/cuda_kernels.cuh|src/utils.cuh|test.py|bench.py|BENCH.md|AGENTS.md|setup.py|.gitignore|tools/*) ;;
            *) printf '%s\n' "$p" ;;
        esac
    done)"
    if [ -n "$UNEXPECTED" ]; then
        bad "B4 hygiene (unexpected paths; see below)"
        printf '%s\n' "$UNEXPECTED" >&2
    else
        ok "B4 hygiene (only expected paths)"
        note "      note: 工作区有未提交改动，因此下面行里的 commit 不完全对应被测代码；提交后再跑一次确认。"
    fi
fi

# ---------------------------------------------------------------- summary
note ""
if [ "$FAIL" -eq 0 ]; then
    note "RESULT: PASS ($PASS gates)"
    note ""
    note "BENCH.md row (fill the 改动 column):"
    note "${ROW:-}"
    exit 0
else
    note "RESULT: FAIL ($FAIL failed, $PASS passed)"
    exit 1
fi
