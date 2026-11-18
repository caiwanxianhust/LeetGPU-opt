"""Render BENCH.md's table from BENCH.json records (never overwrites prose)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

START = "<!-- BENCH:TABLE:START -->"
END = "<!-- BENCH:TABLE:END -->"

HEADER = (
    "| 日期 | commit | kernel | shape | dtype | median ms | min ms | vs 上一版 | vs PyTorch | 改动 |\n"
    "|---|---|---|---|---|---|---|---|---|---|"
)


def _row(record: dict, previous: dict | None) -> str:
    agg = record["agg"]
    if previous is None:
        vs_prev = "基线"
    else:
        base = float(previous["agg"]["median"])
        cur = float(agg["median"])
        vs_prev = f"{(base - cur) / base * 100:+.1f}% ({cur - base:+.3f} ms)"
    ref = record.get("ref") or {}
    speedup = ref.get("speedup")
    vs_torch = f"{float(speedup):.2f}x" if speedup else "待测"
    note = (record.get("note") or "—").replace("|", "\\|")
    git = record.get("git") or {}
    commit = git.get("commit", "?") + ("*" if git.get("dirty") else "")
    return (
        f"| {record['ts'][:10]} | {commit} "
        f"| {record['kernel']} | {record.get('shape', '-')} | {record.get('dtype', '-')} "
        f"| {float(agg['median']):.3f} | {float(agg['min']):.3f} "
        f"| {vs_prev} | {vs_torch} | {note} |"
    )


def render_table(records: Sequence[dict]) -> str:
    rows: list[str] = []
    seen: dict[str, dict] = {}
    for record in records:
        rows.append(_row(record, seen.get(record["kernel"])))
        seen[record["kernel"]] = record
    return "\n".join([HEADER, *rows])


def update_markdown(path: str | Path, records: Sequence[dict]) -> str:
    path = Path(path)
    table = render_table(records)
    if path.exists():
        text = path.read_text()
        if START in text and END in text:
            head, rest = text.split(START, 1)
            _, tail = rest.split(END, 1)
            path.write_text(f"{head}{START}\n{table}\n{END}{tail}")
            return table
    path.write_text(f"# BENCH\n\n{START}\n{table}\n{END}\n")
    return table
