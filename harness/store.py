"""JSON result store: one record per measurement, pinned to the exact binary."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

SCHEMA = 1


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=repo_root(), capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def so_sha256() -> str | None:
    """Pin a measurement to the .so that produced it (kills stale-binary doubt)."""
    files = sorted(repo_root().glob("leetgpu*.so"))
    if not files:
        return None
    digest = hashlib.sha256()
    with open(files[0], "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def gpu_info() -> dict:
    info = {"name": "unknown", "sm_count": None, "max_sm_clock_mhz": None, "driver": ""}
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["name"] = props.name
            info["sm_count"] = props.multi_processor_count
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,clocks.max.sm", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        parts = [p.strip() for p in out.stdout.strip().split(",")]
        if len(parts) >= 2:
            info["driver"] = parts[0]
            info["max_sm_clock_mhz"] = parts[1].replace(" MHz", "")
    except Exception:
        pass
    return info


def fingerprint() -> dict:
    import torch
    return {
        "git": {
            "commit": _git("rev-parse", "--short", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
        },
        "gpu": gpu_info(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "so_sha256": so_sha256(),
    }


def build_record(*, kernel: str, shape: str, dtype: str, protocol: dict,
                 procs: Sequence, agg: dict, ref: dict | None = None,
                 note: str = "") -> dict:
    record = {
        "schema": SCHEMA,
        "ts": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "kernel": kernel,
        "shape": shape,
        "dtype": dtype,
        "protocol": protocol,
        "procs": [p.as_dict() if hasattr(p, "as_dict") else dict(p) for p in procs],
        "agg": agg,
        "ref": ref,
        "note": note,
    }
    record.update(fingerprint())
    return record


def load_records(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    text = path.read_text().strip()
    if not text:
        return []
    data = json.loads(text)
    return data if isinstance(data, list) else [data]


def append_record(path: str | Path, record: dict) -> None:
    path = Path(path)
    records = load_records(path)
    records.append(record)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")


def last_record(records: Sequence[dict], kernel: str) -> dict | None:
    for record in reversed(records):
        if record.get("kernel") == kernel:
            return record
    return None
