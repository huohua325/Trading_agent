from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def replay_from_audit(date_dir: str) -> Dict:
    p = Path(date_dir)
    if not p.exists() or not p.is_dir():
        return {"ok": False, "reason": "audit_dir_not_found", "files": []}
    files = sorted(p.glob("*.jsonl"))
    recs: List[Dict] = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    recs.append(json.loads(line))
                except Exception:
                    continue
    summary = {
        "symbols": sorted({r.get("symbol") for r in recs if r.get("symbol")}),
        "orders_total": sum(len(r.get("orders", [])) for r in recs),
        "analyzer_count": sum(1 for r in recs if r.get("analyzer")),
        "decision_count": sum(1 for r in recs if r.get("decision")),
        "files": [str(f) for f in files],
    }
    return {"ok": True, "summary": summary} 