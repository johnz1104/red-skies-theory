"""Episode records: write a readable JSON trace and summarize outcomes."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from .engine import CampaignEngine, EpisodeResult
from .serialization import canonical_json


def episode_record(engine: CampaignEngine, result: EpisodeResult) -> dict[str, Any]:
    return {
        "seed": result.seed,
        "config": {
            "agents": list(engine.config.agents),
            "chance_mode": engine.config.chance_mode,
            "npc_policy": engine.config.npc_policy,
        },
        "end_reason": result.end_reason,
        "final_quarter": result.final_quarter,
        "decisions": result.decisions,
        "returns": result.returns,
        "outcomes": result.outcomes,
        "timeline": [list(item) for item in result.timeline],
        "trace": engine.s.trace,
        "trace_hash": result.trace_hash,
        **({"assumptions": result.assumptions} if result.assumptions else {}),
    }


def write_json(path: Path, payload: Any) -> None:
    """Atomically write *payload* as indented canonical JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(canonical_json(payload, indent=2))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
