from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
PRIVATE = ARTIFACTS / "private"
RAW = ROOT / "data" / "raw"
SEED = 42
LIMITATIONS = [
    "Public SECOM data, not Seagate data; retrospective quality triage only.",
    "Measurement meanings, units, and availability before final testing are unknown.",
    "Priority scores are not calibrated failure probabilities.",
    "Model contributions and unusual measurements are associations, not physical causes.",
    "Small failure counts and temporal dependence limit generalization.",
    "Generated briefs require human review; no autonomous quality decision.",
]


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_json(path):
    return json.loads(Path(path).read_text())

