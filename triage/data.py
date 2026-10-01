from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
import requests

from triage.common import RAW, ARTIFACTS, now, save_json

BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/"
FILES = ("secom.data", "secom_labels.data", "secom.names")


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    sources = []
    for name in FILES:
        response = requests.get(BASE + name, timeout=60)
        response.raise_for_status()
        content = response.content
        (RAW / name).write_bytes(content)
        sources.append({"file": name, "url": BASE + name, "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest()})
    save_json(ARTIFACTS / "provenance.json", {
        "downloaded_at": now(), "source": "UCI SECOM", "classification": "public historical observations",
        "license": "CC BY 4.0", "attribution": "McCann, M. & Johnston, A. (2008). SECOM. DOI:10.24432/C54305",
        "dataset_url": "https://archive.ics.uci.edu/dataset/179/secom", "sources": sources,
        "timestamp_timezone": "Unspecified by publisher; retained as naive source time",
    })


def load_data():
    x = pd.read_csv(RAW / "secom.data", sep=r"\s+", header=None)
    labels = pd.read_csv(RAW / "secom_labels.data", sep=r"\s+", header=None)
    if x.shape != (1567, 590) or len(labels) != len(x):
        raise ValueError(f"Unexpected source shape: {x.shape}, labels={len(labels)}")
    if not set(labels[0]).issubset({-1, 1}) or int((labels[0] == 1).sum()) != 104:
        raise ValueError("Unexpected source labels")
    if np.isinf(x.to_numpy()).any():
        raise ValueError("Infinite source values are not allowed")
    x.columns = [f"Measurement {i:03d}" for i in range(590)]
    meta = pd.DataFrame({"record_id": [f"SECOM-{i:04d}" for i in range(len(x))],
                         "timestamp": pd.to_datetime(labels[1],
                                                      format="%d/%m/%Y %H:%M:%S", errors="raise"),
                         "failure": (labels[0] == 1).astype(int)})
    return x, meta


def partition_data(x, meta):
    """Freeze chronological boundaries; quarantine conflicts and cross-partition duplicates."""
    meta = meta.copy()
    hashes = pd.util.hash_pandas_object(x, index=False).astype(str)
    meta["measurement_hash"] = hashes
    conflicts = meta.groupby("measurement_hash").failure.nunique()
    bad = set(conflicts[conflicts > 1].index)
    order = meta.sort_values(["timestamp", "record_id"]).index
    ordered_times = meta.loc[order, "timestamp"]
    n = len(meta)
    cut1, cut2 = ordered_times.iloc[int(n * .6)], ordered_times.iloc[int(n * .8)]
    meta["partition"] = np.where(meta.timestamp < cut1, "train",
                                 np.where(meta.timestamp < cut2, "validation", "test"))
    meta["exclusion"] = ""
    meta.loc[meta.measurement_hash.isin(bad), "exclusion"] = "conflicting duplicate labels"
    seen = {}
    for idx in order:
        row = meta.loc[idx]
        if row.exclusion:
            continue
        previous = seen.get(row.measurement_hash)
        if previous and previous != row.partition:
            meta.loc[idx, "exclusion"] = "exact duplicate of earlier partition"
        else:
            seen[row.measurement_hash] = row.partition
    meta["eligible"] = meta.exclusion == ""
    return meta


def audit(x, meta):
    per_feature = pd.DataFrame({"measurement": x.columns, "missing_fraction": x.isna().mean().values,
                                "unique_nonmissing": x.nunique().values,
                                "meaning": "Unknown", "units": "Unknown"})
    per_feature.to_csv(ARTIFACTS / "data_dictionary.csv", index=False)
    rows = []
    distributions = []
    for name in ("train", "validation", "test"):
        mask = (meta.partition == name) & meta.eligible
        subset = x.loc[mask]
        rows.append({"partition": name, "records": int(mask.sum()), "failures": int(meta.loc[mask, "failure"].sum()),
                     "missing_fraction": float(subset.isna().to_numpy().mean()),
                     "start": str(meta.loc[mask, "timestamp"].min()), "end": str(meta.loc[mask, "timestamp"].max())})
        for col in x:
            distributions.append({"partition": name, "measurement": col,
                                  "missing_fraction": float(subset[col].isna().mean()),
                                  "median": None if subset[col].dropna().empty else float(subset[col].median())})
    pd.DataFrame(distributions).to_csv(ARTIFACTS / "partition_distributions.csv", index=False)
    daily = meta.groupby(meta.timestamp.dt.date).failure.agg(["size", "sum"]).reset_index()
    daily.columns = ["date", "records", "failures"]
    daily.to_csv(ARTIFACTS / "daily_quality.csv", index=False)
    result = {"raw_records": len(x), "raw_measurements": x.shape[1], "raw_failures": int(meta.failure.sum()),
              "missing_fraction": float(x.isna().to_numpy().mean()),
              "all_null_columns": int(x.isna().all().sum()), "constant_columns": int((x.nunique() <= 1).sum()),
              "duplicate_rows": int(x.duplicated().sum()), "partitions": rows,
              "exclusions": meta.exclusion[~meta.eligible].value_counts().to_dict()}
    save_json(ARTIFACTS / "audit.json", result)
    meta.to_csv(ARTIFACTS / "split_manifest.csv", index=False)
    return result
