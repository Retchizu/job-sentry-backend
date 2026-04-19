#!/usr/bin/env python3
"""CLI: merge fake + job row-level CSVs (TICKET-007).

Exports that include ``user_risk_class`` (Supabase ``job_postings``) are merged through
``datasets_row_merge``; ``risk_class`` prefers valid ``user_risk_class`` when present (FE-TICKET-004).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Repo root = parent of scripts/
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import datasets_row_merge as drm  # noqa: E402


def _default_artifacts(name: str) -> Path:
    return _ROOT / "artifacts" / "datasets" / name


def main() -> None:
    p = argparse.ArgumentParser(
        description="Combine fake_job_postings_rows.csv and job_postings_rows.csv (TICKET-007)."
    )
    p.add_argument(
        "--fake",
        type=Path,
        default=_default_artifacts("fake_job_postings_rows.csv"),
        help="Path to fake_job_postings_rows.csv",
    )
    p.add_argument(
        "--job",
        type=Path,
        default=_default_artifacts("job_postings_rows.csv"),
        help="Path to job_postings_rows.csv",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=_default_artifacts("combined_job_postings_rows.csv"),
        help="Output combined CSV path",
    )
    p.add_argument(
        "--out-summary",
        type=Path,
        default=_default_artifacts("combined_job_postings_rows.summary.json"),
        help="Output JSON summary path",
    )
    args = p.parse_args()

    merged, summary = drm.merge_sources(args.fake, args.job)
    out_order = list(drm.REQUIRED_COLUMNS) + [
        "user_risk_class",
        "dataset_source",
        "warning_label",
        "risk_class",
    ]
    merged = merged[[c for c in out_order if c in merged.columns]]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)

    summary_out = dict(summary)
    summary_out["outputs"] = {
        "combined_csv": str(args.out_csv.resolve()),
        "summary_json": str(args.out_summary.resolve()),
    }
    with open(args.out_summary, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)

    print(json.dumps(summary_out, indent=2))


if __name__ == "__main__":
    main()
