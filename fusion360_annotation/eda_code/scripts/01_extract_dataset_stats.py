from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from fusion360_annotation.eda_code.config import data_root
from fusion360_annotation.eda_code.dataset_features import extract_features
from fusion360_annotation.eda_code.manifest import discover_groups
from fusion360_annotation.eda_code.progress import progress

HERE = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--limit-groups", type=int, default=None)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    complete, incomplete, _stats = discover_groups()
    print(
        f"Discovered {len(complete) + len(incomplete)} sketch ids: "
        f"{len(complete)} complete (3 annotators), {len(incomplete)} incomplete."
    )

    missing = [
        {
            "sketch_id": g.sketch_id,
            "folder": g.batch,
            "present": sorted(g.files.keys()),
        }
        for g in incomplete
    ]
    records = []
    failures = []
    selected = complete[: args.limit_groups] if args.limit_groups is not None else complete
    for group in progress(selected, desc="dataset stats", total=len(selected)):
        for initial in sorted(group.files):
            fp = group.file_path(initial)
            try:
                feat = extract_features(fp)
                feat.update(
                    {
                        "sketch_id": group.sketch_id,
                        "folder": group.batch,
                        "annotator": initial,
                        "annotator_name": initial,
                    }
                )
                records.append(feat)
            except Exception as exc:  # noqa: BLE001
                failures.append({"filepath": str(fp), "error": str(exc)})
                print(f"[fail] {fp}\n{traceback.format_exc()}")

    summary = {
        "root": str(data_root()),
        "n_sketch_ids": len(complete) + len(incomplete),
        "n_complete": len(complete),
        "n_incomplete": len(incomplete),
        "n_records": len(records),
        "n_failures": len(failures),
        "missing": missing,
        "failures": failures,
    }
    (out_dir / "records.json").write_text(json.dumps(records, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {len(records)} records ({summary['n_failures']} failures) to {out_dir}/")


if __name__ == "__main__":
    main()
