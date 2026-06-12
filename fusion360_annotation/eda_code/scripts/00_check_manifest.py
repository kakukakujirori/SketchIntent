from __future__ import annotations

from collections import Counter

from _bootstrap import bootstrap

bootstrap()

from fusion360_annotation.eda_code.canonical import canonical_dir_for
from fusion360_annotation.eda_code.config import data_root
from fusion360_annotation.eda_code.manifest import discover_manifest


def main() -> None:
    manifest, stats = discover_manifest()
    n_files = sum(len(v) for v in manifest.values())
    print(f"DATA_ROOT = {data_root()}")
    print(f"batches  = {sorted(manifest, key=int)}")
    print(f"sketches with complete trio = {n_files}")
    for batch in sorted(manifest, key=int):
        inits = Counter()
        for files in manifest[batch].values():
            inits.update(files)
        print(
            f"  batch {batch:>2}: {len(manifest[batch]):3d} sketches, "
            f"annotators={'+'.join(sorted(inits))}"
        )
    print("exclusion stats:", dict(stats))
    missing = [
        f"{b}/{base}"
        for b in manifest
        for base in manifest[b]
        if canonical_dir_for(b, base) is None
    ]
    print(f"sketches missing canonical .py: {len(missing)}")
    for item in missing[:10]:
        print("  ", item)


if __name__ == "__main__":
    main()
