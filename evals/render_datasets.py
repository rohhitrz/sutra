"""Render the line-delimited datasets as indented JSON, for reading.

# CONCEPT: one canonical format, one derived view. The `.jsonl` files are the ground truth the
# eval runner streams, and JSON Lines is one object per line by definition — indenting them would
# stop them being JSONL. So the readable copies under `datasets/readable/` are *generated*, never
# edited, and `make validate` fails if they have drifted from the source. A second hand-maintained
# copy of your ground truth is a bug waiting to happen; a generated one is a view.

Run with `make datasets`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "evals" / "datasets"
READABLE_DIR = DATASET_DIR / "readable"

DATASETS = ("golden_qa", "attacks")

HEADER = (
    "GENERATED FILE — do not edit. Produced by `make datasets` from {source}, which is the "
    "canonical copy the eval runner reads. Edit the .jsonl and re-render."
)


def render(name: str) -> str:
    """Pretty-print one dataset: a JSON object with a header note and the rows indented."""
    rows = [
        json.loads(line)
        for line in (DATASET_DIR / f"{name}.jsonl").read_text().splitlines()
        if line.strip()
    ]
    document = {"_note": HEADER.format(source=f"evals/datasets/{name}.jsonl"), "rows": rows}
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def main(check: bool = False) -> int:
    stale: list[str] = []
    for name in DATASETS:
        target = READABLE_DIR / f"{name}.json"
        rendered = render(name)
        if check:
            current = target.read_text() if target.exists() else ""
            if current != rendered:
                stale.append(target.name)
            continue
        target.write_text(rendered)
        print(f"wrote {target.relative_to(REPO_ROOT)}")

    if check and stale:
        print(f"FAIL  readable mirrors are stale: {', '.join(stale)} — run `make datasets`")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(check="--check" in sys.argv))
