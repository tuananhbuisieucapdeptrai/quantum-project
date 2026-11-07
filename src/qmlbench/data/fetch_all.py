"""CLI: download and preprocess the benchmark datasets.

    qmlbench-fetch                  # all four
    qmlbench-fetch --only tides wind
    qmlbench-fetch --refresh        # ignore cached raw files
"""

from __future__ import annotations

import argparse

from . import DATASETS, fetch_dataset, get_dataset


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=DATASETS, default=DATASETS)
    ap.add_argument("--refresh", action="store_true", help="re-download raw files")
    args = ap.parse_args(argv)
    for name in args.only:
        print(f"== {name}")
        path = fetch_dataset(name, refresh=args.refresh)
        print(f"   wrote {path}")
        print("   " + get_dataset(name).summary().replace("\n", "\n   "))


if __name__ == "__main__":
    main()
