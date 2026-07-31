"""Generate examples JSON for LightgenEmissionDataset.

Usage:
    python make_examples_json.py --fixture_root /path/to/fixture --out /path/to/output.json
"""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Generate examples JSON for LightgenEmissionDataset")
    parser.add_argument("--fixture_root", type=str, required=True, help="Root directory containing shape dirs")
    parser.add_argument("--out", type=str, required=True, help="Output JSON file path")
    args = parser.parse_args()

    # List all subdirectories and sort them (exclude those starting with _)
    try:
        dirs = sorted([
            os.path.join(args.fixture_root, d)
            for d in os.listdir(args.fixture_root)
            if os.path.isdir(os.path.join(args.fixture_root, d)) and not d.startswith('_')
        ])
    except Exception as e:
        print(f"Error reading fixture_root: {e}", file=sys.stderr)
        sys.exit(1)

    if not dirs:
        print(f"Warning: No subdirectories found in {args.fixture_root}", file=sys.stderr)

    # Write JSON list of absolute paths
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dirs, f, indent=2)

    print(f"Wrote {len(dirs)} example paths to {args.out}")


if __name__ == "__main__":
    main()
