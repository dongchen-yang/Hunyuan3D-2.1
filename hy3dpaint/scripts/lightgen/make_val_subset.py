"""Deterministically pick a fixed-size validation subset out of the v2-campaign val split.

Why a subset at all: the v2 val split is 387 shapes, and `validation_step` runs a full
30-step sampling loop per shape. At 2500-step validation intervals over a 50k-step run that
is 20 validation passes; 387 shapes each would cost more wall time than it buys us. 64 shapes
is enough for a stable `val/emission_mse` and for an eyeballable val grid.

Why hash-sorted rather than `random.sample` or "the first 64": the shas in the split file are
already lexicographically sorted, so "the first 64" would be a biased slice of sha space (and
the shas are content-ish ids, so that slice is arbitrary but *fixed* in a way that correlates
with nothing we control). Sorting by sha256(sha) gives an unbiased, seed-free, reproducible
shuffle -- rerunning this script on any machine, in any Python version, yields the same 64.

Outputs:
  * a sha list (one per line) -- committed, it pins which shapes we validate on;
  * optionally an examples JSON (list of absolute shape dirs under a fixture root) in the
    format `LightgenEmissionDataset` expects, same as scripts/lightgen/make_examples_json.py.
    The fixture root is machine-specific (e.g. venus's /localscratch/... staging dir), so the
    JSON is generated per machine while the sha list stays the single source of truth.

Usage:
    # sha list only (fixture not staged yet):
    python make_val_subset.py \
        --val_shas ../../data_processing/multiview_render_pipeline/v2_val_shas.txt \
        --n 64 --out_shas scripts/lightgen/val64_v2_shas.txt

    # sha list + examples json for a staged fixture:
    python make_val_subset.py --val_shas <...> --out_shas <...> \
        --fixture_root /localscratch/dya78/lightgen_mvpaint/mv74k \
        --out_json scripts/lightgen/mv74k_val64_venus.json
"""

import argparse
import hashlib
import json
import os
import sys


def hash_sorted_subset(shas, n):
    """Return the first `n` shas under an ordering by sha256 of the sha string.

    Deterministic across machines and Python versions (unlike hash()), and independent of the
    input file's own ordering, since the key depends only on the sha's bytes.
    """
    ordered = sorted(shas, key=lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest())
    return ordered[:n]


def read_shas(path):
    with open(path) as f:
        shas = [line.strip() for line in f]
    shas = [s for s in shas if s]
    if len(shas) != len(set(shas)):
        raise SystemExit(f"{path}: contains duplicate shas")
    return shas


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--val_shas", required=True, help="path to the full val sha list (one sha per line)")
    parser.add_argument("--n", type=int, default=64, help="subset size (default 64)")
    parser.add_argument("--out_shas", required=True, help="where to write the subset sha list")
    parser.add_argument("--fixture_root", default=None, help="if given, also write an examples JSON rooted here")
    parser.add_argument("--out_json", default=None, help="path for the examples JSON (requires --fixture_root)")
    parser.add_argument(
        "--check_exists",
        action="store_true",
        help="fail if any selected shape dir is missing under --fixture_root (use when the fixture is staged locally)",
    )
    args = parser.parse_args()

    if bool(args.fixture_root) != bool(args.out_json):
        raise SystemExit("--fixture_root and --out_json must be given together")

    shas = read_shas(args.val_shas)
    if args.n > len(shas):
        raise SystemExit(f"--n {args.n} exceeds the {len(shas)} shas in {args.val_shas}")
    subset = hash_sorted_subset(shas, args.n)

    out_dir = os.path.dirname(args.out_shas)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_shas, "w") as f:
        f.write("".join(s + "\n" for s in subset))
    print(f"Wrote {len(subset)} of {len(shas)} shas to {args.out_shas}")

    if args.fixture_root:
        # Not os.path.abspath()'d: the root is frequently a *remote* absolute path (venus's
        # /localscratch/...) that does not exist on the machine generating the JSON.
        root = args.fixture_root.rstrip("/")
        dirs = [os.path.join(root, s) for s in subset]
        missing = [d for d in dirs if not os.path.isdir(d)]
        if missing:
            msg = f"{len(missing)}/{len(dirs)} shape dirs not present under {root} (first: {missing[0]})"
            if args.check_exists:
                raise SystemExit("ERROR: " + msg)
            print("NOTE: " + msg, file=sys.stderr)
        json_dir = os.path.dirname(args.out_json)
        if json_dir:
            os.makedirs(json_dir, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(dirs, f, indent=2)
        print(f"Wrote {len(dirs)} example paths to {args.out_json}")


if __name__ == "__main__":
    main()
