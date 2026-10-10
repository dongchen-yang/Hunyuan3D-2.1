"""Gate comparison for the predicted views: how far a candidate run is from a baseline, and whether
that distance is noise (REFACTOR_TODO.md, How we work). Adapted from TEXGen-Emission's
scripts/internal/gate/compare_numeric.py: the files are the six emission views per shape, and there is
no UV metric (the views carry no ground-truth mask; a real difference is sized by bridging and scoring
the shapes it touches).

  python compare_views.py --a <baseline> --b <candidate> --shas_file <shas>
      [--envelope <run> <run> ... [--envelope_shas <shas>] [--envelope_json <cache>]] [--csv <rows.csv>]
  python compare_views.py --a <any run> --shas_file <shas> --envelope <run> <run> ...   (print the envelope)

Both roots are laid out <root>/<sha>/seed00/00{0..5}_emission.png (the predictor's output). Per shape
and view: byte identity, the mean and max absolute pixel difference (1/255 steps) and the share of the
view's channel values that differ.

Tolerance first (Dongchen 2026-10-07, TEXGen's rule): a shape passes as noise when in each of its six
views every pixel differs from the baseline by at most 1/255 and at most 0.5% of the values differ
(512 x 512 x 3 values per view). Every other shape meets the envelope: per measure (each view's mean
and max difference), the largest per-shape difference between any two envelope runs (at least 4 runs
of tag pre-refactor on the same GPU type). --envelope_json caches it; the cache is recomputed when the
runs, the shapes or any file it reads (path, size, mtime) change.

Verdict (exit 0 for the first two):
  IDENTICAL     every view byte-identical, nothing missing
  WITHIN_NOISE  nothing missing, every shape within the tolerance or, if over it, within the envelope
  DIFFERENT     something is missing, or a shape is over the tolerance and no envelope was given
  OUTSIDE_NOISE a shape over the tolerance exceeds the envelope in a measure
  EMPTY         the shapes file lists no shape
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import sys

import numpy as np
from PIL import Image

FILES = tuple(f"seed00/{v:03d}_emission.png" for v in range(6))
KEYS = tuple(f"v{v}" for v in range(6))
TOLERANCE = 1                   # 1/255 steps
TOLERANCE_MAX_FRACTION = 0.005  # share of a view's channel values
MEASURES = [f"{k}_{stat}" for k in KEYS for stat in ("mean", "max")]
ENVELOPE_SCHEMA = 1


def load(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(np.int16)


def compare_pair(root_a, root_b, shas):
    """Per-shape rows {sha, v<k>_state, v<k>_mean, v<k>_max, v<k>_frac}."""
    rows = []
    for sha in shas:
        row = {"sha": sha}
        for fn, k in zip(FILES, KEYS):
            pa, pb = os.path.join(root_a, sha, fn), os.path.join(root_b, sha, fn)
            if not (os.path.isfile(pa) and os.path.isfile(pb)):
                row[f"{k}_state"] = "missing"
                continue
            with open(pa, "rb") as fa, open(pb, "rb") as fb:
                same_bytes = fa.read() == fb.read()
            a, b = load(pa), load(pb)
            d = np.abs(a - b) if a.shape == b.shape else None
            row[f"{k}_state"] = "identical" if same_bytes else ("shape" if d is None else "different")
            row[f"{k}_mean"] = float(d.mean()) if d is not None else math.inf
            row[f"{k}_max"] = float(d.max()) if d is not None else math.inf
            row[f"{k}_frac"] = float(np.count_nonzero(d) / d.size) if d is not None else math.inf
        rows.append(row)
    return rows


def summarize(rows):
    s, counts = {}, {}
    for k in KEYS:
        states = [r.get(f"{k}_state", "missing") for r in rows]
        counts[k] = {st: states.count(st) for st in ("identical", "different", "shape", "missing")}
        for stat in ("mean", "max", "frac"):
            v = [r[f"{k}_{stat}"] for r in rows if f"{k}_{stat}" in r]
            s[f"{k}_{stat}"] = (float(np.mean(v)), float(np.max(v))) if v else (0.0, 0.0)
    return s, counts


def report(tag, s, counts, n):
    same = sum(counts[k]["identical"] for k in KEYS)
    diff = sum(counts[k]["different"] + counts[k]["shape"] for k in KEYS)
    miss = sum(counts[k]["missing"] for k in KEYS)
    print(f"[{tag}] views: same={same} different={diff} missing={miss} (of {6 * n})")
    for k, fn in zip(KEYS, FILES):
        print(f"[{tag}] {fn}: |d| per shape (1/255 steps) mean {s[k + '_mean'][0]:.4g} (worst shape "
              f"{s[k + '_mean'][1]:.4g}), max {s[k + '_max'][1]:.4g}; values differing: worst shape "
              f"{100 * s[k + '_frac'][1]:.3g}%")


def within_tolerance(row):
    return all(row.get(f"{k}_max", math.inf) <= TOLERANCE and row.get(f"{k}_frac", math.inf) <= TOLERANCE_MAX_FRACTION
               for k in KEYS)


def files_digest(runs, shas):
    h = hashlib.sha256()
    for r in runs:
        for sha in shas:
            for fn in FILES:
                p = os.path.join(os.path.abspath(r), sha, fn)
                try:
                    st = os.stat(p)
                    h.update(f"{p} {st.st_size} {st.st_mtime_ns}\n".encode())
                except FileNotFoundError:
                    h.update(f"{p} missing\n".encode())
    return h.hexdigest()


def envelope(runs, shas, cache=None):
    key = {"schema": ENVELOPE_SCHEMA, "measure_names": MEASURES, "runs": [os.path.abspath(r) for r in runs],
           "n": len(shas), "shas_sha256": hashlib.sha256("\n".join(shas).encode()).hexdigest(),
           "files_sha256": files_digest(runs, shas)}
    if cache and os.path.isfile(cache):
        with open(cache) as f:
            e = json.load(f)
        if all(e.get(k) == v for k, v in key.items()):
            return e
    pairs = {}
    for i, j in itertools.combinations(range(len(runs)), 2):
        s, counts = summarize(compare_pair(runs[i], runs[j], shas))
        if sum(counts[k]["missing"] for k in KEYS):
            sys.exit(f"envelope runs {runs[i]} and {runs[j]} are missing views for these shapes")
        pairs[f"{i}-{j}"] = {"identical": all(counts[k]["identical"] == len(shas) for k in KEYS),
                             "measures": {m: s[m][1] for m in MEASURES}}
    env = {m: max(p["measures"][m] for p in pairs.values()) for m in MEASURES}
    if any(not math.isfinite(v) for v in env.values()):
        sys.exit("an envelope pair has a NaN or infinity in a measure; it is not a noise envelope")
    e = {**key, "measures": env, "pairs": pairs}
    if cache:
        os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
        with open(cache + ".part", "w") as f:
            json.dump(e, f, indent=1)
        os.replace(cache + ".part", cache)
    return e


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--a", required=True, help="baseline root (tag pre-refactor)")
    ap.add_argument("--b", help="candidate root (omitted: only measure and print the envelope)")
    ap.add_argument("--shas_file", required=True)
    ap.add_argument("--envelope", nargs="+", help="two or more runs of pre-refactor on the same GPU type")
    ap.add_argument("--envelope_shas", help="shapes the envelope is measured over (default: --shas_file)")
    ap.add_argument("--envelope_json", help="cache file for the envelope")
    ap.add_argument("--csv", help="write the candidate's per-shape rows here")
    a = ap.parse_args(argv)
    runs = a.envelope
    if runs is not None and len(runs) < 2:
        sys.exit("--envelope needs two or more runs")
    shas = [l.strip() for l in open(a.shas_file) if l.strip()]
    if not shas:
        print(f"[views] n=0 verdict=EMPTY ({a.shas_file} lists no shape)")
        return 1
    if a.b is None:
        if runs is None:
            sys.exit("without --b, give --envelope")
        e = envelope(runs, shas, a.envelope_json)
        print(f"[envelope] {len(runs)} runs, {len(e['pairs'])} pairs ({sum(p['identical'] for p in e['pairs'].values())} "
              f"identical), {len(shas)} shapes: " + ", ".join(f"{m} {e['measures'][m]:.3g}" for m in MEASURES))
        return 0
    rows = compare_pair(a.a, a.b, shas)
    s, counts = summarize(rows)
    report("candidate", s, counts, len(shas))
    if a.csv:
        keys = sorted({k for r in rows for k in r} - {"sha"})
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["sha"] + keys)
            w.writeheader()
            w.writerows(rows)
    missing = sum(counts[k]["missing"] for k in KEYS)
    identical = all(counts[k]["identical"] == len(shas) for k in KEYS)
    big = [r for r in rows if not within_tolerance(r)]
    print(f"[tolerance] {len(rows) - len(big)} of {len(rows)} shapes within {TOLERANCE}/255 on at most "
          f"{100 * TOLERANCE_MAX_FRACTION:g}% of the values in all six views; {len(big)} differ by more")
    if identical:
        verdict = "IDENTICAL"
    elif missing:
        verdict = "DIFFERENT"
    elif not big:
        verdict = "WITHIN_NOISE"
    elif runs is None:
        verdict = "DIFFERENT"
    else:
        s, _ = summarize(big)
        eshas = [x.strip() for x in open(a.envelope_shas)] if a.envelope_shas else shas
        eshas = [x for x in eshas if x]
        if set(shas) - set(eshas):
            sys.exit("--shas_file lists shapes the envelope is not measured over")
        e = envelope(runs, eshas, a.envelope_json)
        env = e["measures"]
        print(f"[envelope] {len(runs)} runs, {len(e['pairs'])} pairs ({sum(p['identical'] for p in e['pairs'].values())} "
              f"identical), {len(eshas)} shapes")
        over = [m for m in MEASURES if not (s[m][1] <= env[m])]
        verdict = "WITHIN_NOISE" if not over else "OUTSIDE_NOISE (" + ", ".join(
            f"{m} {s[m][1]:.3g} > {env[m]:.3g}" for m in sorted(over)) + ")"
    print(f"[views] n={len(shas)} verdict={verdict}")
    return 0 if verdict in ("IDENTICAL", "WITHIN_NOISE") else 1


if __name__ == "__main__":
    sys.exit(main())
