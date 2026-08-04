"""Build the venus-side train / val64 example JSONs for the 74k pbr->emission run -- and prove,
before a 57-hour job is submitted, that the training set contains nothing we will evaluate on.

Why this exists instead of `make_examples_json.py --fixture_root <staged root>`: that script
enumerates the directory. The staged root holds *all* of train + val + test, so a
directory-enumeration train JSON would train on 100% of the evaluation set and every number the
run produced afterwards would be worthless. This script is sha-driven -- the committed v2 split
lists are the only source of membership -- and it refuses to write anything unless the split
algebra checks out.

Assertions (all hard; any failure aborts before a file is written):

  A0  every entry of every input list is a bare 32-hex-char sha. Without this, a list entry that
      is an absolute path or contains ".." would join to a directory *outside* the staged root --
      it would compare unequal to every val/test sha and so sail through A1/A5/A6 while pointing
      the trainer straight at an eval shape.
  A1  each input sha list is duplicate-free, and train/val/test are mutually disjoint
  A2  val64 is a subset of the full val split
  A3  the train shas missing from the staged root are *exactly* the expected-missing set
      (--allow_missing), so a half-finished rsync cannot be mistaken for the one known
      unrenderable shape
  A4  len(train json) == --expect_train, AND --expect_train == len(train list) - len(allow_missing).
      The second half is what stops the two knobs being fudged together: without it, "everything
      that is missing is allowed, and I expected exactly what survived" is self-consistent for any
      amount of missing data.
  A5  train json is disjoint from val64
  A6  train json is disjoint from the full val split and the full test split
  A7  every directory named in either JSON exists under the root and holds the full 31-PNG +
      transforms.json payload, each as a regular file of nonzero size

A5 and A6 are implied by A0+A1+A2 as the code stands (train_kept is a subset of the train list).
They are kept because they check the *constructed output* rather than the inputs, so they stay
load-bearing if the way train_kept is built ever changes. A7 certifies structure, not pixels --
the decode sweep that verifies every PNG actually decodes is a separate step, run against the
staged root before this script (see the phase-2 task-4 report).

Run it on the node that will train, so A3/A7 look at the real staged filesystem:

    python scripts/lightgen/make_74k_jsons.py \
        --train_shas  /localscratch/dya78/lightgen_mvpaint/v2_shas/v2_train_shas.txt \
        --val_shas    /localscratch/dya78/lightgen_mvpaint/v2_shas/v2_val_shas.txt \
        --test_shas   /localscratch/dya78/lightgen_mvpaint/v2_shas/v2_test_shas.txt \
        --val64_shas  scripts/lightgen/val64_v2_shas.txt \
        --fixture_root /localscratch/dya78/lightgen_mvpaint/mv74k \
        --out_train   /localscratch/dya78/lightgen_mvpaint/mv74k_train.json \
        --out_val64   /localscratch/dya78/lightgen_mvpaint/mv74k_val64.json \
        --expect_train 71645 \
        --allow_missing 810a644ad9a04b02b10cdf4972d93e70
"""

import argparse
import hashlib
import json
import os
import re
import sys

SHA_RE = re.compile(r"^[0-9a-f]{32}$")
VIEWS = range(6)
SUFFIXES = ("albedo", "emission", "mr", "normal", "pos")
# What one complete shape dir must contain: 30 render_tex PNGs + transforms.json, and render_cond.
EXPECT_TEX = {f"{v:03d}_{s}.png" for v in VIEWS for s in SUFFIXES} | {"transforms.json"}
EXPECT_COND = {"000.png"}


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def ok(msg):
    print(f"PASS: {msg}", flush=True)


def read_shas(path):
    with open(path) as f:
        shas = [ln.strip() for ln in f if ln.strip()]
    if len(shas) != len(set(shas)):
        seen, dupes = set(), []
        for s in shas:
            if s in seen and s not in dupes:
                dupes.append(s)
            seen.add(s)
        fail(f"{path}: duplicate shas (e.g. {dupes[:5]})")
    bad = [s for s in shas if not SHA_RE.match(s)]
    if bad:
        # A0. An entry like "/cs/.../val_sha" or "../val_sha" would os.path.join out of the
        # staged root while still comparing unequal to every val/test sha.
        fail(f"A0: {path}: {len(bad)} entr(y|ies) are not bare 32-hex shas, e.g. {bad[:3]}")
    return shas


def _regular_nonempty(dirpath, expected):
    """Return a reason string if `dirpath` does not hold exactly-these regular, nonempty files."""
    try:
        entries = {e.name: e for e in os.scandir(dirpath)}
    except OSError as e:
        return f"{dirpath} unreadable: {e}"
    missing = expected - set(entries)
    if missing:
        return f"{dirpath} missing {len(missing)} entr(y|ies), e.g. {sorted(missing)[:3]}"
    for name in expected:
        e = entries[name]
        # is_file() alone would accept a directory named 000_albedo.png via a symlink chain;
        # follow_symlinks=True is the default and is what the loader will do when it opens them.
        if not e.is_file():
            return f"{os.path.join(dirpath, name)} is not a regular file"
        if e.stat().st_size == 0:
            return f"{os.path.join(dirpath, name)} is zero bytes"
    return None


def dir_payload_ok(d):
    """Return None if the shape dir is complete, else a short reason string.

    Structural only: names, file-ness and nonzero size. Whether the bytes actually decode is
    established by the separate PNG decode sweep -- doing it here would make the launch gate a
    2.2-million-file decode.
    """
    return (_regular_nonempty(os.path.join(d, "render_tex"), EXPECT_TEX)
            or _regular_nonempty(os.path.join(d, "render_cond"), EXPECT_COND))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train_shas", required=True)
    ap.add_argument("--val_shas", required=True)
    ap.add_argument("--test_shas", required=True)
    ap.add_argument("--val64_shas", required=True)
    ap.add_argument("--fixture_root", required=True)
    ap.add_argument("--out_train", required=True)
    ap.add_argument("--out_val64", required=True)
    ap.add_argument("--expect_train", type=int, required=True)
    ap.add_argument("--allow_missing", nargs="*", default=[],
                    help="train shas known to be unrenderable; the missing set must equal this exactly")
    ap.add_argument("--max_allow_missing", type=int, default=1,
                    help="ceiling on --allow_missing (default 1). Raising it is the explicit, "
                         "logged act of deciding that more than one shape may be dropped.")
    args = ap.parse_args()

    root = args.fixture_root.rstrip("/")
    if not os.path.isdir(root):
        fail(f"--fixture_root {root} is not a directory")

    # ---- A0: every list entry is a bare sha (read_shas aborts otherwise) ------------------
    train = read_shas(args.train_shas)
    val = read_shas(args.val_shas)
    test = read_shas(args.test_shas)
    val64 = read_shas(args.val64_shas)
    ok(f"A0 all {len(train) + len(val) + len(test) + len(val64)} list entries are bare 32-hex shas "
       f"(no absolute path or '..' can escape {root})")
    train_s, val_s, test_s, val64_s = set(train), set(val), set(test), set(val64)

    # ---- A1: the pins themselves are sane -------------------------------------------------
    for a, b, na, nb in ((train_s, val_s, "train", "val"), (train_s, test_s, "train", "test"),
                         (val_s, test_s, "val", "test")):
        if a & b:
            fail(f"A1: input split lists overlap: {na} n {nb} = {len(a & b)} shas "
                 f"(e.g. {sorted(a & b)[:3]})")
    ok(f"A1 input lists duplicate-free and mutually disjoint "
       f"(train {len(train)}, val {len(val)}, test {len(test)})")

    # ---- A2: val64 comes out of val ------------------------------------------------------
    if not val64_s <= val_s:
        fail(f"A2: {len(val64_s - val_s)} of the {len(val64)} val64 shas are not in the val split")
    ok(f"A2 val64 ({len(val64)}) is a subset of the val split ({len(val)})")

    # ---- A3: the only absent train shapes are the expected ones ---------------------------
    allow = set(args.allow_missing)
    if len(allow) > args.max_allow_missing:
        fail(f"A3: --allow_missing has {len(allow)} shas, ceiling is {args.max_allow_missing}. "
             f"A half-staged fixture must not be absorbable by listing everything that is absent; "
             f"raise --max_allow_missing deliberately if the data really did change.")
    if not allow <= train_s:
        fail(f"A3: --allow_missing contains shas that are not in the train split: "
             f"{sorted(allow - train_s)}")
    missing = {s for s in train if not os.path.isdir(os.path.join(root, s))}
    if missing != allow:
        fail(f"A3: staged-root absences != expected. missing-but-not-allowed="
             f"{sorted(missing - allow)[:10]} (n={len(missing - allow)}); "
             f"allowed-but-present={sorted(allow - missing)} ")
    ok(f"A3 train shas absent from the staged root == exactly the expected set "
       f"{sorted(allow) if allow else '{}'} (n={len(missing)})")

    train_kept = [s for s in train if s not in missing]

    # ---- A4: the count is the count, and the two knobs agree with the list ----------------
    if len(train_kept) != args.expect_train:
        fail(f"A4: train json would have {len(train_kept)} entries, expected {args.expect_train}")
    implied = len(train) - len(allow)
    if args.expect_train != implied:
        fail(f"A4: --expect_train {args.expect_train} != len(train list) {len(train)} - "
             f"len(--allow_missing) {len(allow)} = {implied}; --expect_train and --allow_missing "
             f"must not be adjusted together to absorb missing data")
    ok(f"A4 train json entry count == {args.expect_train} == {len(train)} listed - {len(allow)} allowed-missing")

    # ---- A5/A6: no contamination ---------------------------------------------------------
    kept_s = set(train_kept)
    if kept_s & val64_s:
        fail(f"A5: train n val64 = {len(kept_s & val64_s)} shas")
    ok("A5 train n val64 == 0")
    contam = kept_s & (val_s | test_s)
    if contam:
        fail(f"A6: train n (val u test) = {len(contam)} shas (e.g. {sorted(contam)[:5]})")
    ok(f"A6 train n (val u test) == 0 (val {len(val)} + test {len(test)} checked)")

    # ---- A7: every listed dir is really there, and really complete ------------------------
    train_dirs = [os.path.join(root, s) for s in train_kept]
    val64_dirs = [os.path.join(root, s) for s in val64]
    bad = []
    for d in train_dirs + val64_dirs:
        if not os.path.isdir(d):
            bad.append((d, "not a directory"))
        else:
            why = dir_payload_ok(d)
            if why:
                bad.append((d, why))
    if bad:
        for d, why in bad[:10]:
            print(f"  {d}: {why}", file=sys.stderr)
        fail(f"A7: {len(bad)} of {len(train_dirs) + len(val64_dirs)} listed dirs are missing or incomplete")
    ok(f"A7 all {len(train_dirs) + len(val64_dirs)} listed dirs exist and hold 30 render_tex PNGs "
       f"+ transforms.json + render_cond/000.png, each a regular nonempty file")

    # ---- write ---------------------------------------------------------------------------
    for path, dirs in ((args.out_train, train_dirs), (args.out_val64, val64_dirs)):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w") as f:
            json.dump(dirs, f, indent=2)
        print(f"WROTE {path}  entries={len(dirs)}  sha256={sha256_file(path)}", flush=True)

    print("ALL_ASSERTIONS_PASS", flush=True)


if __name__ == "__main__":
    main()
