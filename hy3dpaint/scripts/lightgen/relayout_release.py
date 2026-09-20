"""Re-lay-out an unpacked LightgenBench release into the fixture layout the loader reads.

The released dataset (Hugging Face 3dlg-hcvc/LightgenBench) ships one directory per shape:

    <root>/<uuid>/multiview/00N_{albedo,mr,normal,pos,emission,alpha}.png + transforms.json
    <root>/<uuid>/thumbnail.png

`LightgenEmissionDataset` and `make_74k_jsons.py` were written against the render fixture:

    <root>/<sha>/render_tex/<the same 37 files>
    <root>/<sha>/render_cond/000.png
    <thumbnail_dir>/<sha>.png

The files are the same (the release hard-links `render_tex/` as `multiview/`), only the names
differ, so this renames and links in place instead of touching the loader:

    multiview/                -> render_tex/                  os.rename
    render_tex/001_albedo.png -> render_cond/000.png          hard link
    thumbnail.png             -> <thumbs_out>/<uuid>.png      hard link

`render_cond/000.png` is what it was in the fixture, a copy of `render_tex/001_albedo.png`.
A `ref_source: thumbnail` run never reads it; `make_74k_jsons.py`'s A7 gate requires it.

Run it on node-local storage inside the job, on the tree the tars were just unpacked into.
Never run it on jupiter's `dataset_release`: those files are hard links into `dataset_73k`.
It is idempotent, and it exits 1 if any listed shape is missing a piece.

Usage:
    python relayout_release.py --root $STAGE/release --thumbs_out $STAGE/thumbs \
        --shas train_shas.txt val_shas.txt
"""

import argparse
import os
import shutil
import sys


def _link(src, dst):
    """Hard link, or a byte copy where the filesystem refuses links. No-op if dst exists."""
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def relayout_one(root, thumbs_out, sha):
    """Return None on success, else a short reason string."""
    d = os.path.join(root, sha)
    mv, tex = os.path.join(d, "multiview"), os.path.join(d, "render_tex")
    if os.path.isdir(mv):
        if os.path.exists(tex):
            return f"{d}: both multiview/ and render_tex/ exist"
        os.rename(mv, tex)
    elif not os.path.isdir(tex):
        return f"{d}: no multiview/ (shape not unpacked)"
    front = os.path.join(tex, "001_albedo.png")
    if not os.path.isfile(front):
        return f"{front} missing"
    thumb = os.path.join(d, "thumbnail.png")
    if not os.path.isfile(thumb):
        return f"{thumb} missing"
    os.makedirs(os.path.join(d, "render_cond"), exist_ok=True)
    _link(front, os.path.join(d, "render_cond", "000.png"))
    _link(thumb, os.path.join(thumbs_out, sha + ".png"))
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="the unpacked release: <root>/<uuid>/...")
    ap.add_argument("--thumbs_out", required=True, help="directory that receives <uuid>.png")
    ap.add_argument("--shas", nargs="+", required=True, help="sha list files, one uuid per line")
    args = ap.parse_args()

    shas = []
    for path in args.shas:
        with open(path) as f:
            shas += [ln.strip() for ln in f if ln.strip()]
    os.makedirs(args.thumbs_out, exist_ok=True)

    bad = [why for why in (relayout_one(args.root, args.thumbs_out, s) for s in shas) if why]
    for why in bad[:10]:
        print(f"FAIL: {why}", file=sys.stderr, flush=True)
    if bad:
        print(f"FAIL: {len(bad)} of {len(shas)} listed shapes could not be re-laid-out", file=sys.stderr)
        return 1
    print(f"PASS: {len(shas)} shapes re-laid-out under {args.root}; thumbnails in {args.thumbs_out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
