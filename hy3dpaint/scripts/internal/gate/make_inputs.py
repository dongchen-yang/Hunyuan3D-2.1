"""Build the gate's fixed inputs (REFACTOR_TODO.md step 2): one directory per published run,

  <out>/<run>/shapes/<sha>/render_tex/...   the predictor's example dirs (36 view PNGs, transforms.json)
  <out>/<run>/shapes/<sha>/render_cond/000.png
  <out>/<run>/thumbs/<sha>.png              the reference image (the run's ref_source is "thumbnail")
  <out>/<run>/shas.txt                      every shape, in the order the published inference ran them
  <out>/<run>/subset20.txt                  the routine-step subset (every 10th shape, then the first 20)
  <out>/<run>/MANIFEST.sha256               every file above, checked by the gate hosts after staging

  paper:   the 204 test shapes of the paper run, from the paper-era fixture on jupiter
           (evaluation/mvpaint_74k_monitor/mv74k_test_agentic_clean204.json), thumbnails from
           trellis2_bw/eval_inputs/test388/thumbnails
  release: the 200 test shapes of the release run, from the workstation copy its scored inference
           read (/local-scratch/localhome/dya78/lightgen_release_test200/{release,thumbs})
  train:   the training check's root: the first 18 shapes of the release validation copy
           (/local-scratch/localhome/dya78/lightgen_release_val200/{release_train.json,thumbs}), 16 in
           train.txt and 2 in val.txt (shas.txt and subset20.txt are not written)

Files are copied with their links resolved, so a later change on jupiter cannot change a gate input.

  python make_inputs.py --out /local-scratch/localhome/dya78/lightgen_eval_local/hunyuan_refactor_checks/inputs
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

REPO = "/local-scratch/localhome/dya78/code/lightgen"
JUP = "/cs/3dlg-jupiter-project/lightgen"
REL = "/local-scratch/localhome/dya78/lightgen_release_test200"
SOURCES = {
    "paper": {"examples": f"{REPO}/evaluation/mvpaint_74k_monitor/mv74k_test_agentic_clean204.json",
              "thumbs": f"{JUP}/trellis2_bw/eval_inputs/test388/thumbnails", "n": 204},
    "release": {"examples": f"{REL}/test200_examples.json", "thumbs": f"{REL}/thumbs", "n": 200},
    "train": {"examples": "/local-scratch/localhome/dya78/lightgen_release_val200/release_train.json",
              "thumbs": "/local-scratch/localhome/dya78/lightgen_release_val200/thumbs", "n": 200, "take": 18},
}
VIEWS = [f"{v:03d}_{m}.png" for v in range(6) for m in ("albedo", "mr", "alpha", "normal", "pos", "emission")]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build(run, out):
    src = SOURCES[run]
    dirs = json.load(open(src["examples"]))
    if len(dirs) != src["n"]:
        sys.exit(f"{run}: {src['examples']} lists {len(dirs)} shapes, expected {src['n']}")
    dirs = dirs[:src.get("take", len(dirs))]
    root = os.path.join(out, run)
    if os.path.exists(root):
        sys.exit(f"{root} exists; inputs are built once")
    tmp = root + ".part"
    shutil.rmtree(tmp, ignore_errors=True)
    shas = []
    for d in dirs:
        sha = os.path.basename(d.rstrip("/"))
        need = [os.path.join("render_tex", v) for v in VIEWS] + ["render_tex/transforms.json", "render_cond/000.png"]
        for rel in need:
            s = os.path.join(d, rel)
            if not os.path.isfile(s):
                sys.exit(f"{run}: {s} is missing")
            t = os.path.join(tmp, "shapes", sha, rel)
            os.makedirs(os.path.dirname(t), exist_ok=True)
            shutil.copyfile(s, t)          # follows links: the file's bytes, not the link
        th = os.path.join(src["thumbs"], f"{sha}.png")
        if not os.path.isfile(th):
            sys.exit(f"{run}: thumbnail {th} is missing")
        os.makedirs(os.path.join(tmp, "thumbs"), exist_ok=True)
        shutil.copyfile(th, os.path.join(tmp, "thumbs", f"{sha}.png"))
        shas.append(sha)
    lists = ({"train.txt": shas[:16], "val.txt": shas[16:18]} if run == "train"
             else {"shas.txt": shas, "subset20.txt": shas[::10][:20]})
    for name, l in lists.items():
        with open(os.path.join(tmp, name), "w") as f:
            f.write("\n".join(l) + "\n")
    lines = []
    for dp, _, fns in os.walk(tmp):
        for fn in sorted(fns):
            p = os.path.join(dp, fn)
            lines.append(f"{sha256(p)}  {os.path.relpath(p, tmp)}")
    with open(os.path.join(tmp, "MANIFEST.sha256"), "w") as f:
        f.write("\n".join(sorted(lines, key=lambda l: l.split("  ", 1)[1])) + "\n")
    os.replace(tmp, root)
    print(f"[inputs] {run}: {len(shas)} shapes, {len(lines)} files -> {root}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", nargs="+", default=list(SOURCES))
    a = ap.parse_args()
    for run in a.runs:
        build(run, a.out)


if __name__ == "__main__":
    main()
