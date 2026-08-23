"""pytest for make_74k_jsons.py's A7 thumbnail gate (--thumbnail_dir).

Runs the script as a subprocess on a synthetic staged root, exactly as the launcher invokes it.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

SCRIPT = os.path.join(os.path.dirname(__file__), "make_74k_jsons.py")
SUFFIXES = ("albedo", "emission", "mr", "alpha", "normal", "pos")


def _sha(k):
    return f"{k:032x}"  # bare 32-hex, as A0 requires


def _complete_dir(root, sha, size=8):
    d = os.path.join(root, sha)
    os.makedirs(os.path.join(d, "render_tex"))
    os.makedirs(os.path.join(d, "render_cond"))
    px = Image.fromarray(np.zeros((size, size, 3), np.uint8))
    for v in range(6):
        for s in SUFFIXES:
            px.save(os.path.join(d, "render_tex", f"{v:03d}_{s}.png"))
    with open(os.path.join(d, "render_tex", "transforms.json"), "w") as f:
        json.dump({"frames": []}, f)
    px.save(os.path.join(d, "render_cond", "000.png"))
    return d


def _write(path, shas):
    with open(path, "w") as f:
        f.write("\n".join(shas) + "\n")
    return path


def _setup(tmp):
    """5 train + 2 val + 1 test; val64 = the 2 val; all 8 dirs complete; a thumbnail for all 8."""
    train, val, test = [_sha(k) for k in range(5)], [_sha(5), _sha(6)], [_sha(7)]
    root = os.path.join(tmp, "mv74k")
    os.makedirs(root)
    for s in train + val + test:
        _complete_dir(root, s)
    thumbs = os.path.join(tmp, "thumbs")
    os.makedirs(thumbs)
    for s in train + val + test:
        Image.fromarray(np.zeros((6, 10, 3), np.uint8)).save(os.path.join(thumbs, s + ".png"))
    lists = dict(train=_write(os.path.join(tmp, "train.txt"), train),
                 val=_write(os.path.join(tmp, "val.txt"), val),
                 test=_write(os.path.join(tmp, "test.txt"), test),
                 val64=_write(os.path.join(tmp, "val64.txt"), val))
    return root, thumbs, lists


def _run(tmp, root, lists, extra=()):
    cmd = [sys.executable, SCRIPT,
           "--train_shas", lists["train"], "--val_shas", lists["val"],
           "--test_shas", lists["test"], "--val64_shas", lists["val64"],
           "--fixture_root", root,
           "--out_train", os.path.join(tmp, "train.json"),
           "--out_val64", os.path.join(tmp, "val64.json"),
           "--expect_train", "5", *extra]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_without_flag_behaviour_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        root, _thumbs, lists = _setup(tmp)
        r = _run(tmp, root, lists)
        assert r.returncode == 0, r.stderr
        assert "ALL_ASSERTIONS_PASS" in r.stdout
        assert len(json.load(open(os.path.join(tmp, "train.json")))) == 5


def test_thumbnail_gate_passes_when_every_listed_sha_has_one():
    with tempfile.TemporaryDirectory() as tmp:
        root, thumbs, lists = _setup(tmp)
        r = _run(tmp, root, lists, ["--thumbnail_dir", thumbs])
        assert r.returncode == 0, r.stderr
        assert "ALL_ASSERTIONS_PASS" in r.stdout
        assert "thumbnail" in r.stdout, "the A7 ok line must say the thumbnails were checked"


def test_thumbnail_gate_fails_on_a_missing_thumbnail():
    with tempfile.TemporaryDirectory() as tmp:
        root, thumbs, lists = _setup(tmp)
        os.remove(os.path.join(thumbs, _sha(2) + ".png"))  # a TRAIN sha
        r = _run(tmp, root, lists, ["--thumbnail_dir", thumbs])
        assert r.returncode == 1
        assert "A7" in r.stderr and _sha(2) in r.stderr
        assert not os.path.exists(os.path.join(tmp, "train.json")), "must not write on failure"


def test_thumbnail_gate_fails_on_a_zero_byte_thumbnail():
    with tempfile.TemporaryDirectory() as tmp:
        root, thumbs, lists = _setup(tmp)
        open(os.path.join(thumbs, _sha(5) + ".png"), "w").close()  # a val64 sha
        r = _run(tmp, root, lists, ["--thumbnail_dir", thumbs])
        assert r.returncode == 1
        assert "zero bytes" in r.stderr


def test_thumbnail_dir_must_be_a_directory():
    with tempfile.TemporaryDirectory() as tmp:
        root, _thumbs, lists = _setup(tmp)
        r = _run(tmp, root, lists, ["--thumbnail_dir", os.path.join(tmp, "nope")])
        assert r.returncode == 1
        assert "thumbnail_dir" in r.stderr
