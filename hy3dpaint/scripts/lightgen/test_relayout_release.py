"""Tests for relayout_release.py: the release layout becomes the fixture layout, and the
result passes make_74k_jsons.py's own A7 checks (the gate the job runs next).

    python -m pytest scripts/lightgen/test_relayout_release.py -q
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import make_74k_jsons as gate  # noqa: E402
import relayout_release as rr  # noqa: E402

SHA_A = "0" * 31 + "a"
SHA_B = "0" * 31 + "b"


def _make_shape(root, sha, thumbnail=True):
    mv = os.path.join(root, sha, "multiview")
    os.makedirs(mv)
    for name in gate.EXPECT_TEX:
        with open(os.path.join(mv, name), "wb") as f:
            f.write(name.encode())
    if thumbnail:
        with open(os.path.join(root, sha, "thumbnail.png"), "wb") as f:
            f.write(b"thumb-" + sha.encode())


def test_relayout_matches_the_fixture_layout_and_passes_a7(tmp_path):
    root, thumbs = str(tmp_path / "release"), str(tmp_path / "thumbs")
    _make_shape(root, SHA_A)
    os.makedirs(thumbs)
    assert rr.relayout_one(root, thumbs, SHA_A) is None

    d = os.path.join(root, SHA_A)
    assert not os.path.exists(os.path.join(d, "multiview"))
    assert set(os.listdir(os.path.join(d, "render_tex"))) == gate.EXPECT_TEX
    # render_cond/000.png is the front albedo view, as in the fixture
    assert os.path.samefile(os.path.join(d, "render_cond", "000.png"),
                            os.path.join(d, "render_tex", "001_albedo.png"))
    assert open(os.path.join(thumbs, SHA_A + ".png"), "rb").read() == b"thumb-" + SHA_A.encode()
    # the gate the job runs next
    assert gate.dir_payload_ok(d) is None
    assert gate.thumbnail_ok(thumbs, d) is None


def test_relayout_is_idempotent(tmp_path):
    root, thumbs = str(tmp_path / "release"), str(tmp_path / "thumbs")
    _make_shape(root, SHA_A)
    os.makedirs(thumbs)
    assert rr.relayout_one(root, thumbs, SHA_A) is None
    assert rr.relayout_one(root, thumbs, SHA_A) is None
    assert gate.dir_payload_ok(os.path.join(root, SHA_A)) is None


def test_missing_pieces_are_reported(tmp_path):
    root, thumbs = str(tmp_path / "release"), str(tmp_path / "thumbs")
    _make_shape(root, SHA_A, thumbnail=False)
    os.makedirs(thumbs)
    assert "thumbnail.png missing" in (rr.relayout_one(root, thumbs, SHA_A) or "")
    assert "not unpacked" in (rr.relayout_one(root, thumbs, SHA_B) or "")


def test_cli_exit_codes(tmp_path):
    root, thumbs = str(tmp_path / "release"), str(tmp_path / "thumbs")
    _make_shape(root, SHA_A)
    good, bad = tmp_path / "good.txt", tmp_path / "bad.txt"
    good.write_text(SHA_A + "\n")
    bad.write_text(SHA_A + "\n" + SHA_B + "\n")
    cmd = [sys.executable, os.path.join(HERE, "relayout_release.py"), "--root", root, "--thumbs_out", thumbs]
    assert subprocess.run(cmd + ["--shas", str(good)]).returncode == 0
    assert subprocess.run(cmd + ["--shas", str(bad)]).returncode == 1
