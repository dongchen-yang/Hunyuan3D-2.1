"""Tests for scripts/convert_config.py: the old class paths of configs and checkpoints are converted, every
other byte kept, and every class path the repository's configs name resolves.

    python -m pytest scripts/lightgen/test_convert_config.py -q
"""

import glob
import importlib
import os
import subprocess
import sys
import types

import pytest
import torch

HY3DPAINT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(HY3DPAINT, "scripts"))
sys.path.insert(0, HY3DPAINT)

import convert_config as cc  # noqa: E402

OLD_CONFIG = """\
# a comment naming hunyuanpaintpbr.model_emission.HunyuanPaintEmission stays as it is
model:
  target: hunyuanpaintpbr.model_emission.HunyuanPaintEmission
  params: {num_view: 6}
data:
  target: src.data.objaverse_hunyuan.DataModuleFromConfig
  params:
    train:
      - target: src.data.dataloader.lightgen_emission_loader.LightgenEmissionDataset
        params: {num_view: 6}
    validation:
      -
        target: "src.data.dataloader.lightgen_emission_loader.LightgenEmissionDataset"  # quoted
"""


def test_convert_text_changes_only_the_old_target_lines():
    text, n = cc.convert_text(OLD_CONFIG)
    assert n == 3
    old, new = OLD_CONFIG.split("\n"), text.split("\n")
    changed = [i for i, (a, b) in enumerate(zip(old, new)) if a != b]
    assert len(old) == len(new) and changed == [2, 8, 12]
    assert new[2] == "  target: hunyuanpaintpbr.unet.model.HunyuanPaint"
    assert new[8] == "      - target: src.data.dataloader.objaverse_loader_forTexturePBR.TextureDataset"
    assert new[12] == ('        target: "src.data.dataloader.objaverse_loader_forTexturePBR.TextureDataset"'
                       "  # quoted")
    assert cc.convert_text(text) == (text, 0)


def _targets(text):
    return [m.group(3) for m in map(cc._TARGET.match, text.split("\n")) if m]


def test_every_class_path_the_configs_name_resolves():
    # Our configs; upstream's cfgs/hunyuan-paint-pbr.yaml names a module upstream itself lacks.
    paths = glob.glob(os.path.join(HY3DPAINT, "cfgs", "lightgen-*.yaml"))
    assert len(paths) == 2, paths
    paths.append(os.path.join(HY3DPAINT, "scripts", "internal", "gate", "traincheck", "traincheck.yaml"))
    for path in paths:
        text, _ = cc.convert_text(open(path).read())
        for target in _targets(text):
            module, _, name = target.rpartition(".")
            assert hasattr(importlib.import_module(module), name), f"{path}: {target}"


class _Thing:
    pass


def _fake_module(monkeypatch, dotted, cls_name):
    mod = types.ModuleType(dotted)
    cls = type(cls_name, (_Thing,), {"__module__": dotted})
    setattr(mod, cls_name, cls)
    monkeypatch.setitem(sys.modules, dotted, mod)
    return cls


def test_convert_checkpoint_rewrites_an_old_pickled_class(tmp_path, monkeypatch):
    old_cls = _fake_module(monkeypatch, "hunyuanpaintpbr.model_emission", "HunyuanPaintEmission")
    src, dst = str(tmp_path / "old.ckpt"), str(tmp_path / "new.ckpt")
    torch.save({"state_dict": {"w": torch.arange(4.0)}, "obj": old_cls()}, src)
    assert cc.convert_checkpoint(src, dst) == 1
    new_cls = _fake_module(monkeypatch, "hunyuanpaintpbr.unet.model", "HunyuanPaint")
    monkeypatch.delitem(sys.modules, "hunyuanpaintpbr.model_emission")
    ck = torch.load(dst, weights_only=False)
    assert type(ck["obj"]) is new_cls and torch.equal(ck["state_dict"]["w"], torch.arange(4.0))


def test_convert_checkpoint_writes_nothing_when_there_is_nothing_to_convert(tmp_path):
    src, dst = str(tmp_path / "plain.ckpt"), str(tmp_path / "out.ckpt")
    torch.save({"state_dict": {"w": torch.ones(2)}}, src)
    assert cc.convert_checkpoint(src, dst) == 0
    assert not os.path.exists(dst)


def test_convert_checkpoint_rejects_a_file_that_is_not_a_torch_zip(tmp_path):
    src = tmp_path / "not.ckpt"
    src.write_bytes(b"not a zip")
    with pytest.raises(Exception):
        cc.convert_checkpoint(str(src), str(tmp_path / "out.ckpt"))


def test_convert_checkpoint_leaves_strings_alone(tmp_path):
    # A string holding the exact bytes of an old GLOBAL opcode, and one naming an old class path, are
    # data: the pickle must still load, and nothing is converted.
    src, dst = str(tmp_path / "strings.ckpt"), str(tmp_path / "out.ckpt")
    payload = {"raw": "chunyuanpaintpbr.model_emission\nHunyuanPaintEmission\n",
               "meta": {"target": "hunyuanpaintpbr.model_emission.HunyuanPaintEmission"}}
    torch.save(payload, src)
    assert cc.convert_checkpoint(src, dst) == 0
    assert not os.path.exists(dst)
    assert torch.load(src, weights_only=False) == payload


def test_cli_keeps_line_endings(tmp_path):
    src, dst = tmp_path / "crlf.yaml", tmp_path / "out.yaml"
    src.write_bytes(b"# c\r\nmodel:\r\n  target: hunyuanpaintpbr.model_emission.HunyuanPaintEmission\r\n  params: {}\r\n")
    subprocess.run([sys.executable, os.path.join(HY3DPAINT, "scripts", "convert_config.py"), str(src), str(dst)],
                   check=True, capture_output=True)
    assert dst.read_bytes() == (b"# c\r\nmodel:\r\n  target: hunyuanpaintpbr.unet.model.HunyuanPaint\r\n"
                                b"  params: {}\r\n")


def test_convert_checkpoint_refuses_protocol_4(tmp_path, monkeypatch):
    # Protocol 4 frames its opcodes: resizing a GLOBAL inside a FRAME would corrupt the pickle.
    old_cls = _fake_module(monkeypatch, "hunyuanpaintpbr.model_emission", "HunyuanPaintEmission")
    src = str(tmp_path / "p4.ckpt")
    torch.save({"obj": old_cls()}, src, pickle_protocol=4)
    with pytest.raises(SystemExit, match="protocol-4"):
        cc.convert_checkpoint(src, str(tmp_path / "out.ckpt"))


def test_convert_checkpoint_refuses_writing_over_its_source(tmp_path, monkeypatch):
    old_cls = _fake_module(monkeypatch, "hunyuanpaintpbr.model_emission", "HunyuanPaintEmission")
    src = tmp_path / "old.ckpt"
    torch.save({"obj": old_cls()}, str(src))
    link = tmp_path / "link.ckpt"
    link.symlink_to(src)
    with pytest.raises(SystemExit, match="another path"):
        cc.convert_checkpoint(str(src), str(link))
