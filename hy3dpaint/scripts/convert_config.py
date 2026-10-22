"""Convert a config or a checkpoint written by earlier versions of this code to the current class paths.

    python scripts/convert_config.py <old.yaml> <new.yaml>              (<new> may be <old>)
    python scripts/convert_config.py --ckpt <old.ckpt> <new.ckpt>

This file holds every such conversion; predict_emission_views.py uses it. Earlier versions named the
model and the loader by two modules this repository no longer has:

    hunyuanpaintpbr.model_emission.HunyuanPaintEmission
        -> hunyuanpaintpbr.unet.model.HunyuanPaint
    src.data.dataloader.lightgen_emission_loader.LightgenEmissionDataset
        -> src.data.dataloader.objaverse_loader_forTexturePBR.TextureDataset

They are the same classes, now in upstream's files. A config names them in `target:` lines; only
those lines change and every other byte is kept. predict_emission_views.py converts its --config on
load; train.py needs a converted file.

A checkpoint is converted without loading it: the class paths its pickle names are rewritten and every
tensor is copied as stored. A checkpoint that pickles no class of this repository, only weights and
optimizer state, loads as it is; --ckpt then says so and writes nothing.
"""
import argparse
import os
import pickletools
import re
import shutil
import zipfile

RENAMES = {
    "hunyuanpaintpbr.model_emission.HunyuanPaintEmission": "hunyuanpaintpbr.unet.model.HunyuanPaint",
    "src.data.dataloader.lightgen_emission_loader.LightgenEmissionDataset":
        "src.data.dataloader.objaverse_loader_forTexturePBR.TextureDataset",
}
OLD_MODULES = sorted({old.rpartition(".")[0] for old in RENAMES})
# A `target:` line, plain or as a list item, its value bare or quoted.
_TARGET = re.compile(r"""^(\s*(?:-\s+)?target:\s*)(['"]?)([\w.]+)\2(\s*(?:#.*)?)$""")


def convert_text(text):
    """A config's text with its old class paths converted; returns (text, how many lines changed)."""
    out, n = [], 0
    for line in text.split("\n"):
        m = _TARGET.match(line)
        if m and m.group(3) in RENAMES:
            line = m.group(1) + m.group(2) + RENAMES[m.group(3)] + m.group(2) + m.group(4)
            n += 1
        out.append(line)
    return "\n".join(out), n


def _rewrite_globals(data):
    """data.pkl with its GLOBAL opcodes (protocol 2, torch.save's default) for the old classes rewritten;
    returns (new bytes, how many changed). Only real opcodes are touched, found by parsing the pickle, so
    a string that happens to hold the same bytes is left alone."""
    ops = list(pickletools.genops(data))
    # Protocol 4 is not handled: STACK_GLOBAL takes its names from the stack, and a FRAME's length would
    # go stale once a GLOBAL inside it changes size.
    if any(op.name in ("STACK_GLOBAL", "FRAME") for op, _, _ in ops) and any(m.encode() in data for m in OLD_MODULES):
        raise SystemExit("a protocol-4 pickle (FRAME or STACK_GLOBAL) that mentions an old module: not handled")
    out, last, n = [], 0, 0
    for k, (op, arg, pos) in enumerate(ops):
        if op.name != "GLOBAL":
            continue
        module, name = arg.split(" ", 1)
        new = RENAMES.get(f"{module}.{name}")
        if new is None:
            continue
        end = ops[k + 1][2] if k + 1 < len(ops) else len(data)
        new_module, _, new_name = new.rpartition(".")
        out += [data[last:pos], f"c{new_module}\n{new_name}\n".encode()]
        last, n = end, n + 1
    out.append(data[last:])
    return b"".join(out), n


def convert_checkpoint(src, dst):
    """Copy a torch zip checkpoint, rewriting the class references of its data.pkl; returns how many
    references changed. Nothing is written when none does. Written to <dst>.part, then renamed."""
    with zipfile.ZipFile(src) as zin:
        pkl = [i for i in zin.infolist() if i.filename == "data.pkl" or i.filename.endswith("/data.pkl")]
        if len(pkl) != 1:
            raise SystemExit(f"{src}: expected one data.pkl in a torch zip checkpoint, found {len(pkl)}")
        data, n = _rewrite_globals(zin.read(pkl[0]))
        if n == 0:
            return 0
        if os.path.realpath(src) == os.path.realpath(dst):
            raise SystemExit("--ckpt writes a new file: give <new.ckpt> another path than <old.ckpt>")
        part = dst + ".part"
        with zipfile.ZipFile(part, "w", zipfile.ZIP_STORED, allowZip64=True) as zout:
            for info in zin.infolist():
                out = zipfile.ZipInfo(info.filename, date_time=info.date_time)
                out.compress_type = zipfile.ZIP_STORED
                if info is pkl[0]:
                    zout.writestr(out, data)
                else:
                    out.file_size = info.file_size
                    with zin.open(info) as fi, zout.open(out, "w", force_zip64=info.file_size > 2**31) as fo:
                        shutil.copyfileobj(fi, fo, 64 << 20)
    os.replace(part, dst)
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", action="store_true", help="convert a checkpoint instead of a config")
    ap.add_argument("src", help="a config (or, with --ckpt, a checkpoint) written by the older code")
    ap.add_argument("dst", help="where to write the converted file")
    a = ap.parse_args()
    if a.ckpt:
        n = convert_checkpoint(a.src, a.dst)
        print(f"{n} class reference(s) converted: {a.src} -> {a.dst}" if n else
              f"{a.src} pickles no class path of the older code; it loads as it is (nothing written)")
        return
    with open(a.src, newline="") as f:      # newline="": line endings are kept as they are
        text, n = convert_text(f.read())
    with open(a.dst, "w", newline="") as f:
        f.write(text)
    print(f"{n} class path(s) converted: {a.src} -> {a.dst}")


if __name__ == "__main__":
    main()
