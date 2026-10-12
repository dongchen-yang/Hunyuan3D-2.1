#!/bin/bash
# Unpack the Hugging Face dataset (3dlg-hcvc/LightgenBench) into the data folder this code trains and
# runs on, and write the training config that reads it:
#
#   bash scripts/lightgenbench/unpack_hf_dataset.sh <download_dir> <data_dir>
#
# <download_dir> holds a download of the dataset with at least splits.json, checksums.sha256 and
# data/{train,val,test}/{multiview,thumbnail}/*.tar, for example
#   huggingface-cli download 3dlg-hcvc/LightgenBench --repo-type dataset --local-dir <download_dir> \
#     --include "splits.json" "checksums.sha256" "data/*/multiview/*.tar" "data/*/thumbnail/*.tar"
# The tars are checked against checksums.sha256, unpacked, and laid out the way the loader
# (src/data/dataloader/objaverse_loader_forTexturePBR.py) reads them:
#   <data_dir>/shapes/<uuid>/render_tex/   the six views' maps (00N_{albedo,mr,normal,pos,emission,alpha}.png)
#   <data_dir>/shapes/<uuid>/render_cond/000.png
#   <data_dir>/thumbs/<uuid>.png           the reference image
# Then the example lists, from splits.json: release_train.json (all of train), release_val64.json (the
# 64 validation shapes training validates on, scripts/lightgen/val64_release_shas.txt) and
# release_test.json (all of test). Last, the training config:
#   <data_dir>/lightgen-emission-release-alpha-thumb.yaml
# is cfgs/lightgen-emission-release-alpha-thumb.yaml with its paths filled in. It names the
# Hunyuan3D-2.1 weights the model starts from, at the revision every run used; download them first:
#   huggingface-cli download tencent/Hunyuan3D-2.1 --revision 0b94677654c57bb9a6b6845cd7b704ccf551d327 \
#     --include "hunyuan3d-paintpbr-v2-1/*"
# or set SNAPSHOT=<a copy of its hunyuan3d-paintpbr-v2-1/ directory>. NUM_WORKERS sets the data loader
# workers per GPU (default: CPUs / 4 - 2, at least 2). JOBS (default 8) tars are checked and unpacked
# in parallel. Run it in the training env (requirements-emission.txt). The full release needs about
# 40 GB.
set -euo pipefail
HY=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)   # hy3dpaint/
HFD=$(readlink -f "${1:?download dir}")
D=$(readlink -f -m "${2:?data dir}")
JOBS=${JOBS:-8}
REV=0b94677654c57bb9a6b6845cd7b704ccf551d327
TEMPLATE=$HY/cfgs/lightgen-emission-release-alpha-thumb.yaml
for f in splits.json checksums.sha256; do
  [ -f "$HFD/$f" ] || { echo "ABORT: no $f in $HFD"; exit 2; }
done

# The weights first: a missing download should cost seconds, not the unpack.
if [ -z "${SNAPSHOT:-}" ]; then
  SNAPSHOT=$(python - "$REV" <<'PY'
import sys
from huggingface_hub import snapshot_download
try:
    p = snapshot_download("tencent/Hunyuan3D-2.1", revision=sys.argv[1],
                          allow_patterns=["hunyuan3d-paintpbr-v2-1/*"], local_files_only=True)
except Exception as e:
    sys.exit(f"ABORT: Hunyuan3D-2.1 at revision {sys.argv[1]} is not in the Hugging Face cache ({e}); "
             "download it (see this script's header) or set SNAPSHOT")
print(p + "/hunyuan3d-paintpbr-v2-1")
PY
)
fi
SNAPSHOT=$(readlink -f "$SNAPSHOT")
[ -f "$SNAPSHOT/model_index.json" ] || { echo "ABORT: $SNAPSHOT has no model_index.json"; exit 2; }

mkdir -p "$D/shapes" "$D/thumbs" "$D/splits"
cd "$HFD"
WANT=$D/.want.sha256
grep -E '  (splits\.json|data/(train|val|test)/(multiview|thumbnail)/[^/]+\.tar)$' checksums.sha256 > "$WANT"
echo "[check] $(wc -l < "$WANT") files against checksums.sha256"
xargs -P "$JOBS" -L 1 sh -c 'echo "$0  $1" | sha256sum -c --quiet -' < "$WANT"

# Only the tars just checked are unpacked; any other tar in <download_dir> is ignored. Each holds
# <uuid>/multiview/... or <uuid>/thumbnail.png; multiview/ is unpacked as render_tex/, the name the
# loader reads, so a second run into the same <data_dir> overwrites the same files.
echo "[unpack] multiview and thumbnail tars -> $D/shapes/<uuid>/"
awk '$2 ~ /\.tar$/ {print $2}' "$WANT" \
  | xargs -P "$JOBS" -I{} tar xf {} -C "$D/shapes" --transform 's|^\([^/]*\)/multiview/|\1/render_tex/|'
rm -f "$WANT"

python - "$HFD/splits.json" "$D/splits" <<'PY'
import json, sys
splits = json.load(open(sys.argv[1]))
if sorted(splits) != ["test", "train", "val"]:
    sys.exit(f"{sys.argv[1]}: keys {sorted(splits)}, expected test, train, val")
for k in ("train", "val", "test"):
    open(f"{sys.argv[2]}/{k}.txt", "w").write("".join(u + "\n" for u in splits[k]))
print("[splits]", {k: len(splits[k]) for k in ("train", "val", "test")})
PY

echo "[layout] render_cond/000.png, thumbs/<uuid>.png"
python "$HY/scripts/lightgen/relayout_release.py" --root "$D/shapes" --thumbs_out "$D/thumbs" \
  --shas "$D/splits/train.txt" "$D/splits/val.txt" "$D/splits/test.txt"

# The training lists through the builder the release run used, with its checks: the splits disjoint,
# every listed shape whole on disk with its thumbnail, none missing.
echo "[lists] release_train.json, release_val64.json, release_test.json"
python "$HY/scripts/lightgen/make_74k_jsons.py" \
  --train_shas "$D/splits/train.txt" --val_shas "$D/splits/val.txt" --test_shas "$D/splits/test.txt" \
  --val64_shas "$HY/scripts/lightgen/val64_release_shas.txt" \
  --fixture_root "$D/shapes" --thumbnail_dir "$D/thumbs" \
  --out_train "$D/release_train.json" --out_val64 "$D/release_val64.json" \
  --expect_train "$(wc -l < "$D/splits/train.txt")"
# The test list gets the same check (every file the loader opens), since the loader would replace an
# unreadable shape by another one rather than stop.
python - "$D/splits/test.txt" "$D/shapes" "$D/release_test.json" <<'PY'
import json, os, sys
shas = [l.strip() for l in open(sys.argv[1]) if l.strip()]
dirs = [os.path.join(sys.argv[2], s) for s in shas]
need = [f"render_tex/{v:03d}_{m}.png" for v in range(6)
        for m in ("albedo", "mr", "normal", "pos", "emission", "alpha")] + ["render_cond/000.png"]
bad = [f"{d}/{n}" for d in dirs for n in need if not os.path.isfile(os.path.join(d, n))]
if bad:
    sys.exit(f"ABORT: {len(bad)} files missing from the test shapes, e.g. {bad[0]}")
json.dump(dirs, open(sys.argv[3], "w"))
print(f"WROTE {sys.argv[3]}  entries={len(dirs)}")
PY

# The config: the template's tokens filled, the comment lines left as they are.
CFG=$D/$(basename "$TEMPLATE")
NW=${NUM_WORKERS:-$(( $(nproc) / 4 - 2 ))}; [ "$NW" -ge 2 ] || NW=2
python - "$TEMPLATE" "$CFG" "$SNAPSHOT" "$D" "$NW" <<'PY'
import re, sys
template, cfg, snapshot, data_dir, workers = sys.argv[1:]
out = []
for line in open(template).read().split("\n"):
    if not line.lstrip().startswith("#"):
        line = re.sub(r"^(\s*pretrained_model_name_or_path:).*", lambda m: f"{m.group(1)} {snapshot}", line)
        line = line.replace("__STAGE__", data_dir).replace("__NUM_WORKERS__", workers)
        if re.search(r"__[A-Z_]+__", line):
            sys.exit(f"ABORT: unfilled token in {template}: {line.strip()}")
    out.append(line)
open(cfg, "w").write("\n".join(out))
PY
echo "[config] $CFG (snapshot $SNAPSHOT, num_workers $NW)"
echo "[ok] $D"
