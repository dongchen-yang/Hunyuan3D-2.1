"""A weights-only copy of a published checkpoint for the inference gate (REFACTOR_TODO.md step 2).

The parent's predictor reads only `state["state_dict"]` and `state["global_step"]` from a checkpoint
(`evaluation/mvpaint_bridge/predict_emission_views.py:207-219`). A 17.9 GB trough also carries the
optimizer state; loading it whole needs about 27 GB of host RAM per process, which most Solar nodes
cannot spare. The copy holds the same tensors, so the predictor's output is the same. The training
check's resume keeps reading the full file.

  python slim_ckpt.py <full.ckpt> <out.weights.ckpt>
"""
import os
import sys

import torch


def main(src, dst):
    state = torch.load(src, map_location="cpu", mmap=True, weights_only=False)
    slim = {"state_dict": state["state_dict"], "global_step": state["global_step"], "epoch": state.get("epoch")}
    tmp = dst + ".part"
    torch.save(slim, tmp)
    os.replace(tmp, dst)
    print(f"[slim] {src} -> {dst}: {len(slim['state_dict'])} tensors, global_step {slim['global_step']}, "
          f"{os.path.getsize(dst) / 2**30:.2f} GiB")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
