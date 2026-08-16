import json
import os
import random
import sys

import numpy as np
import torch
from PIL import Image

# Shapes whose load has already been reported, so a permanently-corrupt file does not print on
# every visit. Dataloader workers are separate processes, so this is deliberately per-worker
# state, and with persistent_workers off the workers are respawned each epoch -- so the true
# bound is once per bad shape per worker per epoch (up to num_workers x world_size lines), not
# once per run. That is fine for a log line; carrying a lock across process boundaries is not
# worth it.
_REPORTED_BAD = set()

# How many *distinct* shape dirs may fail to load before __getitem__ gives up and raises. A handful
# of corrupt PNGs in a 71.6k-shape staged fixture must not kill a 57-hour run; a systemic breakage
# (wrong json root, unmounted scratch, truncated rsync) makes every index fail and must still stop
# the run immediately rather than spin forever resampling.
MAX_FAILED_INDICES = 3


class LightgenEmissionDataset(torch.utils.data.Dataset):
    """Fixed-6-view loader for lightgen multiview emission fixtures.

    Differences from upstream TextureDataset: no lighting-suffix logic, no random
    view subsampling (views are the canonical 000-005), adds images_emission and
    images_alpha, and the reference image is render_cond/000.png duplicated twice
    (upstream code slices images_cond[:, 0:1] and [:, 1:2]).

    `images_alpha` is the per-texel glTF opacity map, added for conditioning parity
    with the other LightGen baselines (TEXGen's 13ch variant, TRELLIS.2 and
    SegviGen all condition on alpha). Its PNG stores the scalar replicated across
    RGB, so it needs no special handling here -- it rides the same `_img` path as
    every other map. A fixture rendered before the alpha map existed holds only 30
    render_tex PNGs and will raise here rather than train without the condition;
    upgrade it with `data_processing/multiview_render_pipeline/append_alpha_views.py`."""

    # (batch key, PNG suffix). Order is documentation only -- the model reads by key -- but it
    # matches the conv_in concat order in hunyuanpaintpbr/unet/modules.py, so keep it that way.
    MAPS = [("albedo", "albedo"), ("mr", "mr"), ("alpha", "alpha"),
            ("normal", "normal"), ("position", "pos"), ("emission", "emission")]

    def __init__(self, json_path, num_view=6, image_size=512, use_alpha=True):
        with open(json_path) as f:
            self.dirs = json.load(f)
        self.num_view, self.image_size = num_view, image_size
        # False only to reproduce the archived 20-channel run against a pre-alpha fixture; it
        # must then be false on the model too (HunyuanPaintEmission.use_alpha) and
        # noise_in_channels must be 20. The model cross-checks all three against the real
        # conv_in width per batch, so a half-flipped combination raises rather than trains.
        self.use_alpha = use_alpha
        self.maps = [m for m in self.MAPS if use_alpha or m[0] != "alpha"]

    def __len__(self):
        return len(self.dirs)

    def _img(self, path):
        im = Image.open(path).convert("RGB").resize((self.image_size,) * 2, Image.LANCZOS)
        return torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1)

    def _load(self, i):
        """Load one example. Raises if any of its 37 PNGs is missing or undecodable."""
        d = self.dirs[i]
        views = range(self.num_view)
        out = {
            f"images_{k}": torch.stack([self._img(os.path.join(d, "render_tex", f"{v:03d}_{s}.png")) for v in views])
            for k, s in self.maps
        }
        ref = self._img(os.path.join(d, "render_cond", "000.png"))
        out["images_cond"] = torch.stack([ref, ref])
        out["name"] = d
        return out

    def __getitem__(self, i):
        """Load example `i`, substituting a random other example if its files are unreadable.

        A substituted sample is a real training example from the same split, so the batch shape and
        the split's contents are unaffected; only the visit count of one shape shifts. Substitution
        is bounded (see MAX_FAILED_INDICES) so a global fixture problem still fails loudly.

        The catch is deliberately broad rather than an allowlist of PIL/OS exception types: the
        point of the guard is to survive the *unanticipated* per-file failure at hour 40, and a
        narrow list can only cover what we thought of. Breadth is safe here precisely because of
        the bound -- a programming error in _load fails on every index, so it raises on the third
        one instead of being masked. MemoryError is re-raised immediately: retrying an allocation
        failure only makes the machine worse, and it is never a property of one shape.
        """
        failed = []
        idx = i
        while True:
            try:
                return self._load(idx)
            except MemoryError:
                raise
            except Exception as e:  # missing / truncated / zlib-corrupt PNG, unreadable dir, ...
                d = self.dirs[idx]
                failed.append((d, repr(e)))
                if d not in _REPORTED_BAD:
                    _REPORTED_BAD.add(d)
                    print(f"[LightgenEmissionDataset] UNREADABLE example idx={idx} dir={d}: {e!r} "
                          f"-- substituting a random other example", file=sys.stderr, flush=True)
                if len(failed) >= MAX_FAILED_INDICES:
                    raise RuntimeError(
                        f"{len(failed)} distinct examples failed to load starting from idx={i}; "
                        f"refusing to keep resampling. Failures: {failed}"
                    ) from e
                idx = self._resample(exclude={f[0] for f in failed})

    def _resample(self, exclude):
        """Pick a random index whose dir is not in `exclude`; fall back to any index."""
        n = len(self.dirs)
        for _ in range(32):
            j = random.randrange(n)
            if self.dirs[j] not in exclude:
                return j
        # Dataset is smaller than the exclusion set (only reachable for tiny pilot fixtures).
        candidates = [j for j in range(n) if self.dirs[j] not in exclude]
        if not candidates:
            raise RuntimeError(f"every one of the {n} examples in this dataset failed to load")
        return random.choice(candidates)
