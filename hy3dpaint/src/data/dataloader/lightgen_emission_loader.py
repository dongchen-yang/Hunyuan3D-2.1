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
    view subsampling (views are the canonical 000-005), adds images_emission, and
    the reference image is render_cond/000.png duplicated twice (upstream code
    slices images_cond[:, 0:1] and [:, 1:2])."""

    def __init__(self, json_path, num_view=6, image_size=512):
        with open(json_path) as f:
            self.dirs = json.load(f)
        self.num_view, self.image_size = num_view, image_size

    def __len__(self):
        return len(self.dirs)

    def _img(self, path):
        im = Image.open(path).convert("RGB").resize((self.image_size,) * 2, Image.LANCZOS)
        return torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1)

    def _load(self, i):
        """Load one example. Raises if any of its 31 PNGs is missing or undecodable."""
        d = self.dirs[i]
        views = range(self.num_view)
        out = {
            f"images_{k}": torch.stack([self._img(os.path.join(d, "render_tex", f"{v:03d}_{s}.png")) for v in views])
            for k, s in [("albedo", "albedo"), ("mr", "mr"), ("normal", "normal"),
                         ("position", "pos"), ("emission", "emission")]
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
