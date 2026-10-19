# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import os
import time
import glob
import json
import random
import numpy as np
import torch
import sys

from PIL import Image

from .loader_util import BaseDataset

# Shapes already reported unreadable (per worker process), so each is printed once.
_REPORTED_BAD = set()

# Distinct unreadable shapes __getitem__ substitutes before it raises: a few corrupt files must not
# stop a run, a broken data path must.
MAX_FAILED_INDICES = 3


class TextureDataset(BaseDataset):
    """Six fixed views per shape (render_tex/000-005: albedo, mr, alpha, normal, position, emission)
    and one reference image, chosen by `ref_source`:

      frontal_albedo  render_cond/000.png, the albedo render from the azimuth-0 camera (default)
      thumbnail       <thumbnail_dir>/<basename of the shape dir>.png, the TexVerse thumbnail, read
                      with BaseDataset.load_image and, when `augment_ref` (training only), augmented
                      with augment_image. A missing thumbnail is a bad example, never a fallback.

    `images_cond` holds the reference twice (upstream's shape); the model uses the first.
    """

    # (batch key, PNG suffix); the model reads the maps by key, not in this order.
    MAPS = [("albedo", "albedo"), ("mr", "mr"), ("alpha", "alpha"),
            ("normal", "normal"), ("position", "pos"), ("emission", "emission")]
    REF_SOURCES = ("frontal_albedo", "thumbnail")
    # Upstream's background colors for the reference image.
    _BG_GRAY = [127 / 255.0] * 3
    _BG_BLACK = [0.0] * 3
    _BG_WHITE = [1.0] * 3

    def __init__(self, json_path, num_view=6, image_size=512, use_alpha=True,
                 ref_source="frontal_albedo", thumbnail_dir=None, augment_ref=False):
        # Not BaseDataset.__init__: the list lives in self.dirs (predict_emission_views.py rebinds it).
        with open(json_path) as f:
            self.dirs = json.load(f)
        self.num_view, self.image_size = num_view, image_size
        # False goes with the model's use_alpha false and noise_in_channels 20.
        self.use_alpha = use_alpha
        self.maps = [m for m in self.MAPS if use_alpha or m[0] != "alpha"]
        if ref_source not in self.REF_SOURCES:
            raise ValueError(f"ref_source must be one of {self.REF_SOURCES}, got {ref_source!r}")
        if ref_source == "thumbnail":
            if not thumbnail_dir:
                raise ValueError("ref_source='thumbnail' needs thumbnail_dir (<dir>/<sha>.png)")
            if not os.path.isdir(thumbnail_dir):
                raise FileNotFoundError(f"thumbnail_dir {thumbnail_dir!r} is not a directory")
        elif thumbnail_dir is not None:
            # Rejected, not ignored: it would load render_cond/000.png while meaning thumbnails.
            raise ValueError("thumbnail_dir is only used by ref_source='thumbnail'; passing it "
                             "with ref_source='frontal_albedo' would silently load "
                             "render_cond/000.png instead")
        elif augment_ref:
            raise ValueError("augment_ref is only defined for ref_source='thumbnail'; the "
                             "frontal_albedo rows were never augmented and must stay reproducible")
        self.ref_source = ref_source
        self.thumbnail_dir = thumbnail_dir
        self.augment_ref = bool(augment_ref)

    def __len__(self):
        return len(self.dirs)

    def _img(self, path):
        im = Image.open(path).convert("RGB").resize((self.image_size,) * 2, Image.LANCZOS)
        return torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1)

    @classmethod
    def _draw_bg_color(cls):
        """Upstream's background draw: gray with p=0.6, else black or white; the RGBA composite
        color and augment_image's fill."""
        if random.random() < 0.6:
            return cls._BG_GRAY
        return cls._BG_BLACK if random.random() < 0.5 else cls._BG_WHITE

    def _ref(self, d):
        """The reference image for fixture dir `d`, (3, S, S) float in [0, 1]."""
        if self.ref_source == "frontal_albedo":
            return self._img(os.path.join(d, "render_cond", "000.png"))
        path = os.path.join(self.thumbnail_dir, os.path.basename(d.rstrip("/")) + ".png")
        pil = Image.open(path)
        # load_image takes L, RGB and RGBA; other modes are converted first.
        if pil.mode not in ("RGB", "RGBA"):
            pil = pil.convert("RGB")
        bg = self._draw_bg_color()
        image, _alpha = self.load_image(pil, bg)
        if self.augment_ref:
            image = self.augment_image(image, bg)
        return image

    def _load(self, i):
        """Load one example; raises if any of its PNGs is missing or unreadable."""
        d = self.dirs[i]
        views = range(self.num_view)
        out = {
            f"images_{k}": torch.stack([self._img(os.path.join(d, "render_tex", f"{v:03d}_{s}.png")) for v in views])
            for k, s in self.maps
        }
        ref = self._ref(d)
        out["images_cond"] = torch.stack([ref, ref])
        out["name"] = d
        return out

    def __getitem__(self, i):
        """Load example `i`; if its files are unreadable, substitute a random other example, raising
        after MAX_FAILED_INDICES distinct failures (an error in _load fails on every index).
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
                    print(f"[TextureDataset] UNREADABLE example idx={idx} dir={d}: {e!r} "
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
        # Few examples left: choose among the rest.
        candidates = [j for j in range(n) if self.dirs[j] not in exclude]
        if not candidates:
            raise RuntimeError(f"every one of the {n} examples in this dataset failed to load")
        return random.choice(candidates)
