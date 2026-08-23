import json
import os
import random
import sys

import numpy as np
import torch
from PIL import Image

from .loader_util import BaseDataset

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


class LightgenEmissionDataset(BaseDataset):
    """Fixed-6-view loader for lightgen multiview emission fixtures.

    Differences from upstream TextureDataset: no lighting-suffix logic, no random
    view subsampling (views are the canonical 000-005), adds images_emission and
    images_alpha. The reference image (`images_cond`, the same image stacked twice --
    upstream code slices images_cond[:, 0:1] and [:, 1:2], and only slot 0 is used
    by model_emission.py) is selected by `ref_source`:

      frontal_albedo  render_cond/000.png, our unlit GT-albedo render from upstream's
                      azimuth-0 camera (== render_tex/001_albedo.png). Every Hunyuan row
                      up to and including _nonzero_nocopy trained this way. Default.
      thumbnail       <thumbnail_dir>/<basename(fixture dir)>.png -- the TexVerse thumbnail
                      TEXGen (CLIP) and TRELLIS.2 (DINOv3) condition on; the fixture dir's
                      basename IS the sha that names the thumbnail. Loaded through
                      upstream's own BaseDataset.load_image (plain square resize, RGBA
                      composited onto the drawn background colour) and, when `augment_ref`,
                      through BaseDataset.augment_image with upstream's defaults (training
                      only; validation and inference pass augment_ref=False). No fallback to
                      render_cond: a missing thumbnail is a bad example (substituted, and
                      fatal after MAX_FAILED_INDICES distinct ones) -- TEXGen once trained a
                      whole run on its albedo UV map behind a silent thumbnail fallback.
                      Spec: lightgen docs/superpowers/specs/2026-08-23-hunyuan3d-paint-thumbnail-reference-design.md

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
    REF_SOURCES = ("frontal_albedo", "thumbnail")
    # Upstream's reference background colours (objaverse_loader_forTexturePBR.py:52-54).
    _BG_GRAY = [127 / 255.0] * 3
    _BG_BLACK = [0.0] * 3
    _BG_WHITE = [1.0] * 3

    def __init__(self, json_path, num_view=6, image_size=512, use_alpha=True,
                 ref_source="frontal_albedo", thumbnail_dir=None, augment_ref=False):
        # BaseDataset.__init__ is deliberately NOT called: all it does is read json_path into
        # self.data and set num_view/image_size. This class keeps its list in self.dirs (which
        # predict_emission_views.py rebinds for --limit) and sets the scalars itself; the base
        # class is inherited for load_image / augment_image -- upstream's reference path,
        # reused verbatim rather than copied.
        with open(json_path) as f:
            self.dirs = json.load(f)
        self.num_view, self.image_size = num_view, image_size
        # False only to reproduce the archived 20-channel run against a pre-alpha fixture; it
        # must then be false on the model too (HunyuanPaintEmission.use_alpha) and
        # noise_in_channels must be 20. The model cross-checks all three against the real
        # conv_in width per batch, so a half-flipped combination raises rather than trains.
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
            # Rejected, not ignored: a caller that passes thumbnail_dir but forgets
            # ref_source='thumbnail' would train on render_cond/000.png while believing it
            # trained on thumbnails -- the silent-wrong-condition failure this selector exists
            # to prevent (resolve_ref_source rejects the mirror case at inference).
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
        """Upstream's per-example background draw (objaverse_loader_forTexturePBR.py:101-106):
        gray with p=0.6, otherwise black/white 50/50. Used as the RGBA composite colour and as
        the fill colour of augment_image; an RGB thumbnail only sees it through the fill."""
        if random.random() < 0.6:
            return cls._BG_GRAY
        return cls._BG_BLACK if random.random() < 0.5 else cls._BG_WHITE

    def _ref(self, d):
        """The reference image for fixture dir `d`, (3, S, S) float in [0, 1]."""
        if self.ref_source == "frontal_albedo":
            return self._img(os.path.join(d, "render_cond", "000.png"))
        path = os.path.join(self.thumbnail_dir, os.path.basename(d.rstrip("/")) + ".png")
        pil = Image.open(path)
        # load_image handles L / RGB / RGBA; anything else (palette, CMYK) is normalised to RGB
        # first so it cannot index a missing channel axis.
        if pil.mode not in ("RGB", "RGBA"):
            pil = pil.convert("RGB")
        bg = self._draw_bg_color()
        image, _alpha = self.load_image(pil, bg)
        if self.augment_ref:
            image = self.augment_image(image, bg)
        return image

    def _load(self, i):
        """Load one example. Raises if any of its PNGs (36 views + the reference) is missing or undecodable."""
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
