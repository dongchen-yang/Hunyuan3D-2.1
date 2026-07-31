import json
import os
import numpy as np
import torch
from PIL import Image


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

    def __getitem__(self, i):
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
