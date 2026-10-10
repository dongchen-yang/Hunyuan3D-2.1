"""Print one JSON line summarizing a saved checkpoint: what train-check.sh appends to the record.

  python summarize_ckpt.py <ckpt>        (run from the tree's hy3dpaint/)

Fields: the top-level keys, epoch, global_step, Lightning's lr_schedulers (last_epoch, _last_lr),
the optimizer's param-group lr and state count, the state_dict size, the float64 sum of its finite floating
values and the count of the non-finite ones, a digest of the
state_dict's key names, shapes and dtypes, the float64 sum of the optimizer state, and every non-torch class the file pickles (the class paths a checkpoint depends on).
"""
import hashlib
import json
import os
import pickle
import sys
import types

import torch

sys.path.insert(0, os.getcwd())
classes = set()


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if not module.startswith(("torch", "collections", "builtins", "_codecs", "numpy")):
            classes.add(f"{module}.{name}")
        return super().find_class(module, name)


_pm = types.ModuleType("pickle")
_pm.__dict__.update({k: getattr(pickle, k) for k in dir(pickle) if not k.startswith("__")})
_pm.Unpickler = _Unpickler

ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False, pickle_module=_pm)
sd = ck.get("state_dict", {})
print(json.dumps({
    "kind": "ckpt", "file": os.path.relpath(sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else os.path.basename(sys.argv[1]),
    "keys": sorted(ck.keys()), "epoch": ck.get("epoch"), "global_step": ck.get("global_step"),
    "lightning_scheduler": [[s.get("last_epoch"), s.get("_last_lr")] for s in ck.get("lr_schedulers", [])],
    "optimizer_lr": [[g.get("lr") for g in o.get("param_groups", [])] for o in ck.get("optimizer_states", [])],
    "n_optimizer_states": [len(o.get("state", {})) for o in ck.get("optimizer_states", [])],
    "n_state": len(sd),
    # The schedule buffers (sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod) hold inf at the last
    # timestep: sum the finite values, and count the others as structure.
    "state_sum": float(sum(v[torch.isfinite(v)].double().sum().item() for v in sd.values() if torch.is_tensor(v) and v.is_floating_point())),
    "n_nonfinite": int(sum((~torch.isfinite(v)).sum().item() for v in sd.values() if torch.is_tensor(v) and v.is_floating_point())),
    "state_keys_sha256": hashlib.sha256("\n".join(f"{k} {tuple(v.shape)} {v.dtype}" for k, v in sorted(sd.items())
                                                    if torch.is_tensor(v)).encode()).hexdigest(),
    "optimizer_sum": float(sum(t[torch.isfinite(t)].double().sum().item() for o in ck.get("optimizer_states", [])
                               for st in o.get("state", {}).values() for t in st.values()
                               if torch.is_tensor(t) and t.is_floating_point())),
    "classes": sorted(classes),
}, sort_keys=True))
