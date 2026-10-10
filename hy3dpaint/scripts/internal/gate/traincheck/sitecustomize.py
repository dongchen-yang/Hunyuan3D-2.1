"""Training-check hooks, loaded through PYTHONPATH so train.py and the model run unchanged
(after TEXGen-Emission's scripts/internal/gate/traincheck/sitecustomize.py, for pytorch-lightning 1.9.5).

Python imports `sitecustomize` at startup. This one
  1. turns on deterministic mode (detmode/sitecustomize.py, when HUNYUAN_GATE_DETERMINISTIC=1);
  2. when HUNYUAN_TRAINCHECK_OUT names a file, patches pytorch_lightning.Trainer so that
     - `fit` first calls pl.seed_everything(HUNYUAN_TRAINCHECK_SEED, default 42, workers=True), and
     - every Trainer gets one more callback, which appends one JSON line per event to that file:
       each training step (batch names, lr, loss, the gradient norm before clipping, the sum of the
       trainable weights after the step) and each validation pass (the trainable weight sum, val/*).
The callback only reads; it changes no value the run computes.
"""
import os
import runpy

runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "detmode", "sitecustomize.py"))

_OUT = os.environ.get("HUNYUAN_TRAINCHECK_OUT")
if _OUT:
    import json
    import sys

    import pytorch_lightning as pl
    import torch

    _SEED = int(os.environ.get("HUNYUAN_TRAINCHECK_SEED", "42"))

    def _trainable(m):
        return [p for p in m.parameters() if p.requires_grad]

    def _fsum(tensors):
        parts = [t.detach().double().sum() for t in tensors]
        return float(torch.stack(parts).sum().item()) if parts else 0.0

    class TrainCheckRecorder(pl.Callback):
        def __init__(self, path):
            self.path = path
            self._step = self._val = None

        def _write(self, trainer, rec):
            if trainer.global_rank == 0:
                with open(self.path, "a") as f:
                    f.write(json.dumps(rec, sort_keys=True) + "\n")

        def on_fit_start(self, trainer, m):
            self._write(trainer, {"kind": "fit_start", "seed": _SEED, "global_step": int(m.global_step),
                                  "epoch": int(m.current_epoch), "ckpt_path": getattr(trainer, "_traincheck_ckpt", None),
                                  "n_trainable_params": int(sum(p.numel() for p in _trainable(m))),
                                  "n_params": int(sum(p.numel() for p in m.parameters()))})

        def on_train_batch_start(self, trainer, m, batch, batch_idx, *args):
            self._step = {"kind": "train", "gs": int(m.global_step), "epoch": int(m.current_epoch), "bi": int(batch_idx),
                          "names": [os.path.basename(str(n).rstrip("/")) for n in batch["name"]],
                          "lr": float(trainer.optimizers[0].param_groups[0]["lr"])}

        def on_before_optimizer_step(self, trainer, m, optimizer, *args):
            sq = [p.grad.detach().double().pow(2).sum() for p in m.parameters() if p.grad is not None]
            self._step["grad_norm"] = float(torch.stack(sq).sum().sqrt().item()) if sq else 0.0

        def on_train_batch_end(self, trainer, m, outputs, batch, batch_idx, *args):
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            self._step["loss"] = float(loss)
            self._step["param_sum"] = _fsum(_trainable(m))
            self._write(trainer, self._step)

        def on_validation_epoch_start(self, trainer, m):
            self._val = {"kind": "val", "sanity": bool(trainer.sanity_checking), "gs": int(m.global_step),
                         "epoch": int(m.current_epoch), "param_sum": _fsum(_trainable(m))}

        def on_validation_end(self, trainer, m):
            self._val.update({k: float(v) for k, v in trainer.callback_metrics.items() if k.startswith("val/")})
            self._write(trainer, self._val)

    _init, _fit = pl.Trainer.__init__, pl.Trainer.fit

    def __init__(self, *args, **kwargs):
        kwargs["callbacks"] = list(kwargs.get("callbacks") or []) + [TrainCheckRecorder(_OUT)]
        _init(self, *args, **kwargs)

    def fit(self, *args, **kwargs):
        pl.seed_everything(_SEED, workers=True)
        self._traincheck_ckpt = kwargs.get("ckpt_path")
        print(f"[traincheck] pl.seed_everything({_SEED}, workers=True) right before trainer.fit",
              file=sys.stderr, flush=True)
        return _fit(self, *args, **kwargs)

    pl.Trainer.__init__, pl.Trainer.fit = __init__, fit
    print(f"[traincheck] recording to {_OUT}", file=sys.stderr, flush=True)
