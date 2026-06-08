"""Two-phase fine-tuning of YOLOv9m + DINOv3 ConvNeXt on Open Images V7 @ 640.

Phase 1: backbone frozen, train neck/head fast (MuSGD).
Phase 2: unfreeze DINOv3 ConvNeXt, gentle full fine-tune (AdamW, low lr).

The backbone module sets ``_ultralytics_keep_frozen=True`` in the YAML
(``DINOv3ConvNeXt(..., freeze=True, ...)``). The Ultralytics trainer always
honours that marker, so ``freeze=[...]`` cannot unfreeze it. Phase 2 clears the
marker inside an ``on_pretrain_routine_end`` callback: the trainer rebuilds the
model from YAML (re-applying the marker), so the unfreeze must run on the real
``trainer.model`` after optimizer setup.
"""

from ultralytics import YOLO
from ultralytics.nn.modules import DINOv3ConvNeXt

MODEL_CFG = "ultralytics/cfg/models/26d/yolod9m.yaml"
DATA = "open-images-v7.yaml"
IMGSZ = 640
DEVICE = 0
SEED = 0
BACKBONE_LR_MULT = 0.1  # backbone trains at 0.1x the head LR to preserve pretrained features


def _is_backbone_module(module) -> bool:
    """Identify the DINOv3 ConvNeXt backbone, robust to checkpoint import-path mismatch.

    ``isinstance`` alone can return False for a module reloaded from a ``.pt``
    checkpoint (different import path / class identity), so fall back to the
    class name -- the same guard the Ultralytics fuse() code uses.
    """
    return isinstance(module, DINOv3ConvNeXt) or module.__class__.__name__ == "DINOv3ConvNeXt"



def _unfreeze_backbone(model_root) -> int:
    """Clear the keep-frozen marker and re-enable grads on the DINOv3 backbone.

    Operates on the actual ``nn.Module`` the trainer optimizes (``trainer.model``),
    not on the high-level ``YOLO`` wrapper: the trainer rebuilds the model from
    YAML inside ``get_model``, so any unfreeze done before ``train()`` is lost.
    ``isinstance`` alone is also unreliable across that rebuild (different import
    path / class identity), so ``_is_backbone_module`` matches by class name too.
    Returns the number of backbone modules unfrozen.
    """
    cleared = 0
    for module in model_root.modules():
        if _is_backbone_module(module):
            if getattr(module, "_ultralytics_keep_frozen", False):
                del module._ultralytics_keep_frozen
            for p in module.parameters():
                p.requires_grad_(True)
            cleared += 1
    return cleared


def make_backbone_finetune_callback(lr_mult: float = BACKBONE_LR_MULT):
    """Build an ``on_pretrain_routine_end`` callback that unfreezes the DINOv3
    backbone and gives it a discriminative (lower) learning rate.

    Why this hook: the trainer rebuilds the model from YAML inside ``get_model``
    and re-applies the ``_ultralytics_keep_frozen`` marker, so unfreezing before
    ``train()`` has no effect. ``on_pretrain_routine_end`` runs on the real
    ``trainer.model`` after the optimizer and the LambdaLR scheduler are built
    but before the first scheduler step. ``build_optimizer`` does not filter by
    ``requires_grad``, so the (still-frozen) backbone params are already present
    in the optimizer groups and can be re-enabled and re-grouped here.

    Steps: (1) clear the marker and set ``requires_grad=True`` on the backbone;
    (2) move backbone params into their own param groups with ``lr``/``initial_lr``
    scaled by ``lr_mult`` -- renaming the group also makes the warmup loop treat
    backbone biases as non-bias (floor 0.0 instead of ``warmup_bias_lr``);
    (3) resync the scheduler's ``base_lrs`` and ``lr_lambdas`` to the new, longer
    group list (LambdaLR captured both at construction; appended groups would
    otherwise be dropped by its internal ``zip`` and never decay).
    """

    def _setup(trainer) -> None:
        cleared = _unfreeze_backbone(trainer.model)
        if cleared == 0:
            raise RuntimeError(
                "Backbone fine-tune callback found no DINOv3ConvNeXt module in trainer.model. "
                "Check that the checkpoint/YAML actually contains a DINOv3ConvNeXt backbone."
            )

        backbone_ids = {
            id(p)
            for module in trainer.model.modules()
            if _is_backbone_module(module)
            for p in module.parameters()
        }

        optimizer = trainer.optimizer
        new_groups = []
        for g in optimizer.param_groups:
            backbone_params = [p for p in g["params"] if id(p) in backbone_ids]
            if not backbone_params:
                continue
            # Keep only the head params in the original group.
            g["params"] = [p for p in g["params"] if id(p) not in backbone_ids]
            ng = {k: v for k, v in g.items() if k != "params"}
            ng["params"] = backbone_params
            ng["lr"] = g["lr"] * lr_mult
            if "initial_lr" in g:
                ng["initial_lr"] = g["initial_lr"] * lr_mult
            # Rename so the warmup loop's ``param_group == "bias"`` check skips
            # backbone biases (they then warm up from 0.0, not warmup_bias_lr).
            ng["param_group"] = f"{g.get('param_group', 'weight')}_backbone"
            new_groups.append(ng)

        if not new_groups:
            raise RuntimeError(
                "Backbone params were not present in the optimizer; cannot apply discriminative LR."
            )

        for ng in new_groups:
            optimizer.add_param_group(ng)

        # LambdaLR captured base_lrs and lr_lambdas at construction (one entry per
        # original group). Extend BOTH to the new group count, otherwise the
        # appended backbone groups are dropped by the zip() inside the scheduler
        # and never decay with the cosine schedule.
        if trainer.scheduler is not None:
            trainer.scheduler.base_lrs = [grp["initial_lr"] for grp in optimizer.param_groups]
            trainer.scheduler.lr_lambdas = [trainer.lf] * len(optimizer.param_groups)

        n_bb = sum(len(grp["params"]) for grp in new_groups)
        print(
            f"[backbone-lr] unfroze {cleared} backbone module(s); "
            f"backbone LR set to {lr_mult}x head LR across {n_bb} param tensor(s)"
        )

    return _setup



def phase1_frozen_head() -> str:
    """Train neck/head with the backbone frozen. Returns path to best.pt."""
    model = YOLO(MODEL_CFG)
    results = model.train(
        data=DATA,
        imgsz=IMGSZ,
        epochs=20,            # short: head only needs to latch onto DINO features
        optimizer="MuSGD",
        lr0=0.01,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3,
        batch=48,
        # backbone already frozen via YAML marker; freeze=[0] is belt-and-suspenders
        freeze=[0],
        close_mosaic=5,
        amp=True,
        cache=False,          # OIv7 is huge -> do not cache to RAM/disk
        device=DEVICE,
        seed=SEED,
        name="oiv7_yolod9m_p1_frozen",
    )
    return str(results.save_dir / "weights" / "best.pt")


def phase2_finetune(weights: str) -> None:
    """Unfreeze the DINOv3 ConvNeXt backbone and fine-tune end-to-end.

    The unfreeze and the discriminative LR are both applied inside the
    ``on_pretrain_routine_end`` callback (see make_backbone_finetune_callback),
    because the trainer rebuilds the model from YAML and would otherwise keep the
    backbone frozen via the ``_ultralytics_keep_frozen`` marker.
    """
    model = YOLO(weights)

    # Unfreeze backbone + backbone LR = BACKBONE_LR_MULT x head LR, on trainer.model.
    model.add_callback("on_pretrain_routine_end", make_backbone_finetune_callback())

    model.train(
        data=DATA,
        imgsz=IMGSZ,
        epochs=80,            # the bulk of the budget goes here
        optimizer="AdamW",
        lr0=2e-4,             # critical: keep small so pretrained features survive
        lrf=0.01,
        weight_decay=0.01,    # lower than 0.05: high decay erodes pretrained backbone weights
        cos_lr=True,
        warmup_epochs=3,      # long warmup absorbs the unfreeze shock
        warmup_bias_lr=0.0,   # only affects the head bias now; 0.0 is the safe AdamW warmup floor
        batch=32,             # backbone grads cost memory -> smaller than phase 1
        freeze=None,          # do NOT pass [0] here; backbone must train
        close_mosaic=10,
        amp=True,
        cache=False,
        device=DEVICE,
        seed=SEED,
        name="oiv7_yolod9m_p2_finetune",
    )


if __name__ == "__main__":
    best = phase1_frozen_head()
    print(f"[phase1] best weights: {best}")
    phase2_finetune(best)
