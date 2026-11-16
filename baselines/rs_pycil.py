"""Run PyCIL baselines (LwF, iCaRL, PodNet, WA, FOSTER, BEEF) on the multisource
remote-sensing patch datasets used for BiCM-Prompt, without modifying PyCIL.

What is plugged into PyCIL (monkeypatching, PyCIL stays pristine):
  * dataset  : rs_<name>  -> my_cil/data_<name>/paper_seed<seed>/*.npy (same splits
               as the BiCM-Prompt runs), class order = paper order (no shuffle).
  * backbone : rs_vit     -> two ViT branches (HSI, LiDAR/SAR) with the same
               architecture as BiCM-Prompt (embed 128, depth 4, heads 4, patch 1),
               no prompts; feature = concat of the two branches' mean token (256-d).
               fmaps (for PodNet) = per-layer token maps reshaped to 11x11.
  * memory   : memory_size K (30 / 33 / 35), fixed_memory = False  -> K // seen classes,
               same rule as the BiCM-Prompt runs (2 / 3 / 5 exemplars per class at the end).
  * metrics  : OA / AA / Kappa / per-class / APD (Eq. 20) for both the CNN (fc) and
               NME classifiers after every session -> JSON.
  * speed    : the per-epoch test-set accuracy that PyCIL only logs is disabled.

Hyper-parameters: PyCIL defaults (SGD, lr 0.1, 160-250 epochs, batch 128) unless
--optimizer/--lr/--epochs-scale are given (see README section in the report).

    python baselines/rs_pycil.py --method icarl --dataset houston --seed 0
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
PYCIL = HERE / "PyCIL"
MY_CIL = HERE.parent / "my_cil"
sys.path.insert(0, str(PYCIL))

from sklearn.metrics import cohen_kappa_score  # noqa: E402

import utils.data_manager as dm_mod  # noqa: E402
import utils.inc_net as inc_net  # noqa: E402
from utils.data import iData  # noqa: E402
from utils.factory import get_model  # noqa: E402
from utils.toolkit import count_parameters  # noqa: E402
from models.base import BaseLearner  # noqa: E402
from my_cil.evaluation import calculate_forgetting_rate, convert_to_stage_stats  # noqa: E402
from my_cil.models.prompt_vit import ViT_Prompt  # noqa: E402

DATASETS = {  # init_cls, increment, memory K (paper Tables I-III, Sec. IV-B)
    "houston": dict(init_cls=6, increment=3, memory=30),
    "muufl": dict(init_cls=5, increment=2, memory=33),
    "augsburg": dict(init_cls=3, increment=2, memory=35),
}
METHODS = {  # PyCIL model_name and method-specific args (from PyCIL exps/*.json)
    "lwf": dict(model_name="lwf"),
    "icarl": dict(model_name="icarl"),
    "podnet": dict(model_name="podnet"),
    "wa": dict(model_name="wa"),
    "foster": dict(model_name="foster", beta1=0.96, beta2=0.97, oofc="ft", is_teacher_wa=False,
                   is_student_wa=False, lambda_okd=1, wa_value=1, init_epochs=200, init_lr=0.1,
                   init_weight_decay=5e-4, boosting_epochs=170, compression_epochs=130, lr=0.1,
                   batch_size=128, weight_decay=5e-4, num_workers=4, T=2),
    "beef": dict(model_name="beefiso", logits_alignment=1.7, energy_weight=0.01, is_compress=False,
                 reduce_batch_size=False, init_epochs=200, init_lr=0.1, init_weight_decay=5e-4,
                 expansion_epochs=170, fusion_epochs=60, lr=0.1, batch_size=128, weight_decay=5e-4,
                 num_workers=4, T=2),
}
# module-level hyper-parameters of the methods that do not read them from args
EPOCH_KEYS = ("init_epoch", "epochs", "ft_epochs")
LR_KEYS = ("init_lr", "lrate", "ft_lrate")
MILESTONE_KEYS = ("init_milestones", "milestones")
ARG_EPOCH_KEYS = ("init_epochs", "boosting_epochs", "compression_epochs", "expansion_epochs", "fusion_epochs")
ARG_LR_KEYS = ("init_lr", "lr")


# ---------------------------------------------------------------- dataset
def make_idata(name: str, seed: int):
    data_dir = MY_CIL / f"data_{name}" / f"paper_seed{seed}"

    class RSData(iData):
        use_path = False
        train_trsf, test_trsf, common_trsf = [], [], []

        def download_data(self):
            load = lambda split: (np.load(data_dir / f"{split}_patches.npy").astype(np.float32),
                                  np.load(data_dir / f"{split}_labels.npy").astype(np.int64) - 1)
            self.train_data, self.train_targets = load("train")
            self.test_data, self.test_targets = load("test")
            self.class_order = list(range(int(self.train_targets.max()) + 1))

    meta = json.loads((data_dir / "meta.json").read_text())
    return RSData, meta["hsi_channels"]


def patched_getitem(self, idx):
    image = torch.from_numpy(np.ascontiguousarray(self.images[idx])).permute(2, 0, 1)  # HWC -> CHW
    if self.aug == 1:
        return idx, image, self.labels[idx]
    return (idx, *[image for _ in range(self.aug)], self.labels[idx])


# ---------------------------------------------------------------- backbone
class TwoBranchViT(nn.Module):
    """Same ViT branches as BiCM-Prompt (ViT_Prompt without prompt pool)."""

    def __init__(self, hsi_channels: int, aux_channels: int, embed_dim=128, depth=4, heads=4):
        super().__init__()
        self.hsi_channels = hsi_channels
        make = lambda c: ViT_Prompt(image_size=11, patch_size=1, in_channels=c, embed_dim=embed_dim, num_classes=0,
                                    depth=depth, num_heads=heads, drop_rate=0.1, prompt_pool=False)
        self.branches = nn.ModuleList([make(hsi_channels), make(aux_channels)])
        self.out_dim = 2 * embed_dim

    def _branch(self, vit, x):
        tokens = vit.patch_embed(x)
        tokens = torch.cat((vit.cls_token.expand(x.shape[0], -1, -1), tokens), dim=1)
        tokens = vit.pos_drop(tokens + vit.pos_embed[:, : tokens.shape[1]])
        fmaps = []
        for layer in vit.layers:
            tokens = layer(tokens)
            b, n, c = tokens.shape
            fmaps.append(tokens[:, 1:].transpose(1, 2).reshape(b, c, 11, 11))
        feature = vit.norm(tokens)[:, 1:].mean(dim=1)
        return feature, fmaps

    def forward(self, x):
        f1, m1 = self._branch(self.branches[0], x[:, : self.hsi_channels])
        f2, m2 = self._branch(self.branches[1], x[:, self.hsi_channels:])
        return {"fmaps": [torch.cat([a, b], dim=1) for a, b in zip(m1, m2)],
                "features": torch.cat([f1, f2], dim=1)}


# ---------------------------------------------------------------- hyper-parameters
class OptimShim:
    """Stands in for `torch.optim` inside a PyCIL model module to swap SGD for Adam."""

    def __init__(self, lr_scale: float):
        self.lr_scale = lr_scale
        self.lr_scheduler = torch.optim.lr_scheduler

    def SGD(self, params, lr, momentum=0.9, weight_decay=0.0, **_):  # noqa: N802
        params = list(params)
        if params and isinstance(params[0], dict):  # param groups with their own lr (PodNet)
            params = [{**g, "lr": g["lr"] * self.lr_scale} if "lr" in g else g for g in params]
        return torch.optim.Adam(params, lr=lr * self.lr_scale, weight_decay=weight_decay)

    def __getattr__(self, item):
        return getattr(torch.optim, item)


def apply_overrides(module, args: dict, opts) -> dict:
    changed = {}
    scale = opts.epochs_scale
    for key in EPOCH_KEYS:
        if hasattr(module, key) and scale != 1:
            setattr(module, key, max(1, round(getattr(module, key) * scale)))
            changed[key] = getattr(module, key)
    for key in MILESTONE_KEYS:
        if hasattr(module, key) and scale != 1:
            setattr(module, key, [max(1, round(m * scale)) for m in getattr(module, key)])
            changed[key] = getattr(module, key)
    for key in ARG_EPOCH_KEYS:
        if key in args and scale != 1:
            args[key] = max(1, round(args[key] * scale))
            changed[key] = args[key]
    if opts.batch_size:
        if hasattr(module, "batch_size"):
            module.batch_size = opts.batch_size
        if "batch_size" in args:
            args["batch_size"] = opts.batch_size
        changed["batch_size"] = opts.batch_size
    if hasattr(module, "num_workers"):
        module.num_workers = opts.num_workers
    if opts.optimizer == "adam":
        module.optim = OptimShim(opts.adam_lr / 0.1)  # PyCIL SGD lr 0.1 -> Adam lr adam_lr
        changed["optimizer"] = f"Adam (lr = PyCIL lr x {opts.adam_lr / 0.1:g})"
    for key in ("init_lr", "lrate", "ft_lrate"):
        if hasattr(module, key):
            changed.setdefault(key, getattr(module, key))
    return changed


# ---------------------------------------------------------------- metrics
def metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    classes = np.unique(true)
    per_class = {int(c): float((pred[true == c] == c).mean()) for c in classes}
    return {"OA": float((pred == true).mean() * 100), "AA": float(np.mean(list(per_class.values())) * 100),
            "Kappa": float(cohen_kappa_score(true, pred) * 100), "per_class_acc": per_class}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="sgd",
                        help="sgd = PyCIL default; adam = replace every SGD by Adam")
    parser.add_argument("--adam-lr", type=float, default=1e-3, help="Adam lr used where PyCIL uses 0.1")
    parser.add_argument("--epochs-scale", type=float, default=1.0, help="multiply all PyCIL epoch counts")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, help="override PyCIL batch size (paper: 32)")
    parser.add_argument("--output", type=Path)
    opts = parser.parse_args()

    ds = DATASETS[opts.dataset]
    idata_cls, hsi_channels = make_idata(opts.dataset, opts.seed)
    sample = np.load(MY_CIL / f"data_{opts.dataset}" / f"paper_seed{opts.seed}" / "train_patches.npy", mmap_mode="r")
    aux_channels = sample.shape[-1] - hsi_channels

    # plug into PyCIL
    dm_mod._get_idata = lambda name: idata_cls()
    dm_mod.DummyDataset.__getitem__ = patched_getitem
    original_get_convnet = inc_net.get_convnet
    inc_net.get_convnet = lambda args, pretrained=False: (
        TwoBranchViT(hsi_channels, aux_channels) if args["convnet_type"] == "rs_vit" else original_get_convnet(args, pretrained))
    BaseLearner._compute_accuracy = lambda self, model, loader: -1.0  # only logged by PyCIL; skip (slow on big test sets)
    _orig_evaluate = BaseLearner._evaluate

    def _evaluate_top5_alias(self, y_pred, y_true):  # FOSTER logs ret["top5"]; we run with topk=1
        ret = _orig_evaluate(self, y_pred, y_true)
        ret.setdefault("top5", ret["top{}".format(self.topk)])
        return ret
    BaseLearner._evaluate = _evaluate_top5_alias

    args = {"prefix": "rs", "dataset": f"rs_{opts.dataset}", "memory_size": ds["memory"], "memory_per_class": 20,
            "fixed_memory": False, "shuffle": False, "init_cls": ds["init_cls"], "increment": ds["increment"],
            "convnet_type": "rs_vit", "device": [torch.device("cuda:0" if torch.cuda.is_available() else "cpu")],
            "seed": opts.seed, **copy.deepcopy(METHODS[opts.method])}
    module_name = {"beefiso": "beef_iso"}.get(args["model_name"], args["model_name"])
    module = importlib.import_module(f"models.{module_name}")
    changed = apply_overrides(module, args, opts)
    if "num_workers" in args:
        args["num_workers"] = opts.num_workers

    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    logging.info("args %s | overrides %s", {k: str(v) for k, v in args.items()}, changed)

    data_manager = dm_mod.DataManager(args["dataset"], args["shuffle"], args["seed"], args["init_cls"], args["increment"])
    model = get_model(args["model_name"], args)
    model.topk = 1  # PyCIL default top-5 breaks with < 5 classes (Augsburg base has 3)

    sessions, total_list = [], []
    correct = {"cnn": [], "nme": []}
    start_all = time.time()
    for task in range(data_manager.nb_tasks):
        start = time.time()
        model.incremental_train(data_manager)
        n_params = count_parameters(model._network, True)
        y_pred, y_true = model._eval_cnn(model.test_loader)
        preds = {"cnn": y_pred[:, 0]}
        if getattr(model, "_class_means", None) is not None and hasattr(model, "_eval_nme"):
            try:
                nme_pred, _ = model._eval_nme(model.test_loader, model._class_means)
                preds["nme"] = nme_pred[:, 0]
            except Exception as exc:  # some methods keep no usable class means
                logging.info("NME unavailable: %s", exc)
        model.after_task()

        classes = np.unique(y_true).tolist()
        total_list.append({c: int((y_true == c).sum()) for c in classes})
        result = {"session": task, "seen_classes": len(classes), "trainable_params_M": round(n_params / 1e6, 3),
                  "seconds": round(time.time() - start, 1)}
        for name, pred in preds.items():
            result[name] = metrics(pred, y_true)
            correct[name].append({c: int(((pred == c) & (y_true == c)).sum()) for c in classes})
            if task > 0 and len(correct[name]) == task + 1:
                result[name]["APD"] = calculate_forgetting_rate(convert_to_stage_stats(correct[name], total_list))[-1]
        result.update({k: result["cnn"][k] for k in ("OA", "AA", "Kappa")})
        if task > 0:
            result["APD"] = result["cnn"]["APD"]
        sessions.append(result)
        logging.info("session %d: OA %.2f AA %.2f Kappa %.2f%s | NME OA %s | params %.3fM | %.0fs", task,
                     result["OA"], result["AA"], result["Kappa"], f" APD {result['APD']:.4f}" if task else "",
                     f"{result['nme']['OA']:.2f}" if "nme" in result else "-", n_params / 1e6, result["seconds"])

    summary = {"method": opts.method, "dataset": opts.dataset, "seed": opts.seed,
               "options": {k: str(v) for k, v in vars(opts).items()}, "overrides": {k: str(v) for k, v in changed.items()},
               "sessions": sessions,
               "average": {k: float(np.mean([s[k] for s in sessions])) for k in ("OA", "AA", "Kappa")},
               "total_seconds": round(time.time() - start_all, 1)}
    tag = "" if opts.optimizer == "sgd" and opts.epochs_scale == 1 else f"_{opts.optimizer}_e{opts.epochs_scale:g}"
    if opts.batch_size:
        tag += f"_b{opts.batch_size}"
    output = opts.output or MY_CIL / "results" / "baselines" / f"{opts.method}{tag}_{opts.dataset}_seed{opts.seed}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))
    logging.info("average OA %.2f AA %.2f Kappa %.2f -> %s", summary["average"]["OA"], summary["average"]["AA"],
                 summary["average"]["Kappa"], output)


if __name__ == "__main__":
    os.chdir(PYCIL)  # PyCIL writes logs/ relative to cwd
    main()
