from hydra.utils import instantiate
import torch
from pytorch_tools.losses.focal import FocalLoss
import torch.nn.functional as F

def build_optim(cfg, model):
    optimizer, scheduler, criterion = None, None, None
    if hasattr(cfg, "optimizer"):
        optimizer = instantiate(cfg.optimizer, model.parameters())
    if hasattr(cfg, "scheduler"):
        scheduler = instantiate(cfg.scheduler.scheduler, optimizer)
    if hasattr(cfg, "loss"):
        criterion = instantiate(cfg.loss)  # , pos_weight=torch.ones([1], device='cuda')*2)

    return {
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }

