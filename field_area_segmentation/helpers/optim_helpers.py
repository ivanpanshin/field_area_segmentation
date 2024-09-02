from hydra.utils import instantiate
import segmentation_models_pytorch as smp
import torch

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


class AggregatedLoss(torch.nn.modules.loss._Loss):
    def __init__(self, ce_weight=0.9, dice_weight=0.01, focal_weight=0.09):
        super().__init__()
        self.semantic_combo_loss = CEDiceFocalLoss()
        self.l1 = torch.nn.L1Loss(reduction='none')
        self.l2 = torch.nn.MSELoss(reduction='none')

    def forward(
        self,
        preds,
        binary_semseg_mask,
        center_mask,
        offsets,
    ):
        semantic_loss = self.semantic_combo_loss(preds['masks_semantic'], binary_semseg_mask)
        centers_loss = torch.mean(self.l2(preds['masks_centers'], center_mask) * binary_semseg_mask)

        # Debug print statement
        offsets_loss = (torch.mean(self.l1(preds['masks_offsets'], offsets) * binary_semseg_mask))

        return (semantic_loss + centers_loss + offsets_loss), semantic_loss, centers_loss, offsets_loss


class AggregatedBoundariesLoss(torch.nn.modules.loss._Loss):
    def __init__(self, ce_weight=0.9, dice_weight=0.01, focal_weight=0.09):
        super().__init__()
        self.semantic_combo_loss = CEDiceFocalLoss()
        self.l1 = torch.nn.L1Loss(reduction='none')
        self.l2 = torch.nn.MSELoss(reduction='none')

    def forward(
        self,
        preds,
        binary_semseg_mask,
        binary_boundaries_mask,
        center_mask,
        offsets,
    ):
        semantic_loss = self.semantic_combo_loss(preds['masks_semantic'], binary_semseg_mask)
        boundaries_loss = self.semantic_combo_loss(preds['masks_boundaries'], binary_boundaries_mask)
        centers_loss = torch.mean(self.l2(preds['masks_centers'], center_mask) * binary_semseg_mask)

        # Debug print statement
        offsets_loss = (torch.mean(self.l1(preds['masks_offsets'], offsets) * binary_semseg_mask))

        return (semantic_loss + centers_loss + offsets_loss + boundaries_loss), semantic_loss, boundaries_loss, centers_loss, offsets_loss


class CEDiceFocalLoss(torch.nn.modules.loss._Loss):
    def __init__(self, ce_weight=0.9, dice_weight=0.01, focal_weight=0.09):
        super().__init__()
        self.ce = torch.nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode="binary")
        self.focal = smp.losses.FocalLoss(mode="binary")
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, preds, gt):
        return (
            self.ce_weight * self.ce(preds, gt)
            + self.dice_weight * self.dice(preds, gt)
            + self.focal_weight * self.focal(preds, gt)
        )