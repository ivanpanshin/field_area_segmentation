import os

import cv2
import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.distributed import destroy_process_group, init_process_group
from mining_sites.helpers import (
    build_loaders,
    build_optim,
)
import logging
from tqdm import tqdm
from typing import Dict
import numpy as np
from sklearn.metrics import f1_score
import random
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def save_model(
    model,
    optimizer,
    scheduler,
    current_epoch,
    logging_dir,
) -> None:
    if os.path.dirname(logging_dir) != '':
        os.makedirs(os.path.dirname(logging_dir), exist_ok=True)
    else:
        os.makedirs(logging_dir, exist_ok=True)

    torch.save(
        {
            "model": model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "current_epoch": current_epoch,
        },
        f'{logging_dir}/epoch_{current_epoch}.pt'
    )


def train_step(
    model,
    optimizer,
    criterion,
    scaler,
    scheduler,
    input: torch.Tensor,
    labels: torch.Tensor,
):
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        model_predictions = model(input)
        loss = criterion(
            model_predictions,
            labels,
        )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

    return {"model_predictions": model_predictions, "loss_item": loss.item()}


def train_epoch(
    model,
    optimizer,
    criterion,
    scaler,
    loaders,
    device,
    scheduler,
    ):
    model.train()

    losses = []
    for input, labels in tqdm(loaders["train"]):
        input, labels = input.to(device), labels.to(device)
        preds = train_step(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            scheduler=scheduler,
            input=input,
            labels=labels,
        )

        losses.append(preds["loss_item"])

    return {"loss_mean": np.mean(losses)}

@torch.no_grad()
def val_step(
    model,
    input: torch.Tensor,
):
    with torch.cuda.amp.autocast():
        return model.module(input)


@torch.no_grad()
def val_epoch(
    model,
    loaders,
    loader_name,
    device,
):
    model.eval()

    y_pred = torch.zeros((len(loaders[loader_name].dataset), model.module.num_classes))
    y_true = torch.zeros(len(loaders[loader_name].dataset))

    for batch_index, (input, labels) in enumerate(tqdm(loaders[loader_name])):
        input, labels = input.to(device), labels.to(device)
        preds = val_step(
            model=model,
            input=input,
        )

        start_index = batch_index * loaders[loader_name].batch_size
        end_index = (batch_index + 1) * loaders[loader_name].batch_size

        y_pred[start_index:end_index] = preds
        y_true[start_index:end_index] = labels

    metrics = {}
    probs = torch.softmax(y_pred, dim=1)
    hard_preds = torch.argmax(probs, dim=1)

    metrics['f1_score'] = f1_score(
        y_true=y_true,
        y_pred=hard_preds,
    )
    
   
    return metrics


@hydra.main(config_path="../configs", config_name="train")
def hydra_run(
    cfg: DictConfig,
) -> None:
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    model = instantiate(cfg.model)
    loaders = build_loaders(cfg=cfg)
    optim = build_optim(cfg=cfg, model=model)

    device = f'cuda:{int(os.environ["LOCAL_RANK"])}'
    scaler = torch.cuda.amp.GradScaler()

    model = model.to(device=device)
    if cfg.trainer.trainer_hyps.pretrain is not None:
        model.load_state_dict(torch.load(cfg.trainer.trainer_hyps.pretrain, map_location='cpu')['model'], strict=True)
        logging.info(f'Loaded pretrain {cfg.trainer.trainer_hyps.pretrain}')
    model = DDP(model, device_ids=[device])

    for epoch in range(cfg.trainer.trainer_hyps.num_epochs):
        loaders["train"].sampler.set_epoch(epoch)

        train_metrics = train_epoch(
            model=model,
            optimizer=optim['optimizer'],
            criterion=optim['criterion'],
            scaler=scaler,
            loaders=loaders,
            device=device,
            scheduler=optim['scheduler'],

        )

        if device == 'cuda:0':
            val_loaders_names = [key for key in loaders if key.startswith("val_")]
            val_metrics = {}
            for name in val_loaders_names:
                for k, v in val_epoch(
                    model=model,
                    loaders=loaders,
                    loader_name=name,
                    device=device,
                ).items():
                    val_metrics[f'{name}/{k}'] = v

            logging.info(f'Train metrics: {train_metrics} Val metrics: {val_metrics}')

            save_model(
                model=model,
                optimizer=optim['optimizer'],
                scheduler=optim['scheduler'],
                current_epoch=epoch,
                logging_dir=cfg.logging.logging_dir
            )

        torch.distributed.barrier()

    destroy_process_group()


if __name__ == "__main__":
    hydra_run()
