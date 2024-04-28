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
import pandas as pd

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

def save_preds(
    total_ids,
    total_preds,
    output_path,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    total_ids_np = np.array(total_ids).reshape(-1, 1)
    combined_data = np.hstack((total_ids_np, total_preds.numpy()))
    sub = pd.DataFrame(combined_data)  

    sub.to_csv(output_path, index=False)
    logging.info(f'saved to {output_path}')

@torch.no_grad()
def test_step(
    model,
    input: torch.Tensor,
):
    with torch.cuda.amp.autocast():
        return model.module(input)


@torch.no_grad()
def test_epoch(
    cfg,
    model,
    loaders,
    device,
):
    model.eval()

    total_ids = []
    y_pred = torch.zeros((len(loaders['test'].dataset), model.module.num_classes))
    
    for batch_index, input in enumerate(tqdm(loaders['test'])):
        input = input.to(device)
        preds = test_step(
            model=model,
            input=input,
        )

        start_index = batch_index * loaders['test'].batch_size
        end_index = (batch_index + 1) * loaders['test'].batch_size

        total_ids.extend(getattr(loaders['test'].dataset, cfg.trainer.trainer_hyps.test_dataset_id_attribute, f"{cfg.trainer.trainer_hyps.test_dataset_id_attribute} attribute not found")[start_index:end_index])
        y_pred[start_index:end_index] = preds.detach().cpu()
        
    probs = torch.softmax(y_pred, dim=1)
    
    return total_ids, probs


@hydra.main(config_path="../configs", config_name="test")
def hydra_run(
    cfg: DictConfig,
) -> None:
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    model = instantiate(cfg.model)
    model.load_state_dict(torch.load(cfg.trainer.trainer_hyps.model_path, map_location='cpu')['model'], strict=True)
    loaders = build_loaders(cfg=cfg)

    device = f'cuda:{int(os.environ["LOCAL_RANK"])}'

    model = model.to(device=device)
    model = DDP(model, device_ids=[device])

    total_ids, probs = test_epoch(
        cfg=cfg,
        model=model,
        loaders=loaders,
        device=device,
    )

    save_preds(
        total_ids=total_ids,
        total_preds=probs,
        output_path=cfg.trainer.trainer_hyps.test_preds_output_path,
    )
 
    destroy_process_group()


if __name__ == "__main__":
    hydra_run()
