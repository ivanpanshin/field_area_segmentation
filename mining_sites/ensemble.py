import pandas as pd 
import torch
from glob import glob
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import logging
import numpy as np
import os

@hydra.main(config_path="../configs", config_name="ensemble")
def hydra_run(
    cfg: DictConfig,
) -> None:
    
    os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)
    
    paths = sorted(glob(f'{cfg.subs_root}/*.csv'))
    ensemble_preds = None 
    for path in tqdm(paths):
        df = pd.read_csv(path)
        if ensemble_preds is None:
            ensemble_preds = torch.tensor(df.iloc[:, 1:].values)
        else:
            ensemble_preds += df.iloc[:, 1:].values

    ensemble_preds /= len(paths)
    logging.info(f'normalized by {len(paths)}')
    ensemble_preds = torch.argmax(torch.softmax(ensemble_preds, dim=1), dim=1)

    final_sub = pd.DataFrame({
        0: df.iloc[:,0],
        1: ensemble_preds.numpy()
    })

    # Save to CSV without the index and without a header
    final_sub.to_csv(cfg.output_path, index=False, header=False)
    logging.info(f'saved to {cfg.output_path}')

if __name__ == "__main__":
    hydra_run()
