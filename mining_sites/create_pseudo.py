import pandas as pd 
import torch
from glob import glob
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import numpy as np
import logging
import os

@hydra.main(config_path="../configs", config_name="create_pseudo")
def hydra_run(
    cfg: DictConfig,
) -> None:
    ensemble_preds = None 
    paths = sorted(glob(f'{cfg.subs_root}/*.csv'))
    
    for path in tqdm(paths):
        df = pd.read_csv(path)
        if ensemble_preds is None:
            ensemble_preds = torch.tensor(df.iloc[:, 1:].values)
        else:
            ensemble_preds += df.iloc[:, 1:].values

    ensemble_preds /= len(paths)
    logging.info(f'normalized by {len(paths)}')
    ensemble_preds = torch.softmax(ensemble_preds, dim=1)
    

    total_ids_np = np.array(df.iloc[:,0]).reshape(-1, 1)
    combined_data = np.hstack((total_ids_np, ensemble_preds.numpy()))
    sub = pd.DataFrame(combined_data)  

    # Save to CSV without the index and without a header
    os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)
    sub.to_csv(cfg.output_path, index=False, header=False)
    logging.info(f'saved to {cfg.output_path}')
   

if __name__ == "__main__":
    hydra_run()
