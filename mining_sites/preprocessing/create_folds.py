import hydra
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import logging
import os

def process_dataset(cfg):
    os.makedirs(cfg.output_dir, exist_ok=True)

    ann = pd.read_csv(cfg.input_path)

    skf = StratifiedKFold(n_splits=cfg.num_folds)
    for i, (_, test_index) in enumerate(skf.split(ann.iloc[:, 0], ann.iloc[:, 1])):
        
        val_fold = ann.iloc[test_index].reset_index(drop=True)
        val_fold.to_csv(f'{cfg.output_dir}/fold_{i}.csv', index=False)

    logging.info(f'Folds saved to {cfg.output_dir}')

@hydra.main(config_path="../../configs/preprocessing", config_name="create_folds")
def hydra_run(
    cfg: DictConfig,
) -> None:
    process_dataset(cfg)


if __name__ == "__main__":
    hydra_run()