import hydra
from omegaconf import DictConfig
import os
from tqdm import tqdm
from glob import glob
import cv2
import numpy as np
import multiprocessing

def process_folder(cfg, folder):
    images_paths = glob(f'{cfg.dataset_root}/{folder}/*/*.tif')
    images = [cv2.imread(_) for _ in images_paths]
    images = np.array([cv2.cvtColor(_, cv2.COLOR_BGR2RGB) for _ in images])
    try:
        median_image = np.median(images, axis=0).astype(np.uint8)

        cv2.imwrite(f'{cfg.output_root}/{folder}.png', cv2.cvtColor(median_image, cv2.COLOR_RGB2BGR))
    except:
        pass

def process_dataset(cfg):
    os.makedirs(cfg.output_root, exist_ok=True)

    folders = os.listdir(cfg.dataset_root)
    params = [(cfg, folder) for folder in folders]

    with multiprocessing.Pool(cfg.num_procs) as pool:
        pool.starmap(process_folder, params)
    


@hydra.main(config_path="../../configs/preprocessing", config_name="calculate_median_image")
def hydra_run(
    cfg: DictConfig,
) -> None:
    process_dataset(cfg)


if __name__ == "__main__":
    hydra_run()