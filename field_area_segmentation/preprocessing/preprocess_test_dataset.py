import hydra
from omegaconf import DictConfig
import os
import json
from glob import glob
import tifffile
import cv2
import numpy as np
from tqdm import tqdm


def create_semseg_mask(
    segmentation,
    image_shape,
    fill_value
):
    mask = np.zeros(image_shape, dtype=np.uint8)
    points = np.array(segmentation, dtype=np.int32).reshape(-1, 2)
    cv2.fillPoly(mask, [points], fill_value)
    return mask


def process_dataset(cfg):
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(f'{cfg.output_dir}/images', exist_ok=True)

    images_paths = sorted(glob(f'{cfg.input_dir}/images/*.tif'))

    for image_path in tqdm(images_paths):
        image_path_base = image_path.split('/')[-1]

        im = tifffile.imread(image_path)[:, :, [3, 2, 1]]
        im[np.isnan(im)] = 0
        im = (im - im.min()) / (im.max() - im.min() + 1e-5)
        assert np.min(im) >= 0 and np.max(im) <= 1
        im = (im.copy() * 255).astype(np.uint8)
        assert np.min(im) >= 0 and np.max(im) <= 255

        tifffile.imwrite(f'{cfg.output_dir}/images/{image_path_base.split(".")[0]}.tif', im)


@hydra.main(config_path="../../configs/preprocessing", config_name="preprocess_test_dataset")
def hydra_run(
        cfg: DictConfig,
) -> None:
    process_dataset(cfg)


if __name__ == "__main__":
    hydra_run()