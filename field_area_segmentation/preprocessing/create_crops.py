import hydra
from omegaconf import DictConfig
import albumentations as A
from tqdm import tqdm
import tifffile
import numpy as np
import pandas as pd
import random
import torch
import os
import albumentations.pytorch as AT
import multiprocessing
import time
import cv2
from glob import glob

def create_labels(
    g,
    sigma,
    instance_mask,
    y_coord,
    x_coord,
):

    binary_semseg_mask = np.zeros((instance_mask.shape[1], instance_mask.shape[2]), dtype=np.float16)
    center_mask = np.zeros((instance_mask.shape[1], instance_mask.shape[2]), dtype=np.float32)
    offsets = np.zeros((2, instance_mask.shape[1], instance_mask.shape[2]), dtype=np.float16)

    for mask_object in instance_mask:
        if np.sum(mask_object) == 0:
            continue

        binary_semseg_mask += mask_object

        mask_index = np.where(mask_object == 1)
        center_y, center_x = np.mean(mask_index[0]), np.mean(mask_index[1])
        y, x = int(center_y), int(center_x)
        ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
        br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

        c, d = max(0, -ul[0]), min(br[0], mask_object.shape[1]) - ul[0]
        a, b = max(0, -ul[1]), min(br[1], mask_object.shape[0]) - ul[1]

        cc, dd = max(0, ul[0]), min(br[0], mask_object.shape[1])
        aa, bb = max(0, ul[1]), min(br[1], mask_object.shape[0])
        center_mask[aa:bb, cc:dd] = np.maximum(
            center_mask[aa:bb, cc:dd], g[a:b, c:d])

        offset_y_index = (np.zeros_like(mask_index[0]), mask_index[0], mask_index[1])
        offset_x_index = (np.ones_like(mask_index[0]), mask_index[0], mask_index[1])
        offsets[offset_y_index] = center_y - y_coord[mask_index]
        offsets[offset_x_index] = center_x - x_coord[mask_index]

    assert np.min(center_mask) >=0 and np.max(center_mask) <= 1
    center_mask = (center_mask * 255.0).astype(np.uint8)

    return binary_semseg_mask, center_mask, offsets


def process_val_crop(
    cfg,
    im_path,
    mask_path,
    augs,
    g,
):
    y_coord = np.ones((cfg.height, cfg.width), dtype=np.float32)
    x_coord = np.ones((cfg.height, cfg.width), dtype=np.float32)
    y_coord = np.cumsum(y_coord, axis=0) - 1
    x_coord = np.cumsum(x_coord, axis=1) - 1

    image = tifffile.memmap(im_path, mode='r')
    instance_mask = tifffile.memmap(mask_path, mode='r')

    image = image.copy()
    instance_mask = instance_mask.copy()

    assert image.min() >= 0 and image.max() <= 255
    assert instance_mask.min() >= 0 and instance_mask.max() <= 1

    transformed = augs(image=image, masks=instance_mask)
    image = transformed['image']
    instance_mask = torch.stack([torch.tensor(_) for _ in transformed['masks']], dim=0)

    binary_semseg_mask, center_mask, offsets = create_labels(
        g=g,
        sigma=cfg.sigma,
        instance_mask=instance_mask.numpy(),
        y_coord=y_coord,
        x_coord=x_coord,
    )

    instance_segmentation_mask = np.zeros((instance_mask.shape[1], instance_mask.shape[2]), dtype=np.float16)

    for object_index, object_mask in enumerate(instance_mask):
        instance_segmentation_mask += (object_mask.numpy() * (object_index + 1))

    boundaries = np.where(binary_semseg_mask > 1, 1, 0).astype(bool)

    instance_segmentation_mask = np.where(boundaries == 1, 0, instance_segmentation_mask)
    binary_semseg_mask = np.clip(binary_semseg_mask, 0, 1).astype(bool)
    binary_semseg_mask_no_boundaries = np.where(boundaries == 1, 0, binary_semseg_mask).astype(bool)

    cv2.imwrite(f'{cfg.output_dir}/images/{im_path.split("/")[-1].split(".")[0]}.png',
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    np.save(f'{cfg.output_dir}/instance_masks/{im_path.split("/")[-1].split(".")[0]}.npy',
            instance_segmentation_mask)
    np.save(f'{cfg.output_dir}/semantic_masks/{im_path.split("/")[-1].split(".")[0]}.npy',
            binary_semseg_mask)
    np.save(f'{cfg.output_dir}/boundaries/{im_path.split("/")[-1].split(".")[0]}.npy',
            boundaries)
    np.save(f'{cfg.output_dir}/semantic_masks_no_boundaries/{im_path.split("/")[-1].split(".")[0]}.npy',
            binary_semseg_mask_no_boundaries)
    np.save(f'{cfg.output_dir}/centers_masks/{im_path.split("/")[-1].split(".")[0]}.npy', center_mask)
    np.save(f'{cfg.output_dir}/offsets_masks/{im_path.split("/")[-1].split(".")[0]}.npy', offsets)


def process_test_crop(
    cfg,
    im_path,
    augs,
):
    image = tifffile.memmap(im_path, mode='r')
    image = image.copy()
    assert image.min() >= 0 and image.max() <= 255

    transformed = augs(image=image)
    image = transformed['image']

    cv2.imwrite(f'{cfg.output_dir}/images/{im_path.split("/")[-1].split(".")[0]}.png',
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR))



def process_train_crop(
    cfg,
    im_paths,
    mask_paths,
    augs,
    g,
    crop_index,
):
    idx = random.randint(0, len(im_paths) - 1)

    y_coord = np.ones((cfg.height, cfg.width), dtype=np.float32)
    x_coord = np.ones((cfg.height, cfg.width), dtype=np.float32)
    y_coord = np.cumsum(y_coord, axis=0) - 1
    x_coord = np.cumsum(x_coord, axis=1) - 1

    im_path = im_paths[idx]
    mask_path = mask_paths[idx]

    image = tifffile.imread(im_path)
    instance_mask = tifffile.imread(mask_path)


    height, width = image.shape[:2]
    crop_height_start = random.randint(0, height - cfg.crop_height)
    crop_width_start = random.randint(0, width - cfg.crop_width)

    image = image[crop_height_start:crop_height_start + cfg.crop_height,
            crop_width_start:crop_width_start + cfg.crop_width, :].copy()
    instance_mask = instance_mask[:, crop_height_start:crop_height_start + cfg.crop_height,
                    crop_width_start:crop_width_start + cfg.crop_width].copy()

    assert image.shape == (cfg.crop_height, cfg.crop_width, 3) and instance_mask.shape[1:] == (
    cfg.crop_height, cfg.crop_width)

    assert image.min() >= 0 and image.max() <= 255
    assert instance_mask.min() >= 0 and instance_mask.max() <= 1

    transformed = augs(image=image, masks=instance_mask)
    image = transformed['image']
    instance_mask = torch.stack([torch.tensor(_) for _ in transformed['masks']], dim=0)

    binary_semseg_mask, center_mask, offsets = create_labels(
        g=g,
        sigma=cfg.sigma,
        instance_mask=instance_mask.numpy(),
        y_coord=y_coord,
        x_coord=x_coord,
    )

    instance_segmentation_mask = np.zeros((instance_mask.shape[1], instance_mask.shape[2]), dtype=np.float16)

    for object_index, object_mask in enumerate(instance_mask):
        instance_segmentation_mask += (object_mask.numpy() * (object_index + 1))

    boundaries = np.where(binary_semseg_mask > 1, 1, 0).astype(bool)

    instance_segmentation_mask = np.where(boundaries == 1, 0, instance_segmentation_mask)
    binary_semseg_mask = np.clip(binary_semseg_mask, 0, 1).astype(bool)
    binary_semseg_mask_no_boundaries = np.where(boundaries == 1, 0, binary_semseg_mask).astype(bool)

    os.makedirs(f'{cfg.output_dir}/images/{im_path.split("/")[-1].split(".")[0]}', exist_ok=True)
    os.makedirs(f'{cfg.output_dir}/instance_masks/{im_path.split("/")[-1].split(".")[0]}', exist_ok=True)
    os.makedirs(f'{cfg.output_dir}/semantic_masks/{im_path.split("/")[-1].split(".")[0]}', exist_ok=True)
    os.makedirs(f'{cfg.output_dir}/boundaries/{im_path.split("/")[-1].split(".")[0]}', exist_ok=True)
    os.makedirs(f'{cfg.output_dir}/semantic_masks_no_boundaries/{im_path.split("/")[-1].split(".")[0]}', exist_ok=True)
    os.makedirs(f'{cfg.output_dir}/centers_masks/{im_path.split("/")[-1].split(".")[0]}', exist_ok=True)
    os.makedirs(f'{cfg.output_dir}/offsets_masks/{im_path.split("/")[-1].split(".")[0]}', exist_ok=True)

    cv2.imwrite(f'{cfg.output_dir}/images/{im_path.split("/")[-1].split(".")[0]}/{crop_index}.png',
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    np.save(f'{cfg.output_dir}/instance_masks/{im_path.split("/")[-1].split(".")[0]}/{crop_index}.npy',
            instance_segmentation_mask)
    np.save(f'{cfg.output_dir}/semantic_masks/{im_path.split("/")[-1].split(".")[0]}/{crop_index}.npy',
            binary_semseg_mask)
    np.save(f'{cfg.output_dir}/boundaries/{im_path.split("/")[-1].split(".")[0]}/{crop_index}.npy',
            boundaries)
    np.save(f'{cfg.output_dir}/semantic_masks_no_boundaries/{im_path.split("/")[-1].split(".")[0]}/{crop_index}.npy',
            binary_semseg_mask_no_boundaries)
    np.save(f'{cfg.output_dir}/centers_masks/{im_path.split("/")[-1].split(".")[0]}/{crop_index}.npy', center_mask)
    np.save(f'{cfg.output_dir}/offsets_masks/{im_path.split("/")[-1].split(".")[0]}/{crop_index}.npy', offsets)


def process_dataset(cfg):
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(f'{cfg.output_dir}/images', exist_ok=True)
    os.makedirs(f'{cfg.output_dir}/instance_masks', exist_ok=True)
    os.makedirs(f'{cfg.output_dir}/semantic_masks', exist_ok=True)
    os.makedirs(f'{cfg.output_dir}/boundaries', exist_ok=True)
    os.makedirs(f'{cfg.output_dir}/semantic_masks_no_boundaries', exist_ok=True)
    os.makedirs(f'{cfg.output_dir}/centers_masks', exist_ok=True)
    os.makedirs(f'{cfg.output_dir}/offsets_masks', exist_ok=True)

    train_augs = A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=(-0.05, 0.05), contrast_limit=(-0.05, 0.05), p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
        ], p=0.7),

        A.D4(p=0.7),
        A.Resize(cfg.height, cfg.width, interpolation=2),

    ])

    val_augs = A.Compose(
        [
            A.Resize(height=cfg.height, width=cfg.width, interpolation=2),
        ]
    )

    if cfg.split != 'test':
        csv = pd.read_csv(cfg.ann_path)

        im_paths, mask_paths = [], []
        for file_name in csv.filename.values:
            im_paths.append(f'{cfg.dataset_root}/images/{file_name}.tif')
            mask_paths.append(f'{cfg.dataset_root}/masks/{file_name}.tif')

        size = 6 * cfg.sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * cfg.sigma + 1, 3 * cfg.sigma + 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * cfg.sigma ** 2))


    else:
        im_paths = sorted(glob(f'{cfg.dataset_root}/*.tif'))


    if cfg.split == 'train':
        params = []
        for crop_index in tqdm(range(cfg.num_crops)):
            params.append(
                (
                    cfg,
                    im_paths,
                    mask_paths,
                    train_augs,
                    g,
                    crop_index,
                )
            )
        t1 = time.time()
        with multiprocessing.Pool(cfg.num_procs) as pool:
            pool.starmap(process_train_crop, params)

        print(f'took {time.time() - t1}')


    elif cfg.split == 'val':
        print(len(im_paths), len(mask_paths))

        for im_path, mask_path in tqdm(zip(im_paths, mask_paths)):
            print(f'process {im_path} {mask_path}')
            process_val_crop(
                cfg,
                im_path,
                mask_path,
                val_augs,
                g,
            )
    elif cfg.split == 'test':
        for im_path in tqdm(im_paths):
            process_test_crop(
                cfg,
                im_path,
                val_augs,
            )

    else:
        raise
@hydra.main(config_path="../../configs/preprocessing", config_name="create_test_crops")
def hydra_run(
        cfg: DictConfig,
) -> None:
    process_dataset(cfg)


if __name__ == "__main__":
    hydra_run()