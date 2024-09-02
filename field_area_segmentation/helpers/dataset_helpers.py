from torch.utils.data import Dataset
import cv2
import pandas as pd
import numpy as np
import tifffile
import random
import torch
from tqdm import tqdm
from time import time
import os
from glob import glob
import albumentations.pytorch as AT
import logging

class SemSegDatasetPreprocessed(Dataset):
    def __init__(
        self,
        dataset_root,
        split,
        multiplier=1,
        transform=None,
    ):
        super().__init__()

        if split == 'train':
            self.im_paths, self.semantic_paths, self.boundaries_paths, self.semantic_no_boundaries_paths, self.instance_paths, self.centers_paths, self.offsets_paths = [], [], [], [], [], [], []
            for folder_name in sorted(os.listdir(f'{dataset_root}/images')):
                im_paths = sorted(glob(f'{dataset_root}/images/{folder_name}/*.png'))
                semantic_paths = sorted(glob(f'{dataset_root}/semantic_masks/{folder_name}/*.npy'))
                boundaries_paths = sorted(glob(f'{dataset_root}/boundaries/{folder_name}/*.npy'))
                semantic_no_boundaries_paths = sorted(glob(f'{dataset_root}/semantic_masks_no_boundaries/{folder_name}/*.npy'))
                instance_paths = sorted(glob(f'{dataset_root}/instance_masks/{folder_name}/*.npy'))
                centers_paths = sorted(glob(f'{dataset_root}/centers_masks/{folder_name}/*.npy'))
                offsets_paths = sorted(glob(f'{dataset_root}/offsets_masks/{folder_name}/*.npy'))

                print(len(im_paths), len(semantic_paths), len(boundaries_paths), len(instance_paths), len(centers_paths), len(offsets_paths))
                assert len(im_paths) == len(semantic_paths) == len(boundaries_paths) == len(semantic_no_boundaries_paths) == len(instance_paths) == len(centers_paths) == len(offsets_paths)

                self.im_paths.extend(im_paths)
                self.semantic_paths.extend(semantic_paths)
                self.boundaries_paths.extend(boundaries_paths)
                self.semantic_no_boundaries_paths.extend(semantic_no_boundaries_paths)
                self.instance_paths.extend(instance_paths)
                self.centers_paths.extend(centers_paths)
                self.offsets_paths.extend(offsets_paths)

        elif split == 'val':
            self.im_paths = sorted(glob(f'{dataset_root}/images/*.png'))
            self.semantic_paths = sorted(glob(f'{dataset_root}/semantic_masks/*.npy'))
            self.boundaries_paths = sorted(glob(f'{dataset_root}/boundaries/*.npy'))
            self.semantic_no_boundaries_paths = sorted(glob(f'{dataset_root}/semantic_masks_no_boundaries/*.npy'))
            self.instance_paths = sorted(glob(f'{dataset_root}/instance_masks/*.npy'))
            self.centers_paths = sorted(glob(f'{dataset_root}/centers_masks/*.npy'))
            self.offsets_paths = sorted(glob(f'{dataset_root}/offsets_masks/*.npy'))
            assert len(self.im_paths) == len(self.semantic_paths) == len(self.boundaries_paths) == len(self.semantic_no_boundaries_paths) == len(self.instance_paths) == len(self.centers_paths) == len(self.offsets_paths)

            l = len(self.im_paths)
            if l % 2 == 1:
                l -= 1
                self.im_paths = self.im_paths[:l]
                self.semantic_paths = self.semantic_paths[:l]
                self.boundaries_paths = self.boundaries_paths[:l]
                self.semantic_no_boundaries_paths = self.semantic_no_boundaries_paths[:l]
                self.instance_paths = self.instance_paths[:l]
                self.centers_paths = self.centers_paths[:l]
                self.offsets_paths = self.offsets_paths[:l]

                assert len(self.im_paths) == len(self.semantic_paths) == len(self.boundaries_paths) == len(
                    self.semantic_no_boundaries_paths) == len(self.instance_paths) == len(self.centers_paths) == len(
                    self.offsets_paths)

        else:
            self.im_paths = sorted(glob(f'{dataset_root}/images/*.png'))
            self.im_ids = [_.split('/')[-1].split('.')[0] for _ in self.im_paths]


        self.split = split
        self.transform = transform
        self.base_seed = 100500
        logging.info(f'dataset len: {len(self.im_paths)}')

    def set_epoch(self, epoch):
        epoch_seed = self.base_seed + epoch
        random.seed(epoch_seed)

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, idx):
        if self.split != 'test':
            im_path = self.im_paths[idx]
            semantic_path = self.semantic_paths[idx]
            boundaries_path = self.boundaries_paths[idx]
            semantic_no_boundaries_path = self.semantic_no_boundaries_paths[idx]
            instance_path = self.instance_paths[idx]
            centers_path = self.centers_paths[idx]
            offsets_path = self.offsets_paths[idx]

            image = cv2.imread(im_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            semantic_mask = np.load(semantic_path)
            boundary_mask = np.load(boundaries_path)
            semantic_no_boundary_mask = np.load(semantic_no_boundaries_path)
            instance_mask = np.load(instance_path)
            center_mask = np.load(centers_path)
            offset_mask = np.load(offsets_path)

            assert image.shape[:2] == semantic_mask.shape == boundary_mask.shape == instance_mask.shape == center_mask.shape == offset_mask.shape[1:]

            assert np.min(image) >= 0 and np.max(image) <= 255
            assert np.min(semantic_mask) >= 0 and np.max(semantic_mask) <= 1
            assert np.min(boundary_mask) >= 0 and np.max(boundary_mask) <= 1
            assert np.min(semantic_no_boundary_mask) >= 0 and np.max(semantic_no_boundary_mask) <= 1
            assert np.min(center_mask) >= 0 and np.max(center_mask) <= 255
            assert np.min(offset_mask[0, :, :]) >= -image.shape[0] and np.max(offset_mask[0, :, :]) <= image.shape[0]
            assert np.min(offset_mask[1, :, :]) >= -image.shape[1] and np.max(offset_mask[1, :, :]) <= image.shape[1]

            return (
                AT.ToTensorV2()(image=image)['image'] / 255.0,
                torch.tensor(instance_mask),
                torch.tensor(semantic_mask).unsqueeze(0).float(),
                torch.tensor(center_mask).unsqueeze(0).float(),
                torch.tensor(offset_mask),
            )
        else:
            im_path = self.im_paths[idx]
            image = cv2.imread(im_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            assert np.min(image) >= 0 and np.max(image) <= 255

            return AT.ToTensorV2()(image=image)['image'] / 255.0