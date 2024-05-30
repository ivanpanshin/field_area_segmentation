from torch.utils.data import Dataset
import pandas as pd
import tifffile
import torch
import logging
from glob import glob
import cv2
from tqdm import tqdm


class DatasetOriginal(Dataset):
    def __init__(
        self,
        dataset_root,
        split,
        multiplier=1,
        ann_path=None,
        transform=None,
        max_norm=False,
    ):
        super().__init__()
        self.max_norm = max_norm

        paths = sorted(glob(f'{dataset_root}/*.tif'))
        if split == 'train':
            csv = pd.read_csv(ann_path)
            csv = csv[csv.split == 'train'].reset_index(drop=True)
        elif split == 'val':
            csv = pd.read_csv(ann_path)
            csv = csv[csv.split == 'valid'].reset_index(drop=True)

        if split != 'test':
            csv = csv[csv.image_id.isin([_.split('/')[-1] for _ in paths])].reset_index(drop=True)
            self.paths, self.labels = [], []

            for index in range(csv.shape[0]):
                self.paths.append(f'{dataset_root}/{csv.iloc[index, 0]}')
                self.labels.append(int(csv.iloc[index, 1]))

            self.paths *= multiplier
            self.labels *= multiplier
            self.ids = [_.split('/')[-1] for _ in self.paths]

            assert len(self.paths) == len(self.labels)
            
        else:
            self.paths = paths
            self.ids = [_.split('/')[-1] for _ in self.paths]

        logging.info(f'Dataset len: {len(self.paths)}')
        
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        if self.split != 'test':
            label = self.labels[idx]

        image = tifffile.memmap(path, mode='r')[:, :, [3,2,1]]
        assert image.min() >= 0 and image.max() <= 2, f'min: {image.min()} max: {image.max()}'
        if self.transform:
            image = self.transform(image=image)['image']
            if self.max_norm:
                image = image / (image.max() + 1e-5)

        if self.split != 'test':
            return image, torch.tensor(label)
    
        return image
    

class DatasetSSL4EO(Dataset):
    def __init__(
        self,
        dataset_root,
        transform=None,
    ):
        super().__init__()

        self.paths = glob(f'{dataset_root}/*.png')
        self.ids = [_.split('/')[-1] for _ in self.paths]
        logging.info(f'Num of paths in test dataset: {len(self.paths)}')
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
       
        image = self.transform(image=image)['image']
        image = image / 255.0
        assert image.shape == (3, 512, 512), f"shape: {image.shape}"

        return image
    


class DatasetSSL4EOPseudo(Dataset):
    def __init__(
        self,
        dataset_root,
        ann_path,
        transform=None,
    ):
        super().__init__()

        self.dataset_root = dataset_root
        self.csv = pd.read_csv(ann_path)
        self.transform = transform

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, idx):
        path = f'{self.dataset_root}/{self.csv.iloc[idx, 0]}'
        label_0 = self.csv.iloc[idx, 1]
        label_1 = self.csv.iloc[idx, 2]

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
       
        image = self.transform(image=image)['image']
        image = image / 255.0
        assert image.shape == (3, 512, 512), f"shape: {image.shape}"

        return image, torch.tensor([label_0, label_1])
    
    

class DatasetPseudo(Dataset):
    def __init__(
        self,
        train_dataset_root,
        pseudo_dataset_root,
        ann_path,
        pseudo_ann_path,
        split,
        multiplier=1,
        transform=None,
    ):
        super().__init__()
        train_paths = sorted(glob(f'{train_dataset_root}/*.tif'))

        if split == 'train':
            train_csv = pd.read_csv(ann_path)
            train_csv = train_csv[train_csv.split == 'train'].reset_index(drop=True)
        elif split == 'val':
            train_csv = pd.read_csv(ann_path)
            train_csv = train_csv[train_csv.split == 'valid'].reset_index(drop=True)

        train_csv = train_csv[train_csv.image_id.isin([_.split('/')[-1] for _ in train_paths])].reset_index(drop=True)
        self.paths, self.labels = [], []

        for index in range(train_csv.shape[0]):
            self.paths.append(f'{train_dataset_root}/{train_csv.iloc[index, 0]}')
            if int(train_csv.iloc[index, 1]) == 0:
                self.labels.append((1, 0))
            else:
                self.labels.append((0, 1))

        pseudo_csv = pd.read_csv(pseudo_ann_path)
        self.paths.extend([f'{pseudo_dataset_root}/{_}' for _ in pseudo_csv.iloc[:, 0].values])
        self.labels.extend(pseudo_csv.iloc[:, 1:].values)

        self.paths *= multiplier
        self.labels *= multiplier

        assert len(self.paths) == len(self.labels)
            
        logging.info(f'Dataset len: {len(self.paths)}')
        
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]

        image = tifffile.memmap(path, mode='r')[:, :, [3,2,1]]
        assert image.min() >= 0 and image.max() <= 2, f'min: {image.min()} max: {image.max()}'
        if self.transform:
            image = self.transform(image=image)['image']
            image = image #/ (image.max() + 1e-5)

        return image, torch.tensor(label).float()
    