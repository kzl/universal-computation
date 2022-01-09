import os
import pathlib
from pathlib import Path
import pandas as pd
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms

from universal_computation.datasets.dataset import Dataset

class EuroSatDatasetHelper(torch.utils.data.Dataset):
    def __init__(self, img_dir, ann_file, transform=None, target_transform=None):
        df = pd.read_csv(ann_file)
        self.img_labels = df[['img_name', 'int_label']].reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        label = self.img_labels.iloc[idx, 1]
        img_name = self.img_labels.iloc[idx, 0]
        dir_name = img_name.split('_')[0]
        img_path = os.path.join(self.img_dir, dir_name, img_name)
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label


class EuroSatDataset(Dataset):
    def __init__(self, batch_size, patch_size=None, data_aug=True, *args, **kwargs):
        super(EuroSatDataset, self).__init__(*args, **kwargs)

        self.batch_size = batch_size
        self.patch_size = patch_size

        if data_aug:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224,224), interpolation=3),
                transforms.RandomApply([transforms.GaussianBlur(3)]),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224,224), interpolation=3),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224), interpolation=3),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])
        train_test_dir = 'data/2750'
        self.d_train = DataLoader(
            EuroSatDatasetHelper(train_test_dir, os.path.join(train_test_dir, 'train.csv'), transform=transform),
            batch_size=batch_size, drop_last=True, shuffle=True,
        )
        self.d_test = DataLoader(
            EuroSatDatasetHelper(train_test_dir, os.path.join(train_test_dir, 'test.csv'), transform=val_transform), 
            batch_size=batch_size, drop_last=True, shuffle=True,
        )

        self.train_enum = enumerate(self.d_train)
        self.test_enum = enumerate(self.d_test)

        self.train_size = len(self.d_train)
        self.test_size = len(self.d_test)

    def reset_test(self):
        self.test_enum = enumerate(self.d_test)

    def get_batch(self, batch_size=None, train=True):
        if train:
            _, (x, y) = next(self.train_enum, (None, (None, None)))
            if x is None:
                self.train_enum = enumerate(self.d_train)
                _, (x, y) = next(self.train_enum)
        else:
            _, (x, y) = next(self.test_enum, (None, (None, None)))
            if x is None:
                self.test_enum = enumerate(self.d_test)
                _, (x, y) = next(self.train_enum)

        if self.patch_size is not None:
            x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)

        x = x.to(device=self.device)
        y = y.to(device=self.device)

        self._ind += 1

        return x, y
