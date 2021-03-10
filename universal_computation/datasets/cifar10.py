from einops import rearrange
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from universal_computation.datasets.dataset import Dataset


class CIFAR10Dataset(Dataset):

    def __init__(self, batch_size, patch_size=None, data_aug=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.batch_size = batch_size  # we fix it so we can use dataloader
        self.patch_size = patch_size  # grid of (patch_size x patch_size)

        if data_aug:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.d_train = DataLoader(
            torchvision.datasets.CIFAR10('data/cifar', download=True, train=True, transform=transform),
            batch_size=batch_size, drop_last=True, shuffle=True,
        )
        self.d_test = DataLoader(
            torchvision.datasets.CIFAR10('data/cifar', download=True, train=False, transform=val_transform),
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
                _, (x, y) = next(self.test_enum)

        if self.patch_size is not None:
            x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)

        x = x.to(device=self.device)
        y = y.to(device=self.device)

        self._ind += 1

        return x, y
