from einops import rearrange
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from universal_computation.datasets.dataset import Dataset


class MNISTDataset(Dataset):

    def __init__(self, batch_size, patch_size=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.batch_size = batch_size  # we fix it so we can use dataloader
        self.patch_size = patch_size  # grid of (patch_size x patch_size)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=0., std=1.),
        ])

        self.d_train = DataLoader(
            torchvision.datasets.MNIST('data/mnist', download=True, train=True, transform=transform),
            batch_size=batch_size, drop_last=True, shuffle=True,
        )
        self.d_test = DataLoader(
            torchvision.datasets.MNIST('data/mnist', download=True, train=False, transform=transform),
            batch_size=batch_size, drop_last=True, shuffle=True,
        )

        self.train_enum = enumerate(self.d_train)
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
