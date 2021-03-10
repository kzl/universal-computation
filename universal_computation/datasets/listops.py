import tensorflow_datasets
import torch

from universal_computation.datasets.dataset import Dataset
from universal_computation.datasets.helpers.listops import get_datasets


class ListopsDataset(Dataset):

    def __init__(self, batch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.batch_size = batch_size  # we fix it so we can use dataloader

        self.d_train, self.d_test, *_ = get_datasets(1, 'basic', batch_size=batch_size, data_dir='data/listops/')

        self.train_enum = iter(tensorflow_datasets.as_numpy(self.d_train))
        self.test_enum = iter(tensorflow_datasets.as_numpy(self.d_test))

    def reset_test(self):
        self.test_enum = enumerate(self.d_test)

    def get_batch(self, batch_size=None, train=True):
        if train:
            batch = next(self.train_enum, None)
            if batch is None:
                self.train_enum = iter(tensorflow_datasets.as_numpy(self.d_train))
                batch = next(self.train_enum)
        else:
            batch = next(self.test_enum, None)
            if batch is None:
                self.test_enum = iter(tensorflow_datasets.as_numpy(self.d_test))
                batch = next(self.test_enum)

        x, y = batch['inputs'], batch['targets']
        x = torch.from_numpy(x).long()
        y = torch.from_numpy(y).long()

        x = x.to(device=self.device)
        y = y.to(device=self.device)

        self._ind += 1

        return x, y
