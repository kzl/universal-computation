from tape import utils
from tape.datasets import RemoteHomologyDataset as TAPERemoteHomologyDataset

from universal_computation.datasets.dataset import Dataset


class RemoteHomologyDataset(Dataset):

    """
    Note that we clip all sequences less than max_seq_len = 1024
    for the sake of simplicity in our paper. This leaves a remaining
    236224 examples (out of 242560 -- 97.39%). To correct the accuracy,
    we multiply reported accuracies for the paper by .9739.

    We pad lazily inside this dataset by assigning ID 28 to mean padding.
    This increases the input dimension of the model by 1, so the model
    should have an input dimension of 29.
    """

    def __init__(self, data_subdir='tape', max_seq_len=1024, train_batch_size=2, test_batch_size=8, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_seq_len = max_seq_len

        data_dir = f'data/{data_subdir}'
        train_dataset = TAPERemoteHomologyDataset(data_dir, 'train')
        val_dataset = TAPERemoteHomologyDataset(data_dir, 'valid')

        self.d_train = utils.setup_loader(train_dataset, train_batch_size, -1, 1, 1, 0)
        self.d_test = utils.setup_loader(val_dataset, test_batch_size, -1, 1, 1, 0)

        self.train_enum = enumerate(self.d_train)
        self.test_enum = enumerate(self.d_test)

    def get_batch(self, batch_size=None, train=True):

        seq_len = self.max_seq_len + 1

        while seq_len > self.max_seq_len:
            if train:
                _, data = next(self.train_enum, (None, None))
                if data is None:
                    self.train_enum = enumerate(self.d_train)
                    _, data = next(self.train_enum)
            else:
                _, data = next(self.test_enum, (None, None))
                if data is None:
                    self.test_enum = enumerate(self.d_test)
                    _, data = next(self.test_enum)
            x, y = data['input_ids'], data['targets']
            seq_len = x.shape[1]

        x = x.to(device=self.device)
        y = y.to(device=self.device)

        self._ind += 1
        return x, y
