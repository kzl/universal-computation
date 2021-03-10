import numpy as np

from universal_computation.datasets.dataset import Dataset


class BitXORDataset(Dataset):

    def __init__(self, n=5, num_patterns=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n
        self.num_patterns = num_patterns

    def get_batch_np(self, batch_size, train):
        bits = np.random.randint(low=0, high=2, size=(batch_size, self.num_patterns, self.n))
        xored_bits = bits[:,0]
        for i in range(1, self.num_patterns):
            xored_bits = np.logical_xor(xored_bits, bits[:,i])
        return bits, xored_bits
