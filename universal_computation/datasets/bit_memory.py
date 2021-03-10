import numpy as np

from universal_computation.datasets.dataset import Dataset


class BitMemoryDataset(Dataset):

    def __init__(self, n=1000, num_patterns=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n
        self.num_patterns = num_patterns

    def get_batch_np(self, batch_size, train):
        bits = np.random.randint(low=0, high=2, size=(batch_size, self.num_patterns, self.n))
        bits = 2 * bits - 1
        query_inds = np.random.randint(low=0, high=self.num_patterns, size=batch_size)
        query_bits = bits[range(batch_size), query_inds]
        mask = np.random.randint(low=0, high=2, size=query_bits.shape)
        masked_query_bits = mask * query_bits
        masked_query_bits = masked_query_bits.reshape(batch_size, 1, self.n)
        x = np.concatenate([bits, masked_query_bits], axis=1)
        y = query_bits
        return x, y
