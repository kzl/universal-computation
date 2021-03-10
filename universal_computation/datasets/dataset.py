import torch


class Dataset:

    def __init__(self, device='cpu'):
        self.device = device
        self._ind = 0

    def get_batch(self, batch_size, train=True):
        x, y = self.get_batch_np(batch_size, train=train)
        x = torch.from_numpy(x).to(device=self.device, dtype=torch.float32)
        y = torch.from_numpy(y).to(device=self.device, dtype=torch.long)
        self._ind += 1
        return x, y

    def get_batch_np(self, batch_size, train):
        raise NotImplementedError

    def start_epoch(self):
        self._ind = 0
