import numpy as np
import numba as nb
from torch.utils.data.sampler import Sampler


class RandomContinuousSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, data_len, num, data_index):
        self.dataset = data_len
        self.num = num
        self.data_index = data_index

    def __iter__(self):
        self.indices = random_batch_indice(self.dataset, self.num, self.data_index)
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


# get random continuous random numbers
@nb.jit()
def random_batch_indice(data_len, num, index_list):
    """
    :param data_len: length of dataset
    :param num: continuous random numbers, e.g. num=2
    """
    data_list = list(range(data_len))
    split_list = []
    for idx in range(data_len // num):
        batch = data_list[idx * num : (idx + 1) * num]
        split_list.append(batch)
    for i, value in enumerate(split_list):
        for j in range(len(value)):
            if value[j] in index_list:
                split_list.remove(value)
                continue
    split_list = np.array(split_list)
    np.random.shuffle(split_list)
    return split_list.reshape(-1)
