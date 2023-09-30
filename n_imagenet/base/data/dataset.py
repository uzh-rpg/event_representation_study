from torch.utils.data import Dataset


class ChunkDataset(Dataset):
    """
    Class implementing chunk-based loading.
    """

    def __init__(self, cfg, mode="train"):
        super(ChunkDataset, self).__init__()
        self.cfg = cfg
        self.mode = mode
        self.chunk_idx = 0  # Next chunk index to load

    def shuffle_index(self):
        """
        Shuffle indices for re-sampling chunk.
        """
        raise NotImplementedError

    def load_chunk(self):
        """
        Load a chunk to RAM.
        """
        raise NotImplementedError

    def restart_chunk(self):
        self.chunk_idx = 0

    def free_chunk(self):
        """
        Free all reserved chunks to save RAM space.
        """
        raise NotImplementedError
