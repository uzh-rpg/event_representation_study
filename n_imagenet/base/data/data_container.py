from abc import ABC, abstractmethod
from math import ceil


class DataContainer(ABC):
    """
    Abstract class defining data container.
    A typical data container will encompass all the utilities related to dataset generation and loading.
    Note that cfg contains all the neccessary data needed to create a dataset.
    """

    def __init__(self, cfg, **kwargs):
        super(DataContainer, self).__init__()
        self.cfg = cfg
        # dataset is a dict containing different torch.utils.data.Dataset instances
        self.dataset = {"train": None, "test": None, "val": None}
        # dataloader is a dict containing different torch.utils.data.DataLoader instances
        self.dataloader = {"train": None, "test": None, "val": None}

    @abstractmethod
    def gen_dataset(self, **kwargs):
        """
        Method for generating dataset. Sets the key-value pairs in self.dataset.
        """
        pass

    @abstractmethod
    def gen_dataloader(self, **kwargs):
        """
        Method for creating dataloader. Sets the key-value pairs in self.dataloader.
        It is highly recommended to write a custom collate_fn, to pass around data as a dictionary.
        This enables efficient code recycling.
        """
        assert self.dataset is not None


class DataChunkContainer(DataContainer):
    """
    Abstract class defining data chunk container. Data will be pre-loaded into RAM for faster data loading.
    For DataChunkContainer to successfully work, the Dataset generated from gen_dataset should be an instance of ChunkDataset.
    """

    def __init__(self, cfg, **kwargs):
        super(DataChunkContainer, self).__init__(cfg)
        self.chunk_every = ceil(self.cfg.chunk_size / self.cfg.batch_size)

    def create_chunk(self, batch_idx, mode):
        """
        Load chunk to RAM.
        """
        if batch_idx % self.chunk_every == 0:
            print(f"Creating chunk with index {self.dataset[mode].chunk_idx}")
            self.dataset[mode].load_chunk()

    def refresh_chunk(self, mode):
        """
        Free chunk, and prepare for new epoch.
        """
        print("Refreshing chunk!")
        self.dataset[mode].restart_chunk()
        self.dataset[mode].free_chunk()
        self.dataset[mode].shuffle_index()

    def release_chunk(self, mode):
        """
        Free chunk to save RAM.
        """
        print("Releasing chunk!")
        self.dataset[mode].free_chunk()
