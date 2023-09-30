from real_cnn_model.data.imagenet import ImageNetDataset
from torch.utils.data import DataLoader
from base.data.data_container import DataContainer
import torch


class ImageNetContainer(DataContainer):
    def __init__(self, cfg, **kwargs):
        """
        cfg contains all the contents of a config file
        """
        super(ImageNetContainer, self).__init__(cfg)
        print(f"Initializing data container {self.__class__.__name__}...")
        self.gen_dataset()
        self.gen_dataloader()

    def gen_dataset(self, **kwargs):
        dataset_name = getattr(self.cfg, "dataset_name", "imagenet")
        if dataset_name == "imagenet":
            if self.cfg.mode == "train":
                train_dataset = ImageNetDataset(self.cfg, mode="train")
                val_dataset = ImageNetDataset(self.cfg, mode="val")

                self.dataset["train"] = train_dataset
                self.dataset["val"] = val_dataset

            elif self.cfg.mode == "test":
                if getattr(self.cfg, "streak", False):
                    test_dataset = StreakImageNetDataset(self.cfg)
                    self.dataset["test"] = test_dataset
                else:
                    test_dataset = ImageNetDataset(self.cfg, mode="test")
                    self.dataset["test"] = test_dataset

            else:
                raise AttributeError("Mode not provided")

    def gen_dataloader(self, **kwargs):
        # collate_fn for generating dictionary-style batches.
        def dict_collate_fn(list_data):
            # list_data is as follows: [(img: torch.Tensor(H, W, 3), event: torch.Tensor(H, W, 4), label: int), ...]
            collate_type = getattr(self.cfg, "collate_type", "normal")
            if collate_type == "normal":
                events, labels = list(zip(*list_data))

                event_batch = torch.stack(events, dim=0)
                label_batch = torch.LongTensor(labels)
                return {"input_data": event_batch, "label": label_batch}

        assert self.dataloader is not None
        if self.cfg.mode == "train":
            TrainLoader = DataLoader(
                self.dataset["train"],
                collate_fn=dict_collate_fn,
                batch_size=self.cfg.batch_size,
                shuffle=True,
                num_workers=self.cfg.num_workers,
                drop_last=False,
                pin_memory=self.cfg.pin_memory,
            )

            ValLoader = DataLoader(
                self.dataset["val"],
                collate_fn=dict_collate_fn,
                batch_size=self.cfg.batch_size,
                shuffle=True,
                num_workers=self.cfg.num_workers,
                drop_last=False,
                pin_memory=self.cfg.pin_memory,
            )

            self.dataloader["train"] = TrainLoader
            self.dataloader["val"] = ValLoader

        elif self.cfg.mode == "test":
            TestLoader = DataLoader(
                self.dataset["test"],
                collate_fn=dict_collate_fn,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                drop_last=False,
                pin_memory=self.cfg.pin_memory,
            )
            self.dataloader["test"] = TestLoader

        else:
            raise AttributeError("Mode not provided")
