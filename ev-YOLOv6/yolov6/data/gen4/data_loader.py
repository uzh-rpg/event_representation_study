import torch
import numpy as np
from torch.utils.data.dataloader import default_collate
from data_sampler import RandomContinuousSampler


class Loader:
    def __init__(
        self,
        dataset,
        mode,
        batch_size,
        num_workers,
        pin_memory,
        drop_last,
        sampler,
        data_index=None,
    ):
        if mode == "training":
            self.loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
                collate_fn=collate_events,
            )
        else:
            self.loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
                collate_fn=collate_events_test,
            )

    def __iter__(self):
        for data in self.loader:
            yield data

    def __len__(self):
        return len(self.loader)


def collate_events(data):
    batch_labels = []
    batch_pos_events = []
    batch_neg_events = []
    idx_batch = 0

    for d in data:  # different batch
        for idx in range(len(d[0])):
            label = d[0][idx]
            lb = np.concatenate(
                [label, idx_batch * np.ones((len(label), 1), dtype=np.float32)], 1
            )

            batch_labels.append(lb)
            idx_batch += 1
        batch_pos_events.append(d[1])
        batch_neg_events.append(d[2])
    labels = np.concatenate(batch_labels, 0)
    return labels, batch_pos_events, batch_neg_events


def collate_events_test(data):
    labels = []
    pos_events = []
    neg_events = []
    idx_batch = 0

    for d in data:
        for idx in range(len(d[0])):
            label = d[0][idx]
            lb = np.concatenate(
                [label, idx_batch * np.ones((len(label), 1), dtype=np.float32)], 1
            )
            labels.append(lb)
            idx_batch += 1
        pos_events.append(d[1])
        neg_events.append(d[2])

    labels = np.concatenate(labels, 0)

    return labels, pos_events, neg_events
