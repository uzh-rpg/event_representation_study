import torch
from torch_scatter import scatter


class Operations(object):
    def __init__(self, func, aggregation, height, width):
        self.func = func
        self.aggregation = aggregation
        self.height = height
        self.width = width

    def __call__(self, events):
        return self.exec(events)

    def run(self, src, index):
        if self.aggregation == "variance":
            event_surface = scatter(
                src=src, index=index, dim_size=self.height * self.width, reduce="mean"
            )
            event_surface_squared = scatter(
                src=src**2,
                index=index,
                dim_size=self.height * self.width,
                reduce="mean",
            )
            result = event_surface_squared - event_surface**2
            event_surface = result
        else:
            event_surface = scatter(
                src=src,
                index=index,
                dim_size=self.height * self.width,
                reduce=self.aggregation,
            )
        event_surface = event_surface.reshape(self.height, self.width)

        return event_surface

    def exec(self, events):
        index = torch.tensor(events[:, 0] + events[:, 1] * self.width).to(torch.int64)

        if self.func == "timestamp":
            event_surface = self.run(torch.tensor(events[:, 2]), index)
        elif self.func == "polarity":
            event_surface = self.run(torch.tensor(events[:, 3]), index)
        elif self.func == "count":
            event_surface = self.run(
                torch.tensor(events).new_ones([events.shape[0]]), index
            )
        elif self.func == "timestamp_pos":
            positive_events = events[events[:, 3] == 1]
            positive_index = torch.tensor(
                positive_events[:, 0] + positive_events[:, 1] * self.width
            ).to(torch.int64)
            event_surface = self.run(
                torch.tensor(positive_events[:, 2]), positive_index
            )
        elif self.func == "timestamp_neg":
            negative_events = events[events[:, 3] == -1]
            if len(list(negative_events)) == 0:
                negative_events = events[events[:, 3] == 0]
            negative_index = torch.tensor(
                negative_events[:, 0] + negative_events[:, 1] * self.width
            ).to(torch.int64)
            event_surface = self.run(
                torch.tensor(negative_events[:, 2]), negative_index
            )
        elif self.func == "count_pos":
            positive_events = events[events[:, 3] == 1]
            positive_index = torch.tensor(
                positive_events[:, 0] + positive_events[:, 1] * self.width
            ).to(torch.int64)
            event_surface = self.run(
                torch.tensor(positive_events).new_ones([positive_events.shape[0]]),
                positive_index,
            )
        elif self.func == "count_neg":
            negative_events = events[events[:, 3] == -1]
            if len(list(negative_events)) == 0:
                negative_events = events[events[:, 3] == 0]
            negative_index = torch.tensor(
                negative_events[:, 0] + negative_events[:, 1] * self.width
            ).to(torch.int64)
            event_surface = self.run(
                torch.tensor(negative_events).new_ones([negative_events.shape[0]]),
                negative_index,
            )

        return event_surface.numpy()
