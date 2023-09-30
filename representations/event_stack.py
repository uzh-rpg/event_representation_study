import torch
import numpy as np


class EventStack(object):
    NO_VALUE = 0.0
    STACK_LIST = ["stacked_polarity", "index"]

    def __init__(self, stack_size, num_of_event, height, width):
        self.stack_size = stack_size
        self.num_of_event = num_of_event
        self.height = height
        self.width = width

    def pre_stack(self, event_sequence, last_timestamp):
        x = event_sequence["x"].astype(np.int32)
        y = event_sequence["y"].astype(np.int32)
        p = 2 * event_sequence["p"].astype(np.int8) - 1
        t = event_sequence["t"].astype(np.int64)

        assert len(x) == len(y) == len(p) == len(t)

        past_mask = t <= last_timestamp
        p_x, p_y, p_p, p_t = x[past_mask], y[past_mask], p[past_mask], t[past_mask]
        p_t = p_t - p_t.min()
        past_stacked_event = self.make_stack(p_x, p_y, p_p, p_t)

        future_mask = t > last_timestamp
        if np.sum(future_mask) == 0:
            stacked_event_list = [past_stacked_event]
        else:
            f_x = x[future_mask][::-1]
            f_y = y[future_mask][::-1]
            f_p = p[future_mask][::-1]
            f_t = t[future_mask][::-1]
            f_p = f_p * -1
            f_t = f_t - f_t.min()
            f_t = f_t.max() - f_t
            future_stacked_event = self.make_stack(f_x, f_y, f_p, f_t)

            stacked_event_list = [past_stacked_event, future_stacked_event]

        return stacked_event_list

    def post_stack(self, pre_stacked_event):
        stacked_event_list = []
        for pf_stacked_event in pre_stacked_event:
            stacked_polarity = np.zeros([self.height, self.width, 1], dtype=np.float32)
            cur_stacked_event_list = []
            for stack_idx in range(self.stack_size - 1, -1, -1):
                stacked_polarity.put(
                    pf_stacked_event["index"][stack_idx],
                    pf_stacked_event["stacked_polarity"][stack_idx],
                )
                cur_stacked_event_list.append(np.stack([stacked_polarity], axis=2))
            stacked_event_list.append(
                np.concatenate(cur_stacked_event_list[::-1], axis=2)
            )
        if len(stacked_event_list) == 2:
            stacked_event_list[1] = stacked_event_list[1][:, :, ::-1, :]
        stacked_event = np.stack(stacked_event_list, axis=2)

        return stacked_event[..., 0]

    def make_stack(self, x, y, p, t):
        t = t - t.min()
        time_interval = t.max() - t.min() + 1
        t_s = (t / time_interval * 2) - 1.0
        stacked_event_list = {stack_value: [] for stack_value in self.STACK_LIST}
        cur_num_of_events = len(t)
        for _ in range(self.stack_size):
            stacked_event = self.stack_data(x, y, p, t_s)
            stacked_event_list["stacked_polarity"].append(
                stacked_event["stacked_polarity"]
            )

            cur_num_of_events = cur_num_of_events // 2
            x = x[cur_num_of_events:]
            y = y[cur_num_of_events:]
            p = p[cur_num_of_events:]
            t_s = t_s[cur_num_of_events:]
            t = t[cur_num_of_events:]

        grid_x, grid_y = np.meshgrid(
            np.linspace(0, self.width - 1, self.width, dtype=np.int32),
            np.linspace(0, self.height - 1, self.height, dtype=np.int32),
        )
        for stack_idx in range(self.stack_size - 1):
            prev_stack_polarity = stacked_event_list["stacked_polarity"][stack_idx]
            next_stack_polarity = stacked_event_list["stacked_polarity"][stack_idx + 1]

            assert np.all(
                next_stack_polarity[(prev_stack_polarity - next_stack_polarity) != 0]
                == 0
            )

            diff_stack_polarity = prev_stack_polarity - next_stack_polarity

            mask = diff_stack_polarity != 0
            stacked_event_list["index"].append(
                (grid_y[mask] * self.width) + grid_x[mask]
            )
            stacked_event_list["stacked_polarity"][stack_idx] = diff_stack_polarity[
                mask
            ]

        last_stack_polarity = stacked_event_list["stacked_polarity"][
            self.stack_size - 1
        ]
        mask = last_stack_polarity != 0
        stacked_event_list["index"].append((grid_y[mask] * self.width) + grid_x[mask])
        stacked_event_list["stacked_polarity"][
            self.stack_size - 1
        ] = last_stack_polarity[mask]

        return stacked_event_list

    def stack_data(self, x, y, p, t_s):
        assert len(x) == len(y) == len(p) == len(t_s)

        stacked_polarity = np.zeros([self.height, self.width], dtype=np.int8)

        index = (y * self.width) + x

        stacked_polarity.put(index, p)

        stacked_event = {
            "stacked_polarity": stacked_polarity,
        }

        return stacked_event

    @staticmethod
    def collate_fn(batch):
        batch = torch.utils.data._utils.collate.default_collate(batch)

        return batch
