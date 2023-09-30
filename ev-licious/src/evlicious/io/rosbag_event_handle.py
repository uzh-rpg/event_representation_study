from pathlib import Path
from .utils import event_handle
from .utils.events import Events

try:
    import rosbag
    import rospy
except ImportError:
    print("Cannot use ros api")


import numpy as np
import tqdm


def _find_index_from_timestamps(t_query, t_events):
    return np.searchsorted(t_events, t_query+1e-3)


class RosbagEventHandle(event_handle.EventHandle):
    def __init__(self, bag):
        self.bag = bag

        info = bag.get_type_and_topic_info()
        self.topics = [topic for topic, msg_type in info.topics.items() if msg_type.msg_type == "dvs_msgs/EventArray"][0]

        self.event_timestamps = []
        self.message_timestamps = []
        self.num_events = []
        pbar = tqdm.tqdm(total=bag.get_message_count(topic_filters=[self.topics]))
        for topic, msg, t in bag.read_messages(topics=self.topics):
            self.event_timestamps.append(msg.events[0].ts)
            self.num_events.append(len(msg.events))
            self.message_timestamps.append(t)
            pbar.update(1)
        self.height = msg.height
        self.width = msg.width
        self.num_events = np.cumsum(self.num_events)

    def __del__(self):
        self.bag.close()

    @classmethod
    def from_path(cls, path: Path, height=None, width=None):
        bag = rosbag.Bag(str(path), "r")
        return cls(bag)

    def get_message_time_window(self, i0, i1):
        t0 = self.message_timestamps[i0]
        t1 = self.message_timestamps[i1]
        return t0, t1

    def get_between_idx(self, i0, i1):
        idx0 = np.searchsorted(self.num_events, i0)
        idx1 = np.searchsorted(self.num_events, i1)
        t0_msg, t1_msg = self.get_message_time_window(idx0, idx1)
        return self._read_events(counter=self.num_events[idx0-1] if idx0 > 0 else 0,
                                 i0=i0,
                                 i1=i1,
                                 t0_msg=t0_msg,
                                 t1_msg=t1_msg)


    def _read_events(self, t0_msg, t1_msg, t0=None, t1=None, i0=-1, i1=-1, counter=-1):
        t = []
        p = []
        x = []
        y = []
        by_count = counter >= 0
        by_time = t0 is not None

        for topic, msg, _ in self.bag.read_messages(topics=self.topics, start_time=t0_msg, end_time=t1_msg):
            for e in msg.events:
                if by_count and i0 <= counter and counter < i1 \
                    or by_time and t0 <= e.ts and e.ts < t1:
                    t.append(e.ts.to_nsec()//1e3)
                    x.append(e.x)
                    y.append(e.y)
                    p.append(1 if e.polarity else -1)
                if by_count:
                    counter += 1

        return Events(x=np.array(x).astype("uint16"),
                      y=np.array(y).astype("uint16"),
                      t=np.array(t).astype("int64"),
                      p=np.array(p).astype("int8"),
                      divider=1,
                      height=self.height,
                      width=self.width)


    def get_between_time(self, t0_us: int, t1_us: int):
        t0 = rospy.Time.from_nsec(t0_us * 1e3)
        t1 = rospy.Time.from_nsec(t1_us * 1e3)

        idx0 = np.searchsorted(self.event_timestamps, t0)
        idx1 = np.searchsorted(self.event_timestamps, t1)

        t0_msg, t1_msg = self.get_message_time_window(idx0, idx1)
        return self._read_events(t0=t0,
                                 t1=t1,
                                 t0_msg=t0_msg,
                                 t1_msg=t1_msg)

    def __len__(self):
        return self.num_events[-1]







