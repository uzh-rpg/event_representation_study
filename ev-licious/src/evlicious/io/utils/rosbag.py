import rospy
from .events import Events
from .dvs_msgs._Event import Event
from .dvs_msgs._EventArray import EventArray


def _us_to_rostime(t_us):
    return rospy.Time.from_seconds(t_us / 1e6)

def _event_to_event_msg(event):
    x,y,t,p = event
    return Event(x=x, y=y, ts=_us_to_rostime(t), polarity=p==1)

def events_to_ros_message(events: Events, seq=0) -> EventArray:
    array = EventArray()
    array.height = events.height
    array.width = events.width

    array.header.seq = seq
    array.header.stamp = _us_to_rostime(events.t[-1])

    array.events = list(map(_event_to_event_msg, events.iter_events()))

    return array