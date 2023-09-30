import argparse
import evlicious

try:
    import rosbag
except ImportError:
    print("Cannot use ros api")


def FLAGS():
    parser = argparse.ArgumentParser("""Write events to a rosbag""")
    parser = evlicious.args.add_io_args(parser)
    parser.add_argument("--framerate", type=int, default=30, help=f"framerate of the event arrays in the output rosbag")

    # background activity filter
    parser.add_argument("--topic", type=str, default="/dvs/events")

    flags = parser.parse_args()
    assert flags.events.exists()

    if str(flags.output_path) == ".":
        flags.output_path = flags.output_path / flags.events.name

    return flags


if __name__ == '__main__':
    flags = FLAGS()
    events = evlicious.io.load_events_from_path(flags.events)

    delta_t_us = int(1e6 / flags.framerate)
    t_min_us, t_max_us = events.get_time_limits()
    total = (t_max_us - t_min_us) // delta_t_us

    rosbag_flag = "a" if flags.output_path.exists() else "w"

    with evlicious.tools.task_manager.TaskManager(total=total) as tm:
        for ev in events.iterator(step_size=delta_t_us, step_size_unit="us", window=delta_t_us, window_unit="us", pbar=False):
            if len(ev) > 0:
                tm.new_task(evlicious.io.events_to_ros_message, ev)
            else:
                tm.pbar.update(1)

    with rosbag.Bag(flags.output_path, mode=rosbag_flag) as bag:
        for msg in tm.outputs:
            bag.write(flags.topic, msg, msg.header.stamp)

