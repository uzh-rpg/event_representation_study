import argparse
import evlicious
try:
    import rosbag
    from cv_bridge import CvBridge
    import rospy
except ImportError:
    print("Cannot use ros api")


import numpy as np
import cv2


def FLAGS():
    parser = argparse.ArgumentParser("""Write images to a rosbag""")
    parser = evlicious.args.add_io_args(parser)

    parser.add_argument("--topic", type=str, default="/dvs/reconstruction")

    flags = parser.parse_args()
    assert flags.events.exists()

    if str(flags.output_path) == ".":
        flags.output_path = flags.output_path / flags.events.name

    return flags


if __name__ == '__main__':
    flags = FLAGS()
    bridge = CvBridge()
    images = sorted(list(flags.events.rglob("*.png")))
    timestamps_us = np.genfromtxt(flags.events / "timestamps.txt")
    rosbag_flag = "a" if flags.output_path.exists() else "w"

    with rosbag.Bag(flags.output_path, mode=rosbag_flag) as bag:
        for img_path, t_us in zip(images, timestamps_us):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            msg = bridge.cv2_to_imgmsg(img, "mono8")
            t_ros = rospy.Time.from_seconds(t_us/1e6)
            msg.header.stamp = t_ros
            bag.write(flags.topic, msg, t_ros)

