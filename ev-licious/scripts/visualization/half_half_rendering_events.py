import argparse
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from os.path import join

import evlicious


def FLAGS():
    parser = argparse.ArgumentParser(description='''Blend together events and frames.''')
    parser = evlicious.args.add_io_args(parser)
    parser.add_argument('--images', help='Root of folder with images.', default="")
    parser.add_argument('--height', type=int, help='Height.', default=-1)
    parser.add_argument('--width', type=int, help='Width.', default=-1)
    parser.add_argument('--t_window_us', type=int, help='Time window in microseconds.', default=10000)
    parser.add_argument('--blending_scale', type=float, help='Blending scale.', default=0.02)

    return parser.parse_args()


if __name__ == "__main__":
    args = FLAGS()
    args.width = args.width if args.width > 0 else None
    args.height = args.height if args.height > 0 else None

    # load images
    images = sorted(glob.glob(join(args.images, "*.jpg")))
    image_stamps = np.genfromtxt(join(args.images, "timestamps.txt"))

    events = evlicious.io.load_events_from_path(args.events)

    t0_ms = image_stamps[len(image_stamps) // 2]
    events = events.get_between_time(t0_ms, t0_ms + args.t_window_us)
    image = cv2.imread(images[len(image_stamps) // 2])

    event_rendering = events.render()
    print(event_rendering)
    output = evlicious.art.blend_images(image, event_rendering, args.blending_scale)[...,::-1]

    plt.imshow(output)
    plt.show()