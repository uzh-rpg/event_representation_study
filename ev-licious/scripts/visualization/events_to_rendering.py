import argparse

import evlicious
import cv2


if __name__ == "__main__":
    parser = argparse.ArgumentParser("""Visualize Events Interactively""")
    parser = evlicious.args.add_io_args(parser)
    parser.add_argument("--width", type=int, default=-1)
    parser.add_argument("--height", type=int, default=-1)
    parser.add_argument("--time_step_us", type=int, default=10000)
    parser.add_argument("--time_window_us", type=int, default=40000)
    
    args = parser.parse_args()

    args.width = args.width if args.width > 0 else None
    args.height = args.height if args.height > 0 else None
    handle = evlicious.io.load_events_from_path(args.events, height=args.height, width=args.width)
    
    if not args.output_path.exists():
        args.output_path.mkdir()

    counter = 0
    for events in handle.iterator(step_size=args.time_step_us, window=args.time_window_us, step_size_unit="us", window_unit="us", pbar=True):
        image = events.render(cast=False, rendering_type=evlicious.RenderingType.RED_BLUE_NO_OVERLAP)
        cv2.imwrite(str(args.output_path / ("%05d.png" % counter)), image)
        counter += 1

