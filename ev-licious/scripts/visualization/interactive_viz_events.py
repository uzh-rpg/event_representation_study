import argparse
import evlicious


if __name__ == "__main__":
    parser = argparse.ArgumentParser("""Visualize Events Interactively""")
    parser = evlicious.args.add_io_args(parser)
    parser.add_argument("--width", type=int, default=-1)
    parser.add_argument("--height", type=int, default=-1)
    args = parser.parse_args()

    args.width = args.width if args.width > 0 else None
    args.height = args.height if args.height > 0 else None
    evlicious.io.load_events_from_path(args.events, height=args.height, width=args.width).interactive_viz_loop()
