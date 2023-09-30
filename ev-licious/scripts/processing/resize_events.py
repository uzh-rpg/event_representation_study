import argparse
import evlicious
import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser("""Downsample events""")
    parser = evlicious.args.add_io_args(parser)
    parser.add_argument("--output_height", type=int, default=180)
    parser.add_argument("--output_width", type=int, default=320)
    args = parser.parse_args()

    events = evlicious.io.load_events_from_path(args.events)
    downsampled = evlicious.tools.resize_to_resolution(events, height=args.output_height, width=args.output_width, chunks=100, pbar=tqdm.tqdm(total=100))
    if args.output_path != ".":
        evlicious.io.save_events(args.output_path, downsampled)

