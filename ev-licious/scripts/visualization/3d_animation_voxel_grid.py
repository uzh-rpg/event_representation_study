import argparse
import evlicious


def FLAGS():
    parser = argparse.ArgumentParser(description='''Generate event graph animation.''')
    parser = evlicious.args.add_io_args(parser)
    parser.add_argument('--n_events_window', type=int, help='Downsample number.', default=300000)
    parser.add_argument('--n_events_step', type=int, help='Downsample number.', default=3000)
    return parser.parse_args()


if __name__ == "__main__":
    args = FLAGS()
    events = evlicious.io.load_events_from_path(args.events)
    events.height = 636+1
    events.width = 966+1

    visualizer = evlicious.art.O3DVoxelGridVisualizer(events, factor=1)
    visualizer.loop(args.output_path, n_events_window=args.n_events_window, n_events_step=args.n_events_step)

