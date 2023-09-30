import argparse
import matplotlib.pyplot as plt
import evlicious


def FLAGS():
    parser = argparse.ArgumentParser(description='''Visualize event slice in 2D''')
    parser = evlicious.args.add_io_args(parser)
    parser.add_argument('--height', type=int, help='Height in pixels.', default=-1)
    parser.add_argument('--width', type=int, help='Width in pixels.', default=-1)
    parser.add_argument('--y_coord', type=int, default=-1)
    parser.add_argument('--x_coord', type=int, default=-1)

    return parser.parse_args()


if __name__ == "__main__":
    args = FLAGS()
    events = evlicious.io.load_events_from_path(args.events).load()

    fig, ax = plt.subplots()
    if args.y_coord > 0:
        evlicious.art.plot_y_slice(ax, events, args.y_coord)
    if args.x_coord > 0:
        evlicious.art.plot_x_slice(ax, events, args.x_coord)

    plt.show()


    