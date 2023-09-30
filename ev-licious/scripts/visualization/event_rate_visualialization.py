import argparse
import evlicious
import numpy as np
import matplotlib.pyplot as plt
from evlicious.io.utils.load_data import load_images, load_feature_tracks


def main():
    parser = argparse.ArgumentParser(description='''Generate 3D X-Y-T event plots and optionally also visualize feature tracks''')
    parser = evlicious.args.add_io_args(parser)
    parser.add_argument('--time_window_us', type=int, help="Time window in us", default=100000)
    parser.add_argument('--time_step_us', type=int, help="Time window in us", default=10000)
    args = parser.parse_args()

    event_handle = evlicious.io.load_events_from_path(args.events)
    _, (idx0, idx1) = event_handle.compute_time_and_index_windows(step_size=args.time_step_us, window=args.time_window_us, step_size_unit="us", window_unit="us")
    rate_mhz = (idx1 - idx0) / args.time_window_us

    fig, ax = plt.subplots()
    t0, t1 = event_handle.get_time_limits()
    timestamps = np.linspace(t0, t1, len(rate_mhz)) - t0
    ax.plot(timestamps, rate_mhz)
    ax.set_xlabel("Time [us]")
    ax.set_ylabel("Event Rate [MHz]")
    plt.show()

if __name__ == "__main__":
    main()

