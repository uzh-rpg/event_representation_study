import argparse
import evlicious
from evlicious.io.utils.load_data import load_images, load_feature_tracks


def main():
    parser = argparse.ArgumentParser(description='''Generate 3D X-Y-T event plots and optionally also visualize feature tracks''')
    parser = evlicious.args.add_io_args(parser)
    parser.add_argument('--images', help='Image folder.', default="")
    parser.add_argument('--tracks', help='Path to csv of feature tracks.', default="")
    parser.add_argument('--downsample_images', type=int, help='Downsample images by.', default=1)
    parser.add_argument('--downsample_events', type=int, help='Downsample events by.', default=1)
    parser.add_argument('--time_scaling_factor', type=float, help='Scale time in us by this factor.', default=0.02)
    parser.add_argument('--start_time', type=int, help='Render events from time.', default=-1)
    parser.add_argument('--end_time', type=int, help='Render events to time.', default=-1)
    parser.add_argument('--loop', action="store_true")
    parser.add_argument('--time_step_us', type=int, help="Time step each iteration, in us", default=1000)
    parser.add_argument('--time_window_us', type=int, help="Time window in us", default=100000)
    args = parser.parse_args()

    event_handle = evlicious.io.load_events_from_path(args.events)

    t0, t1 = event_handle.get_time_limits()
    args.start_time = t0 if args.start_time < 0 else args.start_time
    args.end_time = t1 if args.end_time < 0 else args.end_time
    events = event_handle.get_between_time(args.start_time, args.end_time)
    events = events[::args.downsample_events]
    print("Event timestamps: ", events.t[0], events.t[-1])

    # if images are given, load them
    image_data = load_images(args.images, args.downsample_images)
    tracks_data = load_feature_tracks(args.tracks)

    evlicious.art.visualize_3d(events, time_window_us=args.time_window_us,
                               time_step_us=args.time_step_us,
                               images=image_data,
                               factor=args.time_scaling_factor,
                               tracks=tracks_data,
                               output_path=args.output_path,
                               loop=args.loop)


if __name__ == "__main__":
    main()

