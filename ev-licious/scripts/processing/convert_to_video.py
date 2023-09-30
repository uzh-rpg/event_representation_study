import argparse

import evlicious
import e2vid
import cv2


def FLAGS():
    parser = argparse.ArgumentParser("""Converts events to e2vid reconstruction.""")

    parser = evlicious.args.add_io_args(parser)
    parser = evlicious.args.add_filter_args(parser)

    parser = e2vid.options.inference_options.set_inference_options(parser)

    parser.add_argument("--window", type=int, default=20000, help="Window of events passed to e2vid.")
    parser.add_argument("--window_unit", type=str, default="us", help="Can be either 'nr' or 'us'")
    parser.add_argument("--clahe_tileGridSize", type=int, default=50, help="CLAHE argument")
    parser.add_argument("--clahe_clipLimit", type=float, default=2.0, help="CLAHE argument")

    flags = parser.parse_args()
    assert flags.events.exists()

    if str(flags.output_path) == ".":
        flags.output_path = flags.output_path / flags.events.name

    return flags


if __name__ == '__main__':
    flags = FLAGS()
    
    events = evlicious.io.load_events_from_path(flags.events)
    #filter = evlicious.tools.filters.from_flags(flags)

    # e2vid flags expect to know the shape of the sensor
    flags.height = events.height
    flags.width = events.width

    model = e2vid.E2VID(flags)

    if not flags.output_path.exists():
        flags.output_path.mkdir(exist_ok=True)

    clahe = None
    #if flags.clahe_clipLimit > 0:
    #    clahe = cv2.createCLAHE(clipLimit=flags.clahe_clipLimit, tileGridSize=(flags.clahe_tileGridSize, flags.clahe_tileGridSize))

    counter = 0
    timestamps_file = flags.output_path / "timestamps.txt"
    if timestamps_file.exists():
        timestamps_file.remove()

    for ev in events.iterator(step_size=flags.window, window=flags.window, window_unit=flags.window_unit, step_size_unit=flags.window_unit, pbar=True):
        t_event = ev.t[-1]
        #ev = filter.insert(ev)

        voxel_grid = evlicious.tools.events_to_voxel_grid_cuda(ev, num_bins=5, normalize=not flags.no_normalize, device="cuda:0")
        reconstruction = model(voxel_grid)

        if clahe is not None:
            reconstruction = clahe.apply(reconstruction)

        if False:
            output_path = flags.output_path / ("%05d.png" % counter)
            cv2.imwrite(str(output_path), reconstruction)
            counter += 1

            with open(str(timestamps_file), "a") as fh:
                fh.write(f"{t_event}\n")
