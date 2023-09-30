import argparse
import evlicious


def FLAGS():
    parser = argparse.ArgumentParser("""Filters events""")
    parser = evlicious.args.add_io_args(parser)
    parser = evlicious.args.add_filter_args(parser)

    flags = parser.parse_args()
    assert flags.events.exists()

    if str(flags.output_path) == ".":
        flags.output_path = flags.output_path / flags.events.name

    return flags


if __name__ == '__main__':
    flags = FLAGS()
    events = evlicious.io.load_events_from_path(flags.events)

    filter = evlicious.tools.filters.from_flags(flags)

    writer = evlicious.io.H5Writer(flags.output_path)
    for ev in events.iterator(step_size=1000000, step_size_unit="nr", window=1000000, window_unit="nr", pbar=True):
        ev_filtered = filter.insert(ev)
        writer.add_data(ev_filtered)

    idx = 1000000 * (len(events) // 1000000)
    ev = events.get_between_idx(idx, len(events))
    ev_filtered = filter.insert(ev)
    writer.add_data(ev_filtered)
