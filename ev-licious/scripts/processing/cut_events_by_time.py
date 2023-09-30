import argparse

import evlicious


def FLAGS():
    parser = argparse.ArgumentParser("""Cuts events between t0_us and t1_us""")
    parser = evlicious.args.add_io_args(parser)

    parser.add_argument("--t0_us", type=int, default=-1)
    parser.add_argument("--t1_us", type=int, default=-1)
    
    flags = parser.parse_args()
    assert flags.events.exists()

    if str(flags.output_path) == ".":
        flags.output_path = flags.output_path / flags.events.name

    return flags


if __name__ == '__main__':
    flags = FLAGS()
    
    events = evlicious.io.load_events_from_path(flags.events)
    events = events.get_between_time(flags.t0_us, flags.t1_us)
    events.to(flags.output_path)
