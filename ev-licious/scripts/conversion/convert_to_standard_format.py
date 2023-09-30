import argparse
import evlicious
from pathlib import Path


def FLAGS():
    parser = argparse.ArgumentParser("""Converts .raw, .npz files to a .h5""")
    parser = evlicious.args.add_io_args(parser)
    parser.add_argument("--width", default=-1, type=int)
    parser.add_argument("--height", default=-1, type=int)
    parser.add_argument("--divider", default=1, type=int)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--recursive", action="store_true")
    flags = parser.parse_args()
    assert flags.events.exists()

    if str(flags.output_path) == ".":
        flags.output_path = Path(str(flags.output_path / flags.events.name) + ".h5")

    return flags


def convert(f, output_path, height, width, divider):
    handle = evlicious.io.load_events_from_path(f, height=height, width=width, divider=divider)
    handle.to(output_path)


if __name__ == '__main__':
    flags = FLAGS()

    if flags.events.is_dir() and flags.recursive:
        assert flags.suffix != ""
        files = list(flags.events.rglob(f"*.{flags.suffix}"))
        with evlicious.tools.TaskManager(total=len(files)) as tm:
            for f in files:
                output_path = flags.output_path / f.relative_to(flags.events)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path = Path(str(output_path).replace(f".{flags.suffix}", ".h5"))
                tm.new_task(convert, f, output_path, flags.height, flags.width, flags.divider)
    else:
        convert(flags.events, flags.output_path, flags.height, flags.width, flags.divider)


