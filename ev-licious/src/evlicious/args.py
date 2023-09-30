import evlicious
from pathlib import Path


def add_io_args(parser):
    parser.add_argument('events', help='Root of folder with events.', type=Path, default="")
    parser.add_argument("--output_path", type=Path, default="")
    return parser 

def add_filter_args(parser):
    filtering_types = evlicious.tools.filters.Filtering_Type.summary()
    parser.add_argument("--filter_type", type=int, default=-1, help=f"Has to be one of the following: {filtering_types}.")

    # background activity filter
    parser.add_argument("--depth_us", type=int, default=-1)
    parser.add_argument("--radius", type=int, default=-1)

    # ContrastThresholdIncrease
    parser.add_argument("--contrast_threshold_multiplier", type=int, default=-1)

    # random
    parser.add_argument("--random_downsampling_factor", type=int, default=-1)

    return parser