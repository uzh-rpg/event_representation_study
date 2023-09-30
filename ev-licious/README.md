## Install

To install type

    conda env create -f environment.yml
    conda activate evlicious
    pip install ev-licious/
    cd ev-licious/

Optionally, to visualize event graphs also install the [ev_graph](https://github.com/uzh-rpg/ev_graph) package.

## Conversion

To convert datasets use

    python scripts/conversion/convert_npz_to_standard_format.py path/to/folder/ --divider 1 --height 720 --width 1280 --output path/to/output.h5

    python scripts/conversion/convert_to_standard_format.py path/to/folder/ --recursive --divider 1 --height 720 --width 1280 --output path/to/output.h5

Here only use `divider`>1 if the x and y were rescaled by a factor to remove decimal points. Most datasets use `divider`=1.

## Processing

To cut events between two times run 

    python scripts/processing/cut_events_by_time.py path/to/events.h5 --output_path path/to/output.h5 --t0_us time0 --t1_us time1

To convert events to E2VID frame reconstructions, run 

    python convert_to_video.py path/to/events.h5 --window 10000 --window_unit us --output_path path/to/output/

This script requires the `e2vid` package, which can be installed on [this page](https://github.com/uzh-rpg/e2calib).

## Visualization

Test the visualization scripts with

    python scripts/visualization/2d_coordinate_slice_events.py tests/data/events.h5 --y_coord 120

    python scripts/visualization/3d_visualization_events.py tests/data/events.h5 --factor 0.002 --images tests/data/images/ --downsample_images 30

    python scripts/visualization/interactive_viz_events.py tests/data/events.h5

    python scripts/visualization/3d_animation_events_graph.py tests/data/events.h5 --delta_t_us 20000 --radius 17

    python scripts/visualization/half_half_rendering_events.py --images tests/data/art/images --events tests/data/art/events.h5
