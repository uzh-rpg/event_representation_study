import argparse
from pathlib import Path
import evlicious
import ev_graph
import torch


def FLAGS():
    parser = argparse.ArgumentParser(description='''Generate event graph animation.''')
    parser = evlicious.args.add_io_args(parser)
    parser.add_argument('--height', type=int, help='Height.', default=-1)
    parser.add_argument('--width', type=int, help='Width.', default=-1)
    parser.add_argument('--downsample_events', type=int, help='Downsample number.', default=1)
    parser.add_argument('--radius', type=float, help='Downsample number.', default=3)
    parser.add_argument('--delta_t_us', type=int, help='Downsample number.', default=300000)
    parser.add_argument('--t_step_us', type=int, help='Downsample number.', default=3000)
    parser.add_argument('--max_num_neighbors', type=int, help='Downsample number.', default=16)

    parser.add_argument('--front_side', action="store_true", help='Front side projection of events.')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = FLAGS()
    args.width = args.width if args.width > 0 else None
    args.height = args.height if args.height > 0 else None
    events = evlicious.io.load_events_from_path(args.events, height=args.height, width=args.width)

    graph = ev_graph.AsyncGraph(width=events.width, 
                                height=events.height, 
                                delta_t_us=args.delta_t_us, 
                                radius=args.radius, 
                                max_num_neighbors=args.max_num_neighbors)    

    factor = args.radius / args.delta_t_us 
    visualizer = evlicious.art.EventGraphVisualizer(events)
    visualizer.run(args.output_path,
                   args.delta_t_us, 
                   args.t_step_us,
                   args.radius, 
                   args.max_num_neighbors,
                   args.downsample_events, 
                   args.front_side)

