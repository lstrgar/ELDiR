from argparse import ArgumentParser
from simulator.sim import forward_visualization

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('robots_file', type=str)
    parser.add_argument('outdir', type=str)
    args = parser.parse_args()
    forward_visualization(args.robots_file, args.outdir)