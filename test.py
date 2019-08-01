
import sys
import argparse

from lib.utils.config_parse import cfg_from_file
from lib.ssds_train import test_model

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a ssds.pytorch network')
    parser.add_argument('--cfg', dest='config_file',
            help='optional config file', default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def test():
    args = parse_args()
    if args.config_file is not None:
        cfg_from_file(args.config_file)
    test_model()

if __name__ == '__main__':
    test()
