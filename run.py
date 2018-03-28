import numpy as np

import tensorflow as tf
from config import get_config, print_usage
from model import Yolo
from utils.data import load_data


def main(config):
    """The main function."""

    # ----------------------------------------
    # Load pascal voc datasets

    print("\n--- Reading PASCAL {} data".format(config.year))
    dataset_train = load_data(config.data_dir, config.record_file, config.year,
                              'train')
    dataset_val = load_data(config.data_dir, config.record_file, config.year,
                            'val')

    yolo = Yolo(config, dataset_train, dataset_val)
    yolo.train()

    return


if __name__ == "__main__":

    # ----------------------------------------
    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)
