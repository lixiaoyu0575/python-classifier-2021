#!/usr/bin/env python

# Do *not* edit this script.

import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
from team_code3 import training_code
if __name__ == '__main__':
    # Parse arguments.
    if len(sys.argv) != 3:
        raise Exception('Include the data and model folders as arguments, e.g., python train_model.py data model.')

    data_directory = sys.argv[1]
    model_directory = sys.argv[2]

    # Run the training code.
    print('Running training code...')

    training_code(data_directory, model_directory)

    print('Done.')
