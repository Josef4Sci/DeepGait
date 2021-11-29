from myUtils import TensorboardLogger
import argparse
from packaging import version

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb
# from tensorflow.python.summary.summary_iterator import summary_iterator


# Loook for the project DataLogPostprocess with tensorflow!!!!!

if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False,description='Process some integers.')
    parser.add_argument('--logPath', help='Array of integers')

    args = parser.parse_args()

    major_ver, minor_ver, _ = version.parse(tb.__version__).release
    assert major_ver >= 2 and minor_ver >= 3, \
        "This notebook requires TensorBoard 2.3 or later."
    print("TensorBoard version: ", tb.__version__)



