import copy
from genericpath import isdir
import os
import shutil
import numpy as np
import utils.common as c
from utils.common import extract_newgem_raw_json
import utils.target_stats as t
import csv
import numpy as np
import argparse

import json
import numpy as np
import pandas as pd
import re

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib import ticker
from matplotlib.lines import Line2D

from cache_sensitive_names import *
from set_analyze.my_diff_color import *

parser = argparse.ArgumentParser(description="options to get set stats")
# parser.add_argument('-d','--stats_dir', type=str,
#     help='stats dir to analyze',required=True)
# parser.add_argument('--ids',default=16,type=int)
# parser.add_argument('--nsamples',default=2,type=int)
# parser.add_argument('--l3_sets',default=4096,type=int)

opt = parser.parse_args()

if __name__ == '__main__':
    ncore = 2
    use_conf = conf_50M
    test_prefix = use_conf['test_prefix']
    perf_prefix = '95perf'

    work0names = use_conf['cache_work_names']

