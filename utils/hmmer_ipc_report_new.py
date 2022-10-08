import os
import utils.common as c
from utils.common import extract_samples_raw_json
import numpy as np
import argparse

import json

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib import ticker

json_path = "/nfs/home/zhangchuanqi/lvna/5g/DirtyStuff/resources/simpoint_cpt_desc/hwfinal.json"

parser = argparse.ArgumentParser(description="options to get set stats")
# parser.add_argument('-d','--stats_dir', type=str,
#     help='stats dir to analyze',required=True)
# parser.add_argument('--ids',default=16,type=int)
# parser.add_argument('--nsamples',default=2,type=int)
# parser.add_argument('--l3_sets',default=4096,type=int)

opt = parser.parse_args()

mycolor = ['#044E48','#06746B','#20876B','#6A9A48','#B5AC23','#E6B80B','#FACA3E','#FFDF80','#FFEBB0']
mycolor = ['#661900','#B22C00','#E6450F','#FF6500','#FF8C00','#FFB200','#FFCB33','#FFDF80','#FDEDBE']


if __name__ == '__main__':
    # t_work_combine = ['hmmer_o31-hmmer0-hmmer_o30-hmmer1']
    # report_hmmer('/nfs/home/zhangchuanqi/lvna/5g/ff-reshape/log/new_hw_test/16M/hmmer_o31-hmmer0-hmmer_o30-hmmer1')
    all_base  = '/nfs/home/zhangchuanqi/lvna/5g/ff-reshape/log/new_hw_test/period4/try-tb/period_hmmer_o3_0-period_hmmer_o3_3-period_hmmer_o2_0-period_hmmer_o2_2'
    tb_base = os.path.join(all_base,'l3-nopart','l2-nopart')
    tb_bases = os.listdir(tb_base)
    for tb in tb_bases:
        if tb.startswith('l3-tb'):
            tb_path = os.path.join(tb_base,tb)
            extract_samples_raw_json(tb_path)
        # print(tb_path)