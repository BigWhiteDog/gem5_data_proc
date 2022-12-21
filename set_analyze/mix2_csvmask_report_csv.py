from genericpath import isdir
import os
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

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib import ticker

from cache_sensitive_names import *
from set_analyze.my_diff_color import *

parser = argparse.ArgumentParser(description="options to get set stats")
# parser.add_argument('-d','--stats_dir', type=str,
#     help='stats dir to analyze',required=True)
# parser.add_argument('--ids',default=16,type=int)
# parser.add_argument('--nsamples',default=2,type=int)
# parser.add_argument('--l3_sets',default=4096,type=int)

opt = parser.parse_args()

def analyze_one_workload_dict(base_dir,workload_names,ncore=2):
    s_dicts = {}
    partsname = os.listdir(base_dir) #like l3-1,l3-nopoart,l3-1-csv
    for part in partsname:
        if not os.path.isdir(os.path.join(base_dir,part)):
            continue
        ways = part.split('-')
        if ways[0] != 'l3':
            continue
        new_base = os.path.join(base_dir,part)
        last_nsamples=1
        one_dict = extract_newgem_raw_json(new_base,ncores=ncore,last_nsamples=last_nsamples)
        with open(os.path.join(new_base,f'{last_nsamples}period.json'),'r') as f:
            one_dict = json.load(f)
        # ways[-1] as key
        dkey = ways[-1]
        if dkey.isnumeric():
            s_dicts['core0_setway'] = dkey
            s_dicts['static_waymask'] = one_dict
        else:
            s_dicts[dkey] = one_dict

    pd_dict = {}
    pd_dict['workload'] = workload_names
    pd_dict['core0_setway'] = s_dicts['core0_setway']
    for c in range(ncore):
        pd_dict[f'nopart_cpu{c}.ipc'] = s_dicts['nopart'][f'cpu{c}.ipc'][0]
        pd_dict[f'static_cpu{c}.ipc'] = s_dicts['static_waymask'][f'cpu{c}.ipc'][0]
        pd_dict[f'csv_cpu{c}.ipc'] = s_dicts['csv'][f'cpu{c}.ipc'][0]
        pd_dict[f'csv_speedup_cpu{c}'] = pd_dict[f'csv_cpu{c}.ipc']/pd_dict[f'static_cpu{c}.ipc']

    return pd_dict


if __name__ == '__main__':
    ncore = 2
    use_conf = conf_50M
    test_prefix = use_conf['test_prefix']
    perf_prefix = '95perf'

    base_dir = f'/nfs/home/zhangchuanqi/lvna/for_xs/catlog/mix2-csvpolicy-core0-{test_prefix}{perf_prefix}'
    worksname = os.listdir(base_dir) #like omnetpp-xalancbmk
    pd_dict_list = []
    for i,work in enumerate(worksname):
        word_dir = os.path.join(base_dir,work)
        if not os.path.isdir(word_dir):
            continue
        pd_dict = analyze_one_workload_dict(word_dir,work,ncore=ncore)
        pd_dict_list.append(pd_dict)

    pd_dict_list = sorted(pd_dict_list,key=lambda x:x['csv_speedup_cpu1'],reverse=True)

    mycolumns = ['workload','core0_setway']
    c_formats = [
        'nopart_cpu{}.ipc',
        'static_cpu{}.ipc',
        'csv_cpu{}.ipc',
        'csv_speedup_cpu{}'
    ]
    for cf in c_formats:
        for c in range(ncore):
            mycolumns.append(cf.format(c))

    df = pd.DataFrame(pd_dict_list,columns=mycolumns)
    df.to_csv('mix2_csvmask_report.csv',index=False)
