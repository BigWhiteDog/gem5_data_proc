from genericpath import isdir
import os
import numpy as np
import utils.common as c
from utils.common import extract_samples_raw_json
import utils.target_stats as t
import csv
import numpy as np
import argparse

import json

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib import ticker

from utils.insert_avg_dict import insert_avg_dict

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
    base_dir = '/nfs/home/zhangchuanqi/lvna/5g/ff-reshape/log/new_hw_test/period_hmmer_o3_0-period_hmmer_o3_3-period_hmmer_o2_0-period_hmmer_o2_2/2560kBLLC/9tti/try-waymask'
    last_nsamples = 8
    fig,ax = plt.subplots(2,2)
    fig.set_size_inches(16,9)
    s_dicts = {}
    partsname = os.listdir(base_dir) #like l3-1
    for part in partsname:
        if not os.path.isdir(os.path.join(base_dir,part)):
            continue
        ways = part.split('-')
        if ways[0] != 'l3':
            continue
        new_base = os.path.join(base_dir,part,"l2-nopart")
        one_dict = extract_samples_raw_json(new_base,ncores=4,last_nsamples=last_nsamples)
        with open(os.path.join(new_base,f'{last_nsamples}period.json'),'r') as f:
            one_dict = json.load(f)
        insert_avg_dict(one_dict,last_nsamples=last_nsamples)
        # ways[1] as key
        s_dicts[ways[1]] = one_dict

    sorted_keys = sorted(filter(lambda s:s.isnumeric(),s_dicts.keys()),key=lambda x:int(x))
    interesting_keys = sorted_keys[:]
    x = np.arange(len(interesting_keys))
    width = 0.2
    for i in range(4):
        fy = i % 2
        fx = i // 2
        ax_bar = ax[fx,fy]
        ax_bar.set_ylabel('IPC speedup')
        ax_bar.set_ylim(0.85,1.15)
        ax_bar.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax_bar.set_xlabel('number of L3 ways for high priority')
        ax_bar.set_xticks(x,interesting_keys)
        ax_bar.set_title(f'CPU{i} IPC speedup in two stages')
        speedup_s0 = [s_dicts[k][f'avg_ipc{i}_stage0']/s_dicts['nopart'][f'avg_ipc{i}_stage0'] for k in interesting_keys]
        speedup_s1 = [s_dicts[k][f'avg_ipc{i}_stage1']/s_dicts['nopart'][f'avg_ipc{i}_stage1'] for k in interesting_keys]
        rect0 = ax_bar.bar(x-width/2,speedup_s0,width,label='speedup in stage 0')
        rect1 = ax_bar.bar(x+width/2,speedup_s1,width,label='speedup in stage 1')
        ax_bar.legend()

    # ax[0,1].legend(loc='upper right',ncol=1,fontsize=10,bbox_to_anchor=(1.01,1.01))
    plt.tight_layout()
    plt.savefig(f'tti9_25mllc_speedup.png',dpi=300)