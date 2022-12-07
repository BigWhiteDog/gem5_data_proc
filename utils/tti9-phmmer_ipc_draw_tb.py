from asyncio import set_child_watcher
from operator import delitem
import os
import numpy as np
import utils.common as c
from utils.common import extract_samples_raw_json, multi_stats_lastn_factory
from utils.insert_avg_dict import insert_avg_dict
import utils.target_stats as t
import csv
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
    all_base  = '/nfs/home/zhangchuanqi/lvna/5g/ff-reshape/log/new_hw_test/period_hmmer_o3_0-period_hmmer_o3_3-period_hmmer_o2_0-period_hmmer_o2_2/2560kBLLC/9tti/try-tb'
    tb_base = os.path.join(all_base,'l3-nopart','l2-nopart')
    tb_bases = os.listdir(tb_base)
    last_nsamples = 8
    st_dict = {}
    for tb in tb_bases:
        if tb.startswith('l3-tb'):
            tb_path = os.path.join(tb_base,tb)
            # st_json = extract_samples_raw_json(tb_path)
            # with open(os.path.join(tb_path,'4period.json')) as f:
            #     st_json = json.load(f)
            st_json = extract_samples_raw_json(tb_path,last_nsamples=last_nsamples)
            with open(os.path.join(tb_path,f'{last_nsamples}period.json'),'r') as f:
                st_json = json.load(f)
            insert_avg_dict(st_json,last_nsamples=last_nsamples)
            tb_inc = tb.split('-')[3]
            if len(tb_inc) == 0:
                tb_inc = '1024'
            tb_freq = tb.split('-')[2]
            if len(tb_freq) == 0:
                tb_freq = '256'
            st_dict[tb_freq+'-'+tb_inc] = st_json
    st_dict.pop('256-1024',None)

    fig,ax = plt.subplots(2,2,sharex=True)
    fig.set_size_inches(16,9)
    sorted_keys = sorted(st_dict.keys(),key=lambda x: int(x.split('-')[1])/int(x.split('-')[0]),reverse=True)
    max_key = sorted_keys[0]
    
    sorted_keys = sorted_keys[1:]
    x = np.arange(len(sorted_keys))
    xlabels = [s.split('-')[1] for s in sorted_keys]
    width = 0.2
    for i in range(4):
        fy = i % 2
        fx = i // 2
        ax_bar = ax[fx,fy]
        ax_bar.set_ylabel('IPC speedup')
        ax_bar.set_ylim(0.75,1.25)
        ax_bar.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax_bar.set_xlabel('token bucket inc for low priority')
        ax_bar.set_xticks(x,xlabels)
        ax_bar.set_title(f'CPU{i} IPC speedup in two stages')
        speedup_s0 = [st_dict[k][f'avg_ipc{i}_stage0']/st_dict[max_key][f'avg_ipc{i}_stage0'] for k in sorted_keys]
        speedup_s1 = [st_dict[k][f'avg_ipc{i}_stage1']/st_dict[max_key][f'avg_ipc{i}_stage1'] for k in sorted_keys]
        rect0 = ax_bar.bar(x-width/2,speedup_s0,width,label='speedup in stage 0')
        rect1 = ax_bar.bar(x+width/2,speedup_s1,width,label='speedup in stage 1')
        ax_bar.legend()
    plt.tight_layout()
    plt.savefig(f'tti9_tb_ipc_speedup.png',dpi=300)