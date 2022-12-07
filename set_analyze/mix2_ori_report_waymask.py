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

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib import ticker


parser = argparse.ArgumentParser(description="options to get set stats")
# parser.add_argument('-d','--stats_dir', type=str,
#     help='stats dir to analyze',required=True)
# parser.add_argument('--ids',default=16,type=int)
# parser.add_argument('--nsamples',default=2,type=int)
# parser.add_argument('--l3_sets',default=4096,type=int)

opt = parser.parse_args()

mycolor = ['#044E48','#06746B','#20876B','#6A9A48','#B5AC23','#E6B80B','#FACA3E','#FFDF80','#FFEBB0']
mycolor = ['#661900','#B22C00','#E6450F','#FF6500','#FF8C00','#FFB200','#FFCB33','#FFDF80','#FDEDBE']

def draw_one_workload_waymaskbar(ax_bar,base_dir,workload_names,ncore=2,draw_best_n=4):
    s_dicts = {}
    partsname = os.listdir(base_dir) #like l3-1
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
        # ways[1] as key
        s_dicts[ways[1]] = one_dict
    nopart_ipcs = [s_dicts['nopart'][f'cpu{i}.ipc'][0] for i in range(ncore)]
    speedup_dict = {}
    for k in s_dicts:
        if k == 'nopart':
            continue
        speedup_dict[k] = {}
        all_speedup_list = []
        for i in range(ncore):
            s = s_dicts[k][f'cpu{i}.ipc'][0]/nopart_ipcs[i]
            speedup_dict[k][i] = s
            all_speedup_list.append(s)
        speedup_dict[k]['avg'] = np.mean(all_speedup_list)
    sorted_keys = sorted(speedup_dict.keys(),key=lambda x:speedup_dict[x]['avg'],reverse=True)
    interesting_keys = sorted_keys[:3]
    x = np.arange(len(interesting_keys))
    width = 0.2
    ax_bar.set_ylabel('IPC speedup')
    ax_bar.set_ylim(0.8,1.2)
    ax_bar.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax_bar.set_xlabel('number of L3 ways for core0')
    ax_bar.set_xticks(x,interesting_keys)
    ax_bar.set_title(f'{workload_names}')
    for i in range(ncore):
        y = [speedup_dict[k][i] for k in interesting_keys]
        ax_bar.bar(x+i*width,y,width,label=f'core{i}')
    ax_bar.bar(x+ncore*width,[speedup_dict[k]['avg'] for k in interesting_keys],width,label='avg')
    ax_bar.legend()

if __name__ == '__main__':
    base_dir = '/nfs/home/zhangchuanqi/lvna/for_xs/catlog/mix2-nochange/'
    n_works = 12
    fig,ax = plt.subplots(3,4)
    fig.set_size_inches(24,12)
    worksname = os.listdir(base_dir) #like omnetpp-xalancbmk
    for i,work in enumerate(worksname):
        word_dir = os.path.join(base_dir,work)
        if not os.path.isdir(word_dir):
            continue
        fy = i % 4
        fx = i // 4
        ax_bar = ax[fx,fy]
        draw_one_workload_waymaskbar(ax_bar,word_dir,work)


    plt.tight_layout()
    plt.savefig(f'setconf_mix2_ori.png',dpi=300)