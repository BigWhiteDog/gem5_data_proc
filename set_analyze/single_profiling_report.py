from genericpath import isdir
import os
import numpy as np
import utils.common as c
from utils.common import extract_newgem_raw_json
import utils.target_stats as t
import csv
import numpy as np
import argparse
import math
import itertools

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

def draw_one_workload_ipc(ax,s_dicts,workload_names):
    sorted_keys = sorted(s_dicts.keys(),key=lambda x:int(x),reverse=True)
    ipcs = [s_dicts[k]['cpu.ipc'][0] for k in sorted_keys]
    max_ipc = ipcs[0]
    nipcs = [i/max_ipc for i in ipcs]
    ax.set_ylabel('Norm IPC')
    ax.set_ylim(0.5,1)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.set_xlabel('number of L3 ways (1MB/way)')
    ax.set_title(f'{workload_names}')
    ax.plot(sorted_keys,nipcs,marker='o',label='ipc')

    slowdown_2 = nipcs[-2]
    if (slowdown_2<0.95):
        # print(f'{workload_names} get slowdown {slowdown_2:.3} 2way')
        fulfill5_way = 8
        fulfill10_way = 8
        for ipc,way in zip(nipcs,sorted_keys):
            if ipc>0.95:
                fulfill5_way = int(way)
                fulfill10_way = int(way)
            elif ipc>0.9:
                fulfill10_way = int(way)
            else:
                # print(f'{fulfill5_way}way {ipc:.3} over 95% ipc')
                break
        print(f'{workload_names} {slowdown_2:.3} {fulfill5_way} {fulfill10_way}')

def draw_one_workload_missrate(ax,s_dicts,workload_names):
    sorted_keys = sorted(s_dicts.keys(),key=lambda x:int(x),reverse=True)
    missrates = [s_dicts[k]['l3.demandMissRate'][0] for k in sorted_keys]
    ax.set_ylabel('l3 missrate')
    ax.set_ylim(0,1)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.set_xlabel('number of L3 ways (1MB/way)')
    ax.set_title(f'{workload_names}')
    ax.plot(sorted_keys,missrates,marker='o',label='ipc')

def draw_by_func(base_dir,n_rows,worksname,draw_one_func,fig_name):
    fig,ax = plt.subplots(n_rows,4)
    fig.set_size_inches(24, 4*n_rows)
    for i,work in enumerate(worksname):
        word_dir = os.path.join(base_dir,work)
        if not os.path.isdir(word_dir):
            continue
        fy = i % 4
        fx = i // 4
        ax_bar = ax[fx,fy]
        s_dicts = {}
        partsname = os.listdir(word_dir) #like l3-1
        for part in partsname:
            if not os.path.isdir(os.path.join(word_dir,part)):
                continue
            ways = part.split('-')
            if ways[0] != 'l3':
                continue
            new_base = os.path.join(word_dir,part)
            last_nsamples=1
            # one_dict = extract_newgem_raw_json(new_base,ncores=1,last_nsamples=last_nsamples)
            with open(os.path.join(new_base,f'{last_nsamples}period.json'),'r') as f:
                one_dict = json.load(f)
            # ways[1] as key
            s_dicts[ways[1]] = one_dict
        draw_one_func(ax_bar,s_dicts,work)

    plt.tight_layout()
    plt.savefig(fig_name,dpi=300)
    plt.clf()



if __name__ == '__main__':
    base_dir = '/nfs/home/zhangchuanqi/lvna/for_xs/catlog/single-profiling/'
    n_works = 53
    n_rows = math.ceil(n_works/4)
    worksname = os.listdir(base_dir) #like omnetpp
    worksname.sort()
    draw_by_func(base_dir,n_rows,worksname,
        draw_one_func=draw_one_workload_ipc,fig_name='setconf_single_profiling_ipc.png')
    draw_by_func(base_dir,n_rows,worksname,
        draw_one_func=draw_one_workload_missrate,fig_name='setconf_single_profiling_mr.png')