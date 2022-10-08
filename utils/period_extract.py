import os
import numpy as np
import utils.common as c
from utils.common import multi_stats_lastn_factory
import utils.target_stats as t
import csv

import numpy as np
import argparse

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib import ticker

import json

json_path = "/nfs/home/zhangchuanqi/lvna/5g/DirtyStuff/resources/simpoint_cpt_desc/hwfinal.json"

parser = argparse.ArgumentParser(description="options to get set stats")
# parser.add_argument('-d','--stats_dir', type=str,
#     help='stats dir to analyze',required=True)
# parser.add_argument('--ids',default=16,type=int)
# parser.add_argument('--nsamples',default=2,type=int)
# parser.add_argument('--l3_sets',default=4096,type=int)

opt = parser.parse_args()



def extract_period(stats_dir,last_nsamples = 160):
    stat_dict = {}
    st_file = os.path.join(stats_dir,"stats.txt")
    st_filter_file = os.path.join(stats_dir,"period.csv")
    target_keys = ['cpu.ipc','l3.tags.tag_accesses','l3.demand_miss_rate','l3.demand_hits','l3.demand_misses']
    stats_get_func = multi_stats_lastn_factory(t.llc_new_targets, target_keys,last_n=last_nsamples)
    stat_dict = stats_get_func(st_file)
    # with open(st_filter_file, 'w') as testfile:
    #     writer = csv.writer(testfile,target_keys)
    #     writer.writerow(target_keys)
    #     writer.writerows(zip(*[stat_dict[key] for key in target_keys]))
    return stat_dict

if __name__ == '__main__':
    worknames = ['hmmer_o30','hmmer_o31','hmmer_o2_retro0','hmmer_o2_retro1','hmmer0','hmmer1','hmmer2']
    # worknames = [f'spa{i}' for i in range(2)] + \
    #             [f'uaa{i}' for i in range(2)] + \
    #             [f'epa{i}' for i in range(3)] + \
    #             [f'cga{i}' for i in range(1)]
    # worknames = [f'isa{i}' for i in range(4)] + \
    #             [f'mga{i}' for i in range(1)] + \
    #             [f'bta{i}' for i in range(3)]
    # worknames = [f'lua{i}' for i in range(1)] + \
    #             [f'fta{i}' for i in range(3)]
    # ['spa0','spa1','uaa0','uaa1','epa0','epa1','bta0','lua0','fta0']
    with open(json_path) as json_file:
        workload_dict = json.load(json_file)
    
    s_dicts = {}
    for w in worknames:
        base_dir = f'/nfs/home/zhangchuanqi/lvna/5g/ff-reshape/log/16M/{w}/100000'
        s_dicts[w] = extract_period(base_dir)
    
    fig, ax = plt.subplots(6,4,sharex=True)
    xvals = np.arange(0,8,1/20)
    for i,work in enumerate(worknames):
        fy = i % 4
        fx = (i // 4) * 3
        ax[fx,fy].xaxis.set_major_locator(ticker.MultipleLocator(1))
        if fx == 3:
            ax[fx+2,fy].set_xlabel('time(ms)')
        ax[fx,fy].set_ylim(1,2.5)
        ax[fx,fy].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        if fy == 0:
            ax[fx,fy].set_ylabel('ipc')
        ax[fx,fy].plot(xvals,s_dicts[work]['cpu.ipc'],label=work,linewidth=1.5)
        ax[fx,fy].set_title(work)
        ax[fx+1,fy].set_ylim(0,25)
        ax[fx+1,fy].yaxis.set_major_locator(ticker.MultipleLocator(5))
        if fy == 0:
            ax[fx+1,fy].set_ylabel('l3 accesses BW\n(MB/ms)')
        ta = np.array(s_dicts[work]['l3.tags.tag_accesses'])
        bw = ta * 1.28 / 1000
        ax[fx+1,fy].plot(xvals,bw,label=work,linewidth=1.5)
        ax[fx+2,fy].set_ylim(0,1)
        if fy == 0:
            ax[fx+2,fy].set_ylabel('l3 miss rate')
        total_hits = np.sum(s_dicts[work]['l3.demand_hits'])
        total_misses = np.sum(s_dicts[work]['l3.demand_misses'])
        total_miss_rate = total_misses / (total_hits + total_misses)
        ax[fx+2,fy].set_title(f'totalMissRate:{total_miss_rate:.1%}',fontsize=12)
        ax[fx+2,fy].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1,decimals=0))
        ax[fx+2,fy].yaxis.set_major_locator(ticker.MultipleLocator(0.25))
        ax[fx+2,fy].bar(xvals,
            s_dicts[work]['l3.demand_miss_rate'],width=0.04,align="edge",label=work)
    # plt.delaxes(ax[3,3])
    # plt.delaxes(ax[4,3])
    # plt.delaxes(ax[5,3])
    plt.show()