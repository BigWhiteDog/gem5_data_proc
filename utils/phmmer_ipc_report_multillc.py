import os
import re
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

def extract_waymask_raw_json(base_dir,last_nsamples = 4):
    stat_dict = {}
    partsname = os.listdir(base_dir) #like l3-0x1-0xfe-0xfe-0xfe
    for part in partsname:
        if not os.path.isdir(os.path.join(base_dir,part)):
            continue
        ways = part.split('-')
        if ways[0] != 'l3':
            continue
        new_base = os.path.join(base_dir,part,"l2-nopart")
        # one_dict = extract_samples_raw_json(new_base,ncores=4,last_nsamples=last_nsamples)
        with open(os.path.join(new_base,f"{last_nsamples}period.json")) as f:
            one_dict = json.load(f)
        # ways[1] as key
        stat_dict[ways[1]] = one_dict
    return stat_dict


if __name__ == '__main__':
    # t_work_combine = ['hmmer_o31-hmmer0-hmmer_o30-hmmer1']
    top_base_dir = '/nfs/home/zhangchuanqi/lvna/5g/ff-reshape/log/new_hw_test/period_hmmer_o3_0-period_hmmer_o3_3-period_hmmer_o2_0-period_hmmer_o2_2'
    size_dict = {}
    for llcsize in os.listdir(top_base_dir):
        if not os.path.isdir(os.path.join(top_base_dir,llcsize)):
            continue
        size_dict[llcsize] = extract_waymask_raw_json(os.path.join(top_base_dir,llcsize,'try-waymask'),last_nsamples=4)
    size_re = re.compile(r'(\d+)kB')
    sorted_size = sorted(size_dict.keys(),key=lambda x:int(size_re.match(x).group(1)))
    way_stat_maxipc_size_dict = {}
    for s in sorted_size:
        way_stat_maxipc_size_dict[s] = max(size_dict[s].items(),key=lambda x: np.average(x[1]['cpu0.ipc']))

    fig,ax = plt.subplots(2,6,sharex=True)
    fig.set_size_inches(48,9)

    x = np.arange(len(sorted_size))
    xlabels = [ size_re.match(s).group(0) for s in sorted_size]
    for i in range(4):
        # fy = (i % 2) * 3
        # fx = i // 2
        fy = (i // 2) * 3
        fx = i % 2
        ax_ipc_bar = ax[fx,fy]
        width = 0.35
        nopart_ipc_iter = map(lambda x: np.average(size_dict[x]['nopart'][f'cpu{i}.ipc']) ,sorted_size)
        max_ipc_iter = map(lambda x: np.average(way_stat_maxipc_size_dict[x][1][f'cpu{i}.ipc']) ,sorted_size)
        #draw ipc bar of different size
        rect0 = ax_ipc_bar.bar(x-width/2, list(nopart_ipc_iter) , width, label='native mix ipc')
        rect1 = ax_ipc_bar.bar(x+width/2, list(max_ipc_iter), width, label='ipc when core0 reach max')
        ax_ipc_bar.set_xticks(x,xlabels)
        ax_ipc_bar.set_ylabel('average IPC')
        ax_ipc_bar.set_ylim(1.4,1.9)
        ax_ipc_bar.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax_ipc_bar.set_title(f'average IPC of core{i}')
        ax_ipc_bar.legend()
        # ax_ipc_bar.bar_label(rect0, padding=3)
        # ax_ipc_bar.bar_label(rect1, padding=3)
        #draw speedup bar of different size
        ax_speedup_bar = ax[fx,fy+1]
        nopart_ipc_iter = map(lambda x: np.average(size_dict[x]['nopart'][f'cpu{i}.ipc']) ,sorted_size)
        max_ipc_iter = map(lambda x: np.average(way_stat_maxipc_size_dict[x][1][f'cpu{i}.ipc']) ,sorted_size)
        rect3 = ax_speedup_bar.bar(x, list(map(lambda x: x[0]/x[1], zip(max_ipc_iter,nopart_ipc_iter))) , width, label='speedup')
        ax_ipc_bar.set_xticks(x,xlabels)
        ax_speedup_bar.set_ylabel('speedup')
        ax_speedup_bar.set_ylim(0.85,1.15)
        ax_speedup_bar.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax_speedup_bar.set_title(f'speedup of average IPC of core{i}')
        #draw miss rate bar of different size
        ax_missrate_bar = ax[fx,fy+2]
        nopart_missrate_iter = map(lambda x: np.average(size_dict[x]['nopart'][f'l3.demand_miss_rate::.cpu{i}']) ,sorted_size)
        max_missrate_iter = map(lambda x: np.average(way_stat_maxipc_size_dict[x][1][f'l3.demand_miss_rate::.cpu{i}']) ,sorted_size)
        rect4 = ax_missrate_bar.bar(x-width/2, list(nopart_missrate_iter) , width, label='native mix miss rate')
        rect5 = ax_missrate_bar.bar(x+width/2, list(max_missrate_iter), width, label='miss rate when core0 reach max')
        ax_missrate_bar.set_xticks(x,xlabels)
        ax_missrate_bar.set_ylabel('average miss rate')
        ax_missrate_bar.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
        ax_missrate_bar.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax_missrate_bar.set_title(f'average miss rate of core{i}')
        ax_missrate_bar.legend()
        # ax_missrate_bar.bar_label(rect4, padding=3)
        # ax_missrate_bar.bar_label(rect5, padding=3)

        plt.setp(ax_ipc_bar.get_xticklabels(), rotation=30, ha="right",
                rotation_mode="anchor")
        plt.setp(ax_speedup_bar.get_xticklabels(), rotation=30, ha="right",
                rotation_mode="anchor")
        plt.setp(ax_missrate_bar.get_xticklabels(), rotation=30, ha="right",
                rotation_mode="anchor")

    # ax[0,1].legend(loc='upper right',ncol=1,fontsize=10,bbox_to_anchor=(1.01,1.01))
    plt.savefig(f'multi_llc_speedup.png',dpi=300)