import os
import numpy as np
import utils.common as c
from utils.common import multi_stats_lastn_factory
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
        ways = part.split('-')
        if ways[0] != 'l3':
            continue
        st_file = os.path.join(base_dir,part,"l2-nopart","stats.txt")
        st_filter_file = os.path.join(base_dir,part,"l2-nopart",f"{last_nsamples}period.json")
        target_keys = [f'cpu{i}.ipc' for i in range(4)]+['l3.tags.tag_accesses','l3.demand_miss_rate','l3.demand_hits','l3.demand_misses']
        stats_get_func = multi_stats_lastn_factory(t.llc_new_targets, target_keys,last_n=last_nsamples)
        one_dict = stats_get_func(st_file)
        with open(st_filter_file, 'w') as testfile:
            json.dump(one_dict, testfile, indent=4)
        # ways[1] as key
        stat_dict[ways[1]] = one_dict
    return stat_dict

def report_hmmer(base_dir,last_nsamples = 4):
    fig,ax = plt.subplots(2,2,sharex=True)
    fig.set_size_inches(16,9)
    worknames = os.path.basename(os.path.normpath(base_dir))
    works = worknames.split('-')
    assert(len(works) == 4)
    s_dicts = extract_waymask_raw_json(base_dir,last_nsamples=last_nsamples)
    partsname = os.listdir(base_dir) #like l3-0x1-0xfe-0xfe-0xfe
    parts_dict = {}
    for part in partsname:
        ways = part.split('-')
        if ways[0] != 'l3':
            continue
        parts_dict[ways[1]] = partsname
    sorted_keys = sorted(list(filter(lambda x: x.startswith('0'), parts_dict.keys())), key=lambda x: int(x, base=16),reverse=True)
    # sorted_keys.insert(0,'nopart')
    xvals = np.arange(0,8,2)
    for i in range(4):
        fy = i % 2
        fx = i // 2
        ax[fx,fy].xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax[fx,fy].set_xlabel('Time (ms)')
        ax[fx,fy].set_ylabel('IPC speedup')
        ax[fx,fy].set_ylim(0.85,1.15)
        ax[fx,fy].yaxis.set_major_locator(ticker.MultipleLocator(0.05))
        for ci,k in enumerate(sorted_keys[:6]):
            sample_ipc = np.array(s_dicts[k][f'cpu{i}.ipc'])
            nopart_ipc = np.array(s_dicts['nopart'][f'cpu{i}.ipc'])
            speedup = sample_ipc/nopart_ipc
            ax[fx,fy].plot(xvals,speedup,label=f'LC waymask={k}',linewidth=2,color=mycolor[ci])
        ax[fx,fy].set_title(f'{works[i]} on cpu{i}')
    ax[0,1].legend(loc='upper right',ncol=1,fontsize=10,bbox_to_anchor=(1.01,1.01))
    plt.savefig(f'{worknames}_ipc.png',dpi=300)


        
if __name__ == '__main__':
    # t_work_combine = ['hmmer_o31-hmmer0-hmmer_o30-hmmer1']
    report_hmmer('/nfs/home/zhangchuanqi/lvna/5g/ff-reshape/log/new_hw_test/16M/hmmer_o31-hmmer0-hmmer_o30-hmmer1')