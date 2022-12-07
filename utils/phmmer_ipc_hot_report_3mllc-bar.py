from operator import index
import os
import numpy as np
import utils.common as c
from utils.common import extract_samples_raw_json
import utils.target_stats as t
import numpy as np
import argparse
import re

import json

import matplotlib
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

def report_hmmer(s_dicts:dict):
    fig,ax = plt.subplots(1,1)
    fig.set_size_inches(16,9)

    nopart_all_ipc_array = np.array([s_dicts['nopart']['1'][f'cpu{x}.ipc'] for x in range(4)])
    nopart_ipc0_mean = np.mean(nopart_all_ipc_array[0])
    nopart_ipcbg_mean = np.mean(nopart_all_ipc_array[1:])
    nopart_ipc_all_mean = np.mean(nopart_all_ipc_array)
    intrested_list = []
    for w in s_dicts:
        if w == 'nopart':
            continue
        for h in s_dicts[w]:
            if h == '1' and w != '0x1ff':
                continue
            sample_ipc0 = np.array(s_dicts[w][h][f'cpu{0}.ipc'])
            speedup = np.mean(sample_ipc0)/nopart_ipc0_mean
            if speedup < 1.05:
                continue
            intrested_list.append((w,h,speedup))

    intrested_list.sort(key=lambda x:x[2],reverse=True)

    ax_speedup_bar = ax
    width = 0.2
    x = np.arange(len(intrested_list))
    xlabels = [ f'{w}-{h}' for w,h,s in intrested_list]

    ipc0_list = []
    ipcbg_list = []
    ipcall_list = []
    for w,h,s in intrested_list:
        all_ipc_array = np.array([s_dicts[w][h][f'cpu{x}.ipc'] for x in range(4)])
        ipc0_mean = np.mean(all_ipc_array[0])
        ipcbg_mean = np.mean(all_ipc_array[1:])
        ipc_all_mean = np.mean(all_ipc_array)
        ipc0_list.append(ipc0_mean)
        ipcbg_list.append(ipcbg_mean)
        ipcall_list.append(ipc_all_mean)
    speedup0 = np.array(ipc0_list)/nopart_ipc0_mean
    speedupbg = np.array(ipcbg_list)/nopart_ipcbg_mean
    speedupall = np.array(ipcall_list)/nopart_ipc_all_mean

    rect0 = ax_speedup_bar.bar(x-width,speedup0.tolist(),width=width,label='speedup of ipc0 avg')
    rect1 = ax_speedup_bar.bar(x,speedupbg.tolist(),width=width,label='avg speedup of bgipc avg')
    rect2 = ax_speedup_bar.bar(x+width,speedupall.tolist(),width=width,label='avg speedup of allipc avg')

    # ax_speedup_bar.bar_label(rect0,padding=3)
    # ax_speedup_bar.bar_label(rect1,padding=3)
    # ax_speedup_bar.bar_label(rect2,padding=3)
    ax_speedup_bar.legend()

    ax_speedup_bar.set_xticks(x,xlabels)
    ax_speedup_bar.set_ylabel('speedup')
    ax_speedup_bar.set_ylim(0.9,1.1)
    ax_speedup_bar.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax_speedup_bar.set_title(f'speedup of average IPCs')

    plt.setp(ax_speedup_bar.get_xticklabels(), rotation=30, ha="right",
                rotation_mode="anchor")
    plt.tight_layout()
    plt.savefig(f'3m_all_hot_bar.png',dpi=300)

    print(speedup0.tolist())
    print(speedupbg.tolist())
    print(speedupall.tolist())

if __name__ == '__main__':
    all_base  = '/nfs/home/zhangchuanqi/lvna/5g/ff-reshape/log/new_hw_test/period_hmmer_o3_0-period_hmmer_o3_3-period_hmmer_o2_0-period_hmmer_o2_2/3072kBLLC/try-waymask'
    st_dict = {}
    s_2 = re.compile(r'(\w+)-([\w\.]+)')
    for part in os.listdir(all_base):
        #like l3-0x1-0xfe-0xfe-0xfe
        res = s_2.search(part)
        if not res:
            continue
        if res.group(1) != 'l3':
            continue
        lc_waymask = res.group(2)
        st_dict[lc_waymask] = {}
        new_base = os.path.join(all_base,part)
        for hotstr in os.listdir(new_base):
            hres = s_2.search(hotstr)
            if not hres:
                continue
            if hres.group(1) == 'l2':
                hot_key = '1'
            elif hres.group(1) == 'hot':
                hot_key = hres.group(2)
            else:
                continue
            new_base2 = os.path.join(new_base,hotstr)
            # st_dict[lc_waymask][hot_key] = c.extract_samples_raw_json(new_base2)
            with open(os.path.join(new_base2,'4period.json'),'r') as f:
                st_dict[lc_waymask][hot_key] = json.load(f)

    report_hmmer(st_dict)