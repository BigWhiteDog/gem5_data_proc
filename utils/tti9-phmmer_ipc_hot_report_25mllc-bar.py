from operator import index
import os
import numpy as np
import utils.common as c
from utils.common import extract_samples_raw_json
from utils.insert_avg_dict import insert_avg_dict
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
    nstage = 2
    fig,ax = plt.subplots(nstage,1)
    fig.set_size_inches(16,12)

    stage_low_set = [[1,2,3],[0,1,3] ]
    stage_high_set = [0,2]

    nopart_high_ipc_stage = []
    nopart_bg_ipc_stage = []
    nopart_all_ipc_stage = []
    for i in range(nstage):
        nopart_high_ipc_stage.append(s_dicts['nopart']['1'][f'avg_ipc{stage_high_set[i]}_stage{i}'])
        nopart_bg_ipc_stage.append(np.mean([s_dicts['nopart']['1'][f'avg_ipc{l}_stage{i}'] for l in stage_low_set[i]]))
        nopart_all_ipc_stage.append(np.mean([s_dicts['nopart']['1'][f'avg_ipc{l}_stage{i}'] for l in range(4)]))

    intrested_list = []
    for w in s_dicts:
        if w == 'nopart':
            continue
        for h in s_dicts[w]:
            if h == '1' and w != '5':
                continue
            reach_flag = True
            s_tmp_sum = 0
            for i in range(nstage):
                speedup = s_dicts[w][h][f'avg_ipc{stage_high_set[i]}_stage{i}'] / nopart_high_ipc_stage[i]
                s_tmp_sum += speedup
                if speedup < 1.1:
                    reach_flag = False
            if reach_flag:
                intrested_list.append((w,h,s_tmp_sum/nstage))

    intrested_list.sort(key=lambda x:x[2],reverse=True)
    print('========intrested_list========')
    print(intrested_list)

    width = 0.2
    x = np.arange(len(intrested_list))
    xlabels = [ f'{w}-{h}' for w,h,s in intrested_list]

    for i in range(nstage):
        ax_speedup_bar = ax[i]
        ipchi_list = []
        ipcbg_list = []
        ipcall_list = []
        for w,h,s in intrested_list:
            ipchi_mean = s_dicts[w][h][f'avg_ipc{stage_high_set[i]}_stage{i}']
            ipcbg_mean = np.mean([s_dicts[w][h][f'avg_ipc{l}_stage{i}'] for l in stage_low_set[i]])
            ipc_all_mean = np.mean([s_dicts[w][h][f'avg_ipc{l}_stage{i}'] for l in range(4)])

            ipchi_list.append(ipchi_mean)
            ipcbg_list.append(ipcbg_mean)
            ipcall_list.append(ipc_all_mean)
        speedup0 = np.array(ipchi_list)/nopart_high_ipc_stage[i]
        speedupbg = np.array(ipcbg_list)/nopart_bg_ipc_stage[i]
        speedupall = np.array(ipcall_list)/nopart_all_ipc_stage[i]

        rect0 = ax_speedup_bar.bar(x-width,speedup0.tolist(),width=width,label='speedup of high_ipc_avg')
        rect1 = ax_speedup_bar.bar(x,speedupbg.tolist(),width=width,label='avg speedup of bg_ipc_avg')
        rect2 = ax_speedup_bar.bar(x+width,speedupall.tolist(),width=width,label='avg speedup of all_ipc_avg')

        # ax_speedup_bar.bar_label(rect0,padding=3)
        # ax_speedup_bar.bar_label(rect1,padding=3)
        # ax_speedup_bar.bar_label(rect2,padding=3)
        ax_speedup_bar.legend()

        ax_speedup_bar.set_xticks(x,xlabels)
        ax_speedup_bar.set_ylabel('speedup')
        ax_speedup_bar.set_ylim(0.75,1.25)
        ax_speedup_bar.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax_speedup_bar.set_title(f'speedup of average IPCs in stage{i}')

    # plt.setp(ax_speedup_bar.get_xticklabels(), rotation=30, ha="right",
    #             rotation_mode="anchor")
    plt.tight_layout()
    plt.savefig(f'tti9_all_hot_bar.png',dpi=300)

    print(speedup0.tolist())
    print(speedupbg.tolist())
    print(speedupall.tolist())

if __name__ == '__main__':
    all_base  = '/nfs/home/zhangchuanqi/lvna/5g/ff-reshape/log/new_hw_test/period_hmmer_o3_0-period_hmmer_o3_3-period_hmmer_o2_0-period_hmmer_o2_2/2560kBLLC/9tti/try-waymask'
    last_nsamples = 8
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
            # st_dict[lc_waymask][hot_key] = c.extract_samples_raw_json(new_base2,last_nsamples=last_nsamples)
            with open(os.path.join(new_base2,f'{last_nsamples}period.json'),'r') as f:
                st_dict[lc_waymask][hot_key] = json.load(f)
            insert_avg_dict(st_dict[lc_waymask][hot_key],last_nsamples=last_nsamples )

    report_hmmer(st_dict)