import os
import numpy as np
import utils.common as c
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
    fig,ax = plt.subplots(2,2,sharex=True)
    fig.set_size_inches(16,9)
    xvals = np.arange(0,4,1)

    nopart_ipc0 = np.array(s_dicts['nopart']['1'][f'cpu{0}.ipc'])
    nopart_all_ipc_array = np.array([s_dicts['nopart']['1'][f'cpu{x}.ipc'] for x in range(4)])
    intrested_dict = {}
    for w in s_dicts:
        if w == 'nopart':
            continue
        intrested_dict[w] = {}
        for h in s_dicts[w]:
            # if h == '1' and w != '0x1ff':
            #     continue
            sample_ipc0 = np.array(s_dicts[w][h][f'cpu{0}.ipc'])
            speedup = np.mean(sample_ipc0)/np.mean(nopart_ipc0)
            if speedup < 1.05:
                continue
            intrested_dict[w][h] = speedup

    all_ipc_list = []
    nopart_ipc0_mean = np.mean(nopart_ipc0)
    bg_nopart_ipc_mean = np.mean(nopart_all_ipc_array.mean(axis=1)[1:])
    for i in range(4):
        fy = i % 2
        fx = i // 2
        ax[fx,fy].xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax[fx,fy].set_xlabel('TTI')
        ax[fx,fy].set_ylabel('IPC speedup')
        ax[fx,fy].set_ylim(0.85,1.15)
        ax[fx,fy].yaxis.set_major_locator(ticker.MultipleLocator(0.05))
        nopart_ipc = np.array(s_dicts['nopart']['1'][f'cpu{i}.ipc'])
        nopart_ipc_avg = np.mean(nopart_ipc)
        for w in intrested_dict:
            hkeys = intrested_dict[w].keys()
            hkeys = sorted(hkeys,key=lambda x:float(x))
            for h in hkeys:
                sample_ipc = np.array(s_dicts[w][h][f'cpu{i}.ipc'])
                sample_ipc_avg = np.mean(sample_ipc)
                if i == 0:
                    all_ipc_array = np.array([s_dicts[w][h][f'cpu{x}.ipc'] for x in range(4)])
                    all_ipc_mean = np.mean(all_ipc_array)
                    bg_ipc_mean = np.mean(all_ipc_array.mean(axis=1)[1:])
                    print(f'core{i} lc:{w} h:{h}')
                    print(f'core0_ipc_speedup:{sample_ipc_avg} bg_ipc_speedup:{bg_ipc_mean/bg_nopart_ipc_mean}')
                    # all_ipc_list.append( (bg_ipc_mean,f'{w}-{h}',all_ipc_array.mean(axis=1).tolist()) )
                    all_ipc_list.append( (bg_ipc_mean/bg_nopart_ipc_mean,f'{w}-{h}',sample_ipc_avg/nopart_ipc0_mean) )
                speedup = sample_ipc/nopart_ipc
                ax[fx,fy].plot(xvals,speedup,label=f'{w}-{h}',linewidth=1.5)
        ax[fx,fy].set_title(f'cpu{i}')
    ax[0,1].legend(loc='upper right',ncol=1,fontsize=10,bbox_to_anchor=(1.01,1.01))
    # fig_path = os.path.join(base_dir,'tb_ipc_speedup.png')
    plt.savefig(f'3m_all_hot.png',dpi=300)
    all_ipc_list.sort(key=lambda x: x[0] ,reverse=True)
    print(all_ipc_list)

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