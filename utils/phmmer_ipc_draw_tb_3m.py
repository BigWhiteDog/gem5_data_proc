from asyncio import set_child_watcher
from operator import delitem
import os
import numpy as np
import utils.common as c
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

def report_hmmer(s_dicts:dict):
    fig,ax = plt.subplots(2,2,sharex=True)
    fig.set_size_inches(16,9)
    sorted_keys = sorted(s_dicts.keys(),key=lambda x: int(x.split('-')[1])/int(x.split('-')[0]),reverse=False)
    max_key = sorted_keys[-1]
    xvals = np.arange(0,4,1)
    for i in range(4):
        fy = i % 2
        fx = i // 2
        ax[fx,fy].xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax[fx,fy].set_xlabel('TTI')
        ax[fx,fy].set_ylabel('IPC speedup')
        # ax[fx,fy].set_ylim(0.85,1.15)
        # ax[fx,fy].yaxis.set_major_locator(ticker.MultipleLocator(0.05))
        for ci,k in enumerate(sorted_keys):
            sample_ipc = np.array(s_dicts[k][f'cpu{i}.ipc'])
            nopart_ipc = np.array(s_dicts[max_key][f'cpu{i}.ipc'])
            speedup = sample_ipc/nopart_ipc
            ax[fx,fy].plot(xvals,speedup,label=f'BE inc={k}',linewidth=1.5,color=mycolor[ci%len(mycolor)])
        ax[fx,fy].set_title(f'cpu{i}')
    ax[0,1].legend(loc='upper right',ncol=1,fontsize=10,bbox_to_anchor=(1.01,1.01))
    # fig_path = os.path.join(base_dir,'tb_ipc_speedup.png')
    plt.savefig(f'3m_tb_period_ipc_speedup.png',dpi=300)

if __name__ == '__main__':
    # t_work_combine = ['hmmer_o31-hmmer0-hmmer_o30-hmmer1']
    # report_hmmer('/nfs/home/zhangchuanqi/lvna/5g/ff-reshape/log/new_hw_test/16M/hmmer_o31-hmmer0-hmmer_o30-hmmer1')
    all_base  = '/nfs/home/zhangchuanqi/lvna/5g/ff-reshape/log/new_hw_test/period_hmmer_o3_0-period_hmmer_o3_3-period_hmmer_o2_0-period_hmmer_o2_2/3072kBLLC/try-tb'
    tb_base = os.path.join(all_base,'l3-nopart','l2-nopart')
    tb_bases = os.listdir(tb_base)
    st_dict = {}
    for tb in tb_bases:
        if tb.startswith('l3-tb'):
            tb_path = os.path.join(tb_base,tb)
            c.extract_samples_raw_json(tb_path)
            with open(os.path.join(tb_path,'4period.json')) as f:
                st_json = json.load(f)
            tb_inc = tb.split('-')[3]
            if len(tb_inc) == 0:
                tb_inc = '1024'
            tb_freq = tb.split('-')[2]
            if len(tb_freq) == 0:
                tb_freq = '256'
            st_dict[tb_freq+'-'+tb_inc] = st_json
    st_dict.pop('256-1024',None)
    report_hmmer(st_dict)