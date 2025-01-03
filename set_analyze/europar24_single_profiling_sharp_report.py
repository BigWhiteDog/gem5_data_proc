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
from set_analyze.my_diff_color import *
# from cache_sensitive_names import *

parser = argparse.ArgumentParser(description="options to get set stats")
# parser.add_argument('-d','--stats_dir', type=str,
#     help='stats dir to analyze',required=True)
# parser.add_argument('--ids',default=16,type=int)
# parser.add_argument('--nsamples',default=2,type=int)
# parser.add_argument('--l3_sets',default=4096,type=int)
parser.add_argument('-j','--json', type=str,
    default=None)

opt = parser.parse_args()

confs=[
    # "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_xs_tailbm50M.json",
    "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_goldencove_tailbm50M.json",
    # "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_goldencove24M_tailbm50M.json",
    # "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_goldencove48M_tailbm50M.json",
    # "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_oldinc_tailbm50M.json",
    # "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_skylake_tailbm50M.json",
]

mycolor = ['#044E48','#06746B','#20876B','#6A9A48','#B5AC23','#E6B80B','#FACA3E','#FFDF80','#FFEBB0']
mycolor = ['#661900','#B22C00','#E6450F','#FF6500','#FF8C00','#FFB200','#FFCB33','#FFDF80','#FDEDBE']

def draw_one_workload_ipc(ax,s_dicts,workload_names):
    sorted_keys = sorted(s_dicts.keys(),key=lambda x:int(x),reverse=False)
    ipcs = [s_dicts[k]['cpu.ipc'][0] for k in sorted_keys]
    max_ipc = ipcs[-1]
    nipcs = [i/max_ipc for i in ipcs]

    min_slowdown = np.min(nipcs)
    if (min_slowdown <=0.9):
        nipc_a = np.array(nipcs)
        #get abs diff of each nipc
        nipc_diff = np.diff(nipc_a)
        #turn to abs value
        nipc_diff = np.abs(nipc_diff)
        max_diff = np.max(nipc_diff)
        # print(f'{workload_names} nipc diffs {nipc_diff}')
        if max_diff >= 0.05:
            #find each idx of diff > 0.05
            idxs = np.where(nipc_diff >= 0.05)[0]
            # start from min idx
            # idxs = idxs + 1
            idxs_start = idxs[0]
            print(f'{workload_names} get 5% sharp slowdown at idxs {idxs}')
            print(f'{workload_names} diffs: {nipc_diff[idxs]}')
            # print('"',f'{workload_names}',end='",',sep='')

        use_conf["cache_work_names"].append(workload_names)
        # print(f'{workload_names} get slowdown {slowdown_2:.3} 2way')

    ax.set_ylabel('相对IPC')
    ax.set_xlim(-0.2,len(sorted_keys)-0.8)
    # ax.set_ylim(0.65,1)
    # ax.set_ylim(0.7,1)
    # ax.set_ylim(None,1)
    # ax.set_ylim(None, 1)
    ax.set_xticks(np.arange(0, len(sorted_keys), step=2))
    ax.set_xticks(np.arange(0, len(sorted_keys), step=1), minor=True)
    # ax.minorticks_on()
    ax.grid(True, which='both')
    # ax.grid(True)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    # ax.set_xlabel('number of L3 ways (1MB/way)')
    ax.set_xlabel('L3分配的路数目')
    ax.set_title(f'{workload_names}')
    interest_keys = sorted_keys[idxs_start:idxs_start+len(idxs)+1]
    interest_nipcs = nipcs[idxs_start:idxs_start+len(idxs)+1]
    ax.plot(sorted_keys,nipcs, color='#004988',marker='o',label='ipc')
    ax.plot(interest_keys,interest_nipcs, color='#FF4500',marker='o',label='ipc sharp')


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
    parameters = {'axes.labelsize': 30,
          'axes.titlesize': 30,
          'xtick.labelsize': 24,
          'ytick.labelsize': 24,
          'legend.fontsize': 30,
          'font.size': 20,
          'axes.facecolor': 'white',
          'figure.facecolor': 'white',
          'savefig.facecolor': 'white',
          }
    plt.rcParams.update(parameters)
    
    n_cols = 3
    fig,ax = plt.subplots(n_rows,n_cols)
    fig.set_size_inches(n_cols*6, 5*n_rows)

    for i,work in enumerate(worksname):
        word_dir = os.path.join(base_dir,work)
        if not os.path.isdir(word_dir):
            continue
        fy = i % n_cols
        fx = i // n_cols
        if n_rows == 1:
            ax_bar = ax[fy]
        else:
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

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.tight_layout(pad=0.5)
    plt.savefig(fig_name,dpi=300)
    plt.clf()



def run_one_conf(select_json:str):
    with open(select_json,'r') as f:
        global use_conf
        use_conf = json.load(f)
    if use_conf is None:
        exit(255)
    base_dir_format = use_conf['base_dir_format']
    test_prefix = use_conf['test_prefix']
    base_dir = base_dir_format.format(test_prefix)

    #clear conf old data
    use_conf["cache_work_names"] = []
    use_conf["cache_work_90perfways"] = {}
    use_conf["cache_work_95perfways"] = {}
    use_conf["cache_work_full85perfways"] = {}
    use_conf["cache_work_full90perfways"] = {}
    use_conf["cache_work_full95perfways"] = {}
    use_conf.pop("cache_work_fullways", None)

    pic_dir_path = f'set_analyze/{test_prefix}pics'
    os.makedirs(pic_dir_path, exist_ok=True)
    worksname = os.listdir(base_dir) #like omnetpp
    worksname.sort()
    # worksname = ["cam4","cc_sv","fotonik3d","imgdnn","lbm","parest","pr_spmv","roms","xalancbmk","xapian",]
    worksname = ["cam4","cc_sv","imgdnn","fotonik3d","parest","xapian",]

    n_works = len(worksname)
    n_rows = math.ceil(n_works/3)
    # draw_by_func(base_dir,n_rows,worksname,
    #     draw_one_func=draw_one_workload_ipc,fig_name=os.path.join(pic_dir_path,'setconf_single_profiling_ipc.png'))
    # draw_by_func(base_dir,n_rows,worksname,
    #     draw_one_func=draw_one_workload_missrate,fig_name=os.path.join(pic_dir_path,'setconf_single_profiling_mr.png'))
    draw_by_func(base_dir,n_rows,worksname,
        draw_one_func=draw_one_workload_ipc,fig_name=os.path.join(pic_dir_path,'doc-europar24_sharp_way.png'))

    # with open(select_json,'w') as f:
    #     json.dump(use_conf,f,indent=4)

if __name__ == '__main__':
    # base_dir = '/nfs/home/zhangchuanqi/lvna/for_xs/catlog/single-profiling/'
    if opt.json:
        select_json = opt.json
        run_one_conf(select_json)
    else:
        for co in confs:
            select_json = co
            run_one_conf(select_json)
