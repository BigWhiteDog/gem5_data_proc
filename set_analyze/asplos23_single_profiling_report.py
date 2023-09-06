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
    # "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_goldencove_tailbm50M.json",
    "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_goldencove24M_tailbm50M.json",
    "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_goldencove48M_tailbm50M.json",
    # "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_oldinc_tailbm50M.json",
    # "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_skylake_tailbm50M.json",
]

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
    # ax.set_xlabel('number of L3 ways (1MB/way)')
    ax.set_xlabel('number of L3 ways')
    ax.set_title(f'{workload_names}')
    ax.plot(sorted_keys,nipcs,marker='o',label='ipc')

    slowdown_2 = nipcs[-2]
    min_slowdown = np.min(nipcs)
    if (min_slowdown <=0.9):
        use_conf["cache_work_names"].append(workload_names)
        # print(f'{workload_names} get slowdown {slowdown_2:.3} 2way')
        fulfill85_way = len(nipcs)
        fulfill90_way = len(nipcs)
        fulfill95_way = len(nipcs)
        for nipc,way in zip(nipcs,sorted_keys):
            if nipc>=0.95:
                fulfill85_way = int(way)
                fulfill90_way = int(way)
                fulfill95_way = int(way)
            elif nipc>=0.9:
                fulfill85_way = int(way)
                fulfill90_way = int(way)
            elif nipc>=0.85:
                fulfill85_way = int(way)
            else:
                # print(f'{fulfill5_way}way {ipc:.3} over 95% ipc')
                break
        print(f'{workload_names} {slowdown_2:.3} {fulfill85_way} {fulfill90_way} {fulfill95_way}')
        #update ways    
        use_conf["cache_work_90perfways"][workload_names] = min(fulfill90_way, use_conf["max_assoc"]-1)
        use_conf["cache_work_95perfways"][workload_names] = min(fulfill95_way, use_conf["max_assoc"]-1)
        use_conf["cache_work_full85perfways"][workload_names] = fulfill85_way
        use_conf["cache_work_full90perfways"][workload_names] = fulfill90_way
        use_conf["cache_work_full95perfways"][workload_names] = fulfill95_way


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
            one_dict = extract_newgem_raw_json(new_base,ncores=1,last_nsamples=last_nsamples)
            with open(os.path.join(new_base,f'{last_nsamples}period.json'),'r') as f:
                one_dict = json.load(f)
            # ways[1] as key
            s_dicts[ways[1]] = one_dict
        draw_one_func(ax_bar,s_dicts,work)

    plt.tight_layout()
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
    n_works = len(worksname)
    n_rows = math.ceil(n_works/4)
    draw_by_func(base_dir,n_rows,worksname,
        draw_one_func=draw_one_workload_ipc,fig_name=os.path.join(pic_dir_path,'setconf_single_profiling_ipc.png'))
    draw_by_func(base_dir,n_rows,worksname,
        draw_one_func=draw_one_workload_missrate,fig_name=os.path.join(pic_dir_path,'setconf_single_profiling_mr.png'))

    with open(select_json,'w') as f:
        json.dump(use_conf,f,indent=4)

if __name__ == '__main__':
    # base_dir = '/nfs/home/zhangchuanqi/lvna/for_xs/catlog/single-profiling/'
    if opt.json:
        select_json = opt.json
        run_one_conf(select_json)
    else:
        for co in confs:
            select_json = co
            run_one_conf(select_json)
