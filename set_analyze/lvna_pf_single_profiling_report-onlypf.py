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
from set_analyze.my_diff_color import *

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
    "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_lvnapf_50M.json",
]

mycolor = ['#044E48','#06746B','#20876B','#6A9A48','#B5AC23','#E6B80B','#FACA3E','#FFDF80','#FFEBB0']
mycolor = ['#661900','#B22C00','#E6450F','#FF6500','#FF8C00','#FFB200','#FFCB33','#FFDF80','#FDEDBE']

def draw_one_workload_ipc(ax,s_dicts,workload_names):
    pf_keys = [f'{w}-pf' for w in range(1,max_assoc+1)]
    pf_ipcs = [s_dicts[k]['cpu.ipc'][0] for k in pf_keys]
    fullcache_pf_ipc = pf_ipcs[-1]
    normal_pf_ipc = [ipc/fullcache_pf_ipc for ipc in pf_ipcs]
    nopf_keys = [f'{w}-nopf' for w in range(1,max_assoc+1)]
    nopf_ipcs = [s_dicts[k]['cpu.ipc'][0] for k in nopf_keys]
    fullcache_nopf_ipc = nopf_ipcs[-1]
    normal_nopf_ipc = [ipc/fullcache_pf_ipc for ipc in nopf_ipcs]
    x_labels = list(range(1,max_assoc+1))
    ax.plot(x_labels,normal_pf_ipc,marker='o',markersize=8,label='pf')
    ax.plot(x_labels,normal_nopf_ipc,marker='D',markersize=8,label='nopf')
    ax.set_ylabel('Norm IPC')
    ax.set_ylim(0,1.01)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel('number of L3 ways (1MB/way)')
    ax.set_xlim(0,max_assoc+1)
    ax.set_xticks(x_labels)
    ax.set_title(f'{workload_names}')
    ax.legend()

    pf_sense_flag = False
    cap_sense_flag = False

    # if fullcache_nopf_ipc/fullcache_pf_ipc < 0.9:
    #     print(f'{workload_names}')
    #     # use_conf['cache_cap_sense_names']
    #     use_conf['cache_pf_sense_names'].append(workload_names)
    #     pf_sense_flag = True

    # min_cat_nipc = min(normal_pf_ipc)
    # if min_cat_nipc < 0.9:
    #     use_conf['cache_cap_sense_names'].append(workload_names)
    #     cap_sense_flag = True

    # if cap_sense_flag and pf_sense_flag:
    #     use_conf['cache_cap_and_pf_sense_names'].append(workload_names)

    min_cat_nipc = min(normal_pf_ipc)
    # min_cat_nopf_nipc = min(normal_nopf_ipc)
    if (min_cat_nipc <=0.9):
        # use_conf["cache_sensitive_work_names"].append(workload_names)
        # print(f'{workload_names} get slowdown {slowdown_2:.3} 2way')
        fulfill85_way = max_assoc
        fulfill90_way = max_assoc
        fulfill95_way = max_assoc
        for w1 in range(max_assoc,0,-1):
            nipc = normal_pf_ipc[w1-1]
            if nipc>=0.95:
                fulfill85_way = w1
                fulfill90_way = w1
                fulfill95_way = w1
            elif nipc>=0.9:
                fulfill85_way = w1
                fulfill90_way = w1
            elif nipc>=0.85:
                fulfill85_way = w1
            else:
                # print(f'{fulfill5_way}way {ipc:.3} over 95% ipc')
                # break
                pass
        print(f'{workload_names} {fulfill85_way} {fulfill90_way} {fulfill95_way}')
        #update ways
        # use_conf["cache_work_90perfways"][workload_names] = min(fulfill90_way, use_conf["max_assoc"]-1)
        use_conf["cache_work_95perfways"][workload_names] = min(fulfill95_way, use_conf["max_assoc"]-1)
        # use_conf["cache_work_full85perfways"][workload_names] = fulfill85_way
        # use_conf["cache_work_full90perfways"][workload_names] = fulfill90_way
        # use_conf["cache_work_full95perfways"][workload_names] = fulfill95_way


def draw_by_func(base_dir,n_rows,worksname,draw_one_func,fig_name,force_update=False):
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
            if len(ways) != 3:
                continue
            if ways[2] != 'pf' and ways[2] != 'nopf':
                continue
            remain_key = '-'.join(ways[1:])
            new_base = os.path.join(word_dir,part)
            last_nsamples=1
            ncore = 1
            if force_update:
                one_dict = extract_newgem_raw_json(new_base,ncores=ncore,last_nsamples=last_nsamples)
            elif not os.path.exists(os.path.join(new_base,f'{last_nsamples}period.json')):
                one_dict = extract_newgem_raw_json(new_base,ncores=ncore,last_nsamples=last_nsamples)
            else:
                with open(os.path.join(new_base,f'{last_nsamples}period.json'),'r') as f:
                    one_dict = json.load(f)
            # ways[1] as key
            s_dicts[remain_key] = one_dict
        draw_one_func(ax_bar,s_dicts,work)

    plt.tight_layout()
    # plt.savefig(fig_name,dpi=300)
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
    # base_dir += '-new'

    global max_assoc
    max_assoc = use_conf["max_assoc"]

    #clear conf old data
    # use_conf["cache_work_names"] = []
    # use_conf["cache_work_90perfways"] = {}
    use_conf["cache_work_95perfways"] = {}
    # use_conf["cache_work_full85perfways"] = {}
    # use_conf["cache_work_full90perfways"] = {}
    # use_conf["cache_work_full95perfways"] = {}
    # use_conf.pop("cache_work_fullways", None)
    # use_conf.pop("cache_work_90perfways", None)
    # use_conf.pop("cache_work_95perfways", None)
    # use_conf.pop("cache_work_full85perfways", None)
    # use_conf.pop("cache_work_full90perfways", None)
    # use_conf.pop("cache_work_full95perfways", None)
    # use_conf['cache_cap_sense_names'] = []
    # use_conf['cache_pf_sense_names'] = []
    # use_conf['cache_cap_and_pf_sense_names'] = []

    pic_dir_path = f'set_analyze/{test_prefix}pics'
    os.makedirs(pic_dir_path, exist_ok=True)
    worksname = os.listdir(base_dir) #like omnetpp
    worksname.sort()
    # ignore_names = [
    #     'bc',
    #     'bfs',
    #     'blender.17',
    #     'calculix.06',
    #     'cc',
    #     'deepsjeng.17',
    #     'exchange2.17',
    #     'gamess.06',
    #     'gobmk.06',
    #     'hmmer.06',
    #     'imagick.17',
    #     'leela.17',
    #     'namd.06',
    #     'perlbench.06',
    #     'povray.06',
    #     'povray.17',
    #     'pr',
    #     'sjeng.06',
    #     'specjbb',
    #     'tonto.06',
    #     'x264.17',
    # ]
    ignore_names = []
    # remain are interested workloads
    interested_works = [w for w in worksname if w not in ignore_names]

    n_works = len(interested_works)
    n_rows = math.ceil(n_works/4)
    # draw_by_func(base_dir,n_rows,interested_works,
    #     draw_one_func=draw_one_workload_cat_ipc,fig_name=os.path.join(pic_dir_path,'single_pfcat_nipc.png'))
    draw_by_func(base_dir,n_rows,interested_works,
        draw_one_func=draw_one_workload_ipc,fig_name=os.path.join(pic_dir_path,'single_profiling_interested_nipc.png'))
    
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
