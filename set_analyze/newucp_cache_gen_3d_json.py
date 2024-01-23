import os
import re
import numpy as np
import utils.common as c
from utils.common import extract_newgem_raw_json
import utils.target_stats as t
import numpy as np
import argparse
import math

import json

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib import ticker
from matplotlib.patches import Patch
import sqlite3


parser = argparse.ArgumentParser(description="options to get set stats")
# parser.add_argument('-d','--stats_dir', type=str,
#     help='stats dir to analyze',required=True)
# parser.add_argument('--ids',default=16,type=int)
# parser.add_argument('--nsamples',default=2,type=int)
# parser.add_argument('--l3_sets',default=4096,type=int)
parser.add_argument('-j','--json', type=str,
    default=None)
parser.add_argument('--interval',type=int,default=10_000_000)
parser.add_argument('--granularity',type=int,default=256)
parser.add_argument('--stride',type=bool,default=False)

opt = parser.parse_args()

confs=[
    # "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_goldencove48M_tailbm50M.json",
    "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_oldincLRU_tailbm250M.json",
    "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_oldincLRU_tailbm50M.json",
    # "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_goldencoveLRU_tailbm250M.json",
    # "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_goldencoveLRU_tailbm50M.json",
]

# from cache_sensitive_names import *
from set_analyze.my_diff_color import *

def draw_one_workload_way_need(ax,s_dicts,workload_name,full_ass,pos:tuple):
    ax.set_title(f'{workload_name}')
    # if pos == (0,0):
    #     ax.legend(shadow=0, bbox_to_anchor=(-0.01,1.6), loc = 'upper left',  \
    #         borderaxespad=0.2, ncol = 1, columnspacing=0.5, labelspacing=0.1)
    #     # ax.legend(shadow=0, fontsize = 12, bbox_to_anchor=(-0.01,1.3,0,0), loc = 'upper left',  \
    #     #     borderaxespad=0.2, ncol = 10, columnspacing=0.5, labelspacing=0.1)


def draw_db_by_func(base_dir,n_rows,worksname_waydict,draw_one_func,fig_name,n_cols=6,
                    force_update_json = False):

    work_stats_dict = {}
    json_base_dir = f'/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/{test_prefix}other/json/'
    if opt.stride:
        json_name = f'newucp_cache_stride_group{opt.granularity}_int{opt.interval}.json'
    else:
        json_name = f'newucp_cache_group{opt.granularity}_int{opt.interval}.json'
    json_path = os.path.join(json_base_dir,json_name)    
    #options of dict and json

    dict_updated = False

    #try load from json
    if json_path is not None and os.path.isfile(json_path) and not force_update_json:
        with open(json_path,'r') as f:
            json_dict = json.load(f)
            if len(json_dict) > 0:
                #it has data
                work_stats_dict.update(json_dict)
                dict_updated = True


    n_rows = math.ceil(len(worksname_waydict) / n_cols)

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



    fig = plt.figure()
    fig.set_size_inches(n_cols*8, 6*n_rows)

    ax = []

    for i in range(n_rows):
        for j in range(n_cols):
            ax.append(fig.add_subplot(n_rows,n_cols,i*n_cols+j+1,projection='3d'))
            # ax.append(fig.add_subplot(n_rows,n_cols,i*n_cols+j+1))


    s_2 = re.compile(r'(\w+)-([\w\.]+)')

    gran = opt.granularity * 2
    ngroups = all_set // gran


    new_work_stats_dict = work_stats_dict

    while ngroups > 1:
        for i,work in enumerate(worksname_waydict):
            work = worksname[i]
            # full_ass = max_assoc
            word_dir = os.path.join(base_dir,work)
            if not os.path.isdir(word_dir):
                continue
            fy = i % n_cols
            fx = i // n_cols
            ax_bar = ax[i]
            s_dicts = new_work_stats_dict[work]
            new_s_dicts = {}

            origin_each_interval_data = s_dicts['each_interval_data']
            new_s_dicts['each_interval_data'] = []
            for ori_group_hit_cnts in origin_each_interval_data:
                tmp_a = np.array(ori_group_hit_cnts)
                merge_a = tmp_a.reshape(-1,2).sum(axis=1)
                new_s_dicts['each_interval_data'].append(merge_a.tolist())

            new_work_stats_dict[work] = new_s_dicts

        #save to json
        if opt.stride:
            save_json_name = f'newucp_cache_stride_group{gran}_int{opt.interval}.json'
        else:
            save_json_name = f'newucp_cache_group{gran}_int{opt.interval}.json'
        save_json_path = os.path.join(json_base_dir,save_json_name)
        if save_json_path is not None:
            jdpath = os.path.dirname(save_json_path)
            os.makedirs(jdpath,exist_ok=True)
            with open(save_json_path,'w') as f:
                json.dump(new_work_stats_dict,f,indent=2)
    
        gran = gran * 2
        ngroups = all_set // gran


def run_one_conf(select_json:str):
    with open(select_json,'r') as f:
        global use_conf
        use_conf = json.load(f)
    if use_conf is None:
        exit(255)
    global test_prefix
    test_prefix = use_conf['test_prefix']
    base_dir_format = use_conf['base_dir_format']
    base_dir = base_dir_format.format(test_prefix)
    pic_dir_path = f'set_analyze/{test_prefix}pics'
    os.makedirs(pic_dir_path, exist_ok=True)
    global worksname
    worksname = use_conf['cache_work_names'] #like mcf
    cache_work_90perfways = use_conf['cache_work_90perfways']
    cache_work_95perfways = use_conf['cache_work_95perfways']
    cache_work_95perf_maxfull_ways = use_conf['cache_work_full95perfways']
    # cache_work_fullways = use_conf['cache_work_fullways']
    global all_set
    all_set = use_conf['all_set']
    global max_assoc
    max_assoc = use_conf['max_assoc']

    print(worksname)
    # worksname.remove('lbm')
    # cache_work_95perf_maxfull_ways.pop('lbm')

    n_works = len(worksname)
    n_cols = 5
    n_rows = math.ceil(n_works/n_cols)

    if opt.stride:
        fig_name = f'newucp_hitcnt_stride_group{opt.granularity}_int{opt.interval}.png'
    else:
        fig_name = f'newucp_hitcnt_group{opt.granularity}_int{opt.interval}.png'

    draw_db_by_func(base_dir,n_rows,cache_work_95perf_maxfull_ways,
                    n_cols=n_cols,
        draw_one_func=draw_one_workload_way_need, fig_name=os.path.join(pic_dir_path,fig_name))


if __name__ == '__main__':
    # base_dir = '/nfs/home/zhangchuanqi/lvna/for_xs/catlog/single-profiling/'
    if opt.json:
        select_json = opt.json
        run_one_conf(select_json)
    else:
        for co in confs:
            select_json = co
            run_one_conf(select_json)