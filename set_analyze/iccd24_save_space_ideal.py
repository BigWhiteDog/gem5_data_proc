from collections import OrderedDict
from genericpath import isdir
import os
import re
import numpy as np
import utils.common as c
from utils.common import extract_newgem_raw_json
import utils.target_stats as t
import numpy as np
import argparse
import math
from matplotlib.ticker import MaxNLocator

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

opt = parser.parse_args()

confs=[
    # "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_xs_tailbm50M.json",
    "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_goldencove_tailbm50M.json",
    # "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_oldinc_tailbm50M.json",
    # "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_skylake_tailbm50M.json",
]

from set_analyze.my_diff_color import *

def draw_one_ideal_savespace(ax,s_dicts,workload_name,full_perf_ways,pos:tuple):
    min_ways = s_dicts['min_ways_no_extra_miss']

    n_less_way = 3
    x_val = np.arange(1,n_less_way+1)

    saved_kb = []
    saved_setnum = []
    for i in range(1,n_less_way+1):
        saved_way = i
        if saved_way > full_perf_ways:
            saved_kb.append(0)
            continue
        num_satisfy_set = len(list(filter(lambda x: full_perf_ways - saved_way >= x, min_ways)))
        saved_kb.append(num_satisfy_set * saved_way * 64 // 1024)
        saved_setnum.append(num_satisfy_set)

    bar_width = 0.3

    ax.bar(x_val - bar_width/2,saved_kb, color = contrasting_orange[0], width = bar_width)

    # if pos[1] == 0:
    #     ax.set_ylabel('saved size(KB)')
    ax.set_ylabel('Space (KB)')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    # ax.set_ylim(0, 1000)
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    # ax.yaxis.set_major_locator(MaxNLocator('auto',integer=True))
    # if pos[0] == 1:
    #     ax.set_xlabel('#ways between bounds')
    ax.set_title(f'{workload_name}')

    ax2 = ax.twinx()
    ax2.bar(x_val + bar_width/2,saved_setnum, color = contrasting_orange[1], width = bar_width)
    ax2.set_ylabel('Set number')
    ax2.set_ylim(0, 1000)

    # if pos == (0,0):
    #     ax.legend(shadow=0, fontsize = 13, bbox_to_anchor=(-0.01,1.4), loc = 'upper left',  \
    #         borderaxespad=0.2, ncol = 1, columnspacing=0.5, labelspacing=0.1)
        # ax.legend(shadow=0, fontsize = 12, bbox_to_anchor=(-0.01,1.3,0,0), loc = 'upper left',  \
        #     borderaxespad=0.2, ncol = 10, columnspacing=0.5, labelspacing=0.1)

def analyze_void(work_stats_dict,work,work_dir,full_ass):
    if work in work_stats_dict:
        return
    s_dicts = {}
    work_stats_dict[work] = s_dicts

def draw_db_by_func(base_dir,n_rows,worksname_waydict,analyze_func,draw_one_func,fig_name,
    csv_top_dir,
    input_stats_dict):

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


    fig,ax = plt.subplots(n_rows,n_cols,sharex=True,sharey=True)
    fig.set_size_inches(n_cols*6, 3.4*n_rows)

    # work_stats_dict = {}
    # if input_stats_dict is not None:
    #     work_stats_dict = input_stats_dict
    work_stats_dict = input_stats_dict

    for i,work in enumerate(worksname_waydict):
        work = worksname[i]
        full_perf_ways = worksname_waydict[work]
        work_dir = os.path.join(base_dir,work)
        if not os.path.isdir(work_dir):
            continue
        fy = i % n_cols
        fx = i // n_cols
        ax_bar = ax[fx,fy]
        analyze_func(work_stats_dict,work,work_dir,full_perf_ways)
        s_dicts = work_stats_dict[work]
        s_dicts['min_ways_no_extra_miss'] = []
        csv_file = os.path.join(csv_top_dir,f'{work}.csv')
        with open(csv_file,'r') as f:
            for i in range(all_set):
                s_dicts['min_ways_no_extra_miss'].append(int(f.readline().strip()))
        draw_one_func(ax_bar,s_dicts,work,full_perf_ways,(fx,fy))     

    for i in range(len(worksname_waydict),n_rows*n_cols):
        fx = i // n_cols
        fy = i % n_cols
        ax[fx,fy].remove()

    legends = [ Patch(color=contrasting_orange[0],label='0-tail-utility space (KB)'),
                Patch(color=contrasting_orange[1],label='0-tail-utility set number'),
                ]
    fig.legend(handles = legends, loc = 'upper left', ncol = 2, borderaxespad=0)
    fig.text(0.5, 0.04, f'number of ways between bounds', ha='center', va='center', fontsize=30)

    plt.tight_layout(rect=(0, 0.01, 1, 0.93))
    plt.savefig(fig_name,dpi=300)
    plt.clf()

    return work_stats_dict


def run_one_conf(select_json:str):
    with open(select_json,'r') as f:
        global use_conf
        use_conf = json.load(f)
    if use_conf is None:
        exit(255)

    global all_set
    all_set = use_conf['all_set']
    global max_assoc
    max_assoc = use_conf['max_assoc']

    global test_prefix
    test_prefix = use_conf['test_prefix']
    base_dir_format = use_conf['base_dir_format']
    base_dir = base_dir_format.format(test_prefix)
    csv_dir_path = f'set_analyze/{test_prefix}other/csv'
    pic_dir_path = f'set_analyze/{test_prefix}pics'
    os.makedirs(pic_dir_path, exist_ok=True)
    global worksname
    worksname = use_conf['cache_work_names'] #like mcf
    worksname.remove('lbm')

    n_works = len(worksname)

    global n_cols
    n_cols= 12
    n_rows = math.ceil(n_works/n_cols)

    waydict_format = 'cache_work_{}ways'
    perf_prefixs = ['95perf']
    # perf_prefixs = ['90perf','95perf','full']
    draw_picformat = [
        (draw_one_ideal_savespace,'iccd24_ideal_save_{}.png',os.path.join(csv_dir_path,'min0way_{}')),
    ]

    for perf_prefix in perf_prefixs:
        waydict_name = waydict_format.format(perf_prefix)
        waydict = use_conf[waydict_name]
        waydict.pop('lbm')
        ret_dict = {}
        for draw_func,pic_name_format,csv_dir_format in draw_picformat:
            csv_dir = csv_dir_format.format(perf_prefix)
            draw_db_by_func(base_dir,n_rows,waydict,
                analyze_func=analyze_void,
                draw_one_func=draw_func,
                fig_name=os.path.join(pic_dir_path,pic_name_format.format(perf_prefix)),
                csv_top_dir=csv_dir,
                input_stats_dict=ret_dict)
            
if __name__ == '__main__':
    # base_dir = '/nfs/home/zhangchuanqi/lvna/for_xs/catlog/single-profiling/'
    if opt.json:
        select_json = opt.json
        run_one_conf(select_json)
    else:
        for co in confs:
            select_json = co
            run_one_conf(select_json)