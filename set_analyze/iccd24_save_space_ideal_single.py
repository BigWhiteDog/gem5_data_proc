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
from matplotlib import patches
from matplotlib import lines
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
    return
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

def analyze_void(s_dicts,work,work_dir,full_perf_ways):
    min_ways = s_dicts['min_ways_no_extra_miss']
    n_less_way = s_dicts['n_less_way']

    for i in range(1,n_less_way+1):
        saved_way = i
        if saved_way > full_perf_ways:
            s_dicts[f'{i}_way_saved_kb'] = 0
            s_dicts[f'{i}_way_saved_setnum'] = 0
            continue
        num_satisfy_set = len(list(filter(lambda x: full_perf_ways - saved_way >= x, min_ways)))
        s_dicts[f'{i}_way_saved_kb'] = (num_satisfy_set * saved_way * 64 // 1024)
        s_dicts[f'{i}_way_saved_setnum'] = num_satisfy_set
    
def draw_db_by_func(base_dir,n_rows,worksname_waydict,analyze_func,draw_one_func,fig_name,
    csv_top_dir,
    input_stats_dict):

    parameters = {'axes.labelsize': 25,
            'axes.titlesize': 30,
            'xtick.labelsize': 28,
            'ytick.labelsize': 24,
            'legend.fontsize': 28,
            'font.size': 45,
            'lines.linewidth': 3,
            'axes.facecolor': 'white',
          'figure.facecolor': 'white',
          'savefig.facecolor': 'white',
            }
    plt.rcParams.update(parameters)
    fig_inchs = (20,5.5)

    fig, ax = plt.subplots()
    fig.set_size_inches(fig_inchs)

    work_stats_dict = {}
    # if input_stats_dict is not None:
    #     work_stats_dict = input_stats_dict
    # work_stats_dict = input_stats_dict

    n_less_way = 3
    space_data_dict = {}
    for i in range(1,n_less_way+1):
        space_data_dict[i] = []

    setnum_ax = ax.twinx()
    bar_width = 0.2

    line_color = '#E6450F'

    for i,work in enumerate(worksname_waydict):
        work = worksname[i]
        full_perf_ways = worksname_waydict[work]
        work_dir = os.path.join(base_dir,work)
        if not os.path.isdir(work_dir):
            continue
        s_dicts = {}
        s_dicts['n_less_way'] = n_less_way
        s_dicts['min_ways_no_extra_miss'] = []
        csv_file = os.path.join(csv_top_dir,f'{work}.csv')
        with open(csv_file,'r') as f:
            for _ in range(all_set):
                s_dicts['min_ways_no_extra_miss'].append(int(f.readline().strip()))
        analyze_func(s_dicts,work,work_dir,full_perf_ways)
        # draw_one_func(None,s_dicts,work,full_perf_ways,(fx,fy))
        setnum_data = []
        setnum_xvals = np.linspace(i,i+n_less_way*bar_width, n_less_way, endpoint=False)
        for lw in range(1,n_less_way+1):
            setnum_data.append(s_dicts[f'{lw}_way_saved_setnum'])
            space_data_dict[lw].append(s_dicts[f'{lw}_way_saved_kb'])
        setnum_data = np.array(setnum_data)
        setnum_ax.plot(setnum_xvals,setnum_data, label=work, color=line_color, marker='o', markersize=10, linewidth=3)
    
    w0xlabels = worksname
    box_step = n_less_way
    x = np.arange(len(w0xlabels))
    colors = [leaf_yellow[i] for i in range(2, len(leaf_yellow),3)]
    for lw in range(1,n_less_way+1):
        ax.bar(x + (lw-1) * bar_width, space_data_dict[lw], bar_width, label=f'{lw}-way', color=colors[lw-1])
    xlabel0_val = (box_step -1) * bar_width / 2
    print(xlabel0_val)
    ax.set_xticks([x+xlabel0_val for x in range(0, len(w0xlabels))])
    ax.set_xticklabels(w0xlabels)
    ax.set_xlim(-bar_width, len(w0xlabels)-bar_width*(box_step-1))
    ax.set_ylabel('Saved space (KB)')
    ax.grid(axis='y', linestyle='--', linewidth=1)
    setnum_ax.set_ylabel('Saved set number')
    setnum_ax.grid(False)
    setnum_ax.set_ylim(0, None)
    
    legends = []
    for lw in range(1,n_less_way+1):
        legends.append(Patch(color=colors[lw-1],label=f'{lw}-less-way space'))
    legends.append(lines.Line2D([], [], color=line_color,marker='o',markersize=10,label='Saved set number'))

    fig.legend(handles = legends, loc = 'upper left', ncol = 4, borderaxespad=0, labelspacing=0, handlelength=0.5)
    # fig.text(0.5, 0.04, f'number of ways between bounds', ha='center', va='center', fontsize=30)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="center",
        # rotation_mode="anchor")
        rotation_mode="default")
    plt.tight_layout(rect=(0, 0, 1, 0.87),pad=0)
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
        (draw_one_ideal_savespace,'iccd24_ideal_save_{}_single.png',os.path.join(csv_dir_path,'min0way_{}')),
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