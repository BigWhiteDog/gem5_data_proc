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
from sortedcontainers import SortedDict,SortedList,SortedKeyList
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

opt = parser.parse_args()

from cache_sensitive_names import *
from set_analyze.my_diff_color import *

all_set = 16384
# full_ass = 8
tail_set = int(0.001*all_set)


def draw_one_workload_trace_scatter(ax,s_dicts,workload_name,full_ass,pos:tuple):
    tag_t_list = s_dicts['tag_trace']
    time_t_list = s_dicts['timestamp_trace']
    ax.scatter(time_t_list,tag_t_list, s=0.1, label='tag trace', color = contrasting_orange[5])
    ax.set_xlabel('cycles')
    ax.set_ylabel('tag')
    ax.set_ylim(9900,10100)
    ax.set_title(f'{workload_name}')

def analyze_workload_len_est(work_stats_dict,work,work_dir,full_ass):
    if work in work_stats_dict:
        return
    s_2 = re.compile(r'(\w+)-([\w\.]+)')
    s_dicts = {}
    tag_trace = []
    timestamp_trace = []

    partsname = os.listdir(work_dir) #like l3-1
    for part in partsname:
        if not os.path.isdir(os.path.join(work_dir,part)):
            continue
        res = s_2.search(part)
        if not res:
            continue
        if res.group(1) != 'l3':
            continue
        ways = int(res.group(2))
        if ways != full_ass:
            continue

        new_base = os.path.join(work_dir,part)
        db_path = os.path.join(new_base,'hm.db')
        all_access_query = 'SELECT SETIDX,TAG,STAMP FROM HitMissTrace;'
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        f = cur.execute(all_access_query)

        stamp0 = 0
        for idx,tag,stamp in f:
            idx = int(idx)
            tag = int(tag)
            stamp = int(stamp)
            if stamp0 == 0:
                stamp0 = stamp
            delta_stamp = stamp - stamp0
            tag_trace.append(tag)
            timestamp_trace.append(delta_stamp)

        cur.close()
    
    s_dicts['tag_trace'] = tag_trace
    s_dicts['timestamp_trace'] = timestamp_trace

    work_stats_dict[work] = s_dicts

def draw_db_by_func(base_dir,n_rows,worksname_waydict,analyze_func,draw_one_func,fig_name,input_stats_dict=None):
    fig,ax = plt.subplots(n_rows,4)
    fig.set_size_inches(24, 4.5*n_rows+3)
    # fig,ax = plt.subplots(1,1)
    # fig.set_size_inches(24, 16)

    work_stats_dict = {}
    if input_stats_dict is not None:
        work_stats_dict = input_stats_dict

    for i,work in enumerate(worksname_waydict):
        # if work != 'astar_biglakes':
        #     continue
        full_ass = worksname_waydict[work]
        work_dir = os.path.join(base_dir,work)
        if not os.path.isdir(work_dir):
            continue
        ax_bar = ax
        analyze_func(work_stats_dict,work,work_dir,full_ass)
        s_dicts = work_stats_dict[work]
        draw_one_func(ax_bar,s_dicts,work,full_ass,(0,0))     

    # for i in range(len(worksname_waydict),n_rows*4):
    #     fx = i // 4
    #     fy = i % 4
    #     ax[fx,fy].remove()

    plt.tight_layout()
    plt.savefig(fig_name,dpi=300)
    plt.clf()

    return work_stats_dict


if __name__ == '__main__':
    use_conf = conf_50M
    test_prefix = use_conf['test_prefix']
    base_dir = base_dir_format.format(test_prefix)
    pic_dir_path = f'set_analyze/{test_prefix}pics'
    os.makedirs(pic_dir_path, exist_ok=True)
    worksname = use_conf['cache_work_names'] #like mcf

    n_works = len(worksname)
    n_rows = math.ceil(n_works/4)

    waydict_format = 'cache_work_{}ways'
    perf_prefixs = ['90perf','95perf','full']
    draw_picformat = [
        (draw_one_workload_trace_scatter,'tag_trace_{}_dis.png'),
    ]

    for perf_prefix in perf_prefixs:
        waydict_name = waydict_format.format(perf_prefix)
        waydict = use_conf[waydict_name]
        ret_dict = {}
        for draw_func,pic_name_format in draw_picformat:
            draw_db_by_func(base_dir,n_rows,waydict,
                analyze_func=analyze_workload_len_est,
                draw_one_func=draw_func,
                fig_name=os.path.join(pic_dir_path,pic_name_format.format(perf_prefix)),
                input_stats_dict=ret_dict)