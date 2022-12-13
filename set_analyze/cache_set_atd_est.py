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


def draw_one_workload_atd_est(ax,s_dicts,workload_name,full_ass,pos:tuple):
    label_s = ['min_ways_no_extra_miss','min_ways_1_extra_miss', 'min_ways_2_extra_miss']
    extra0_list = s_dicts[label_s[0]]
    extra1_list = s_dicts[label_s[1]]
    extra2_list = s_dicts[label_s[2]]
    sorted_setlist = sorted(zip(extra0_list,extra1_list,extra2_list))
    s_extra0_list,s_extra1_list,s_extra2_list = zip(*sorted_setlist)
    x_val = np.arange(all_set)
    full_ass_vals = np.full(all_set,full_ass)

    extra0_list_color = contrasting_orange[2]
    extra1_list_color = contrasting_orange[3]
    extra2_list_color = contrasting_orange[4]
    alpha_set = 0.8
    ax.plot(s_extra2_list, label='min ways 2 extra miss', color = extra2_list_color,linewidth=1)
    ax.fill_between(x_val,full_ass_vals, s_extra2_list, color = extra2_list_color, alpha=alpha_set)
    ax.plot(s_extra1_list, label='min ways 1 extra miss', color = extra1_list_color,linewidth=1)
    ax.fill_between(x_val, full_ass_vals, s_extra1_list, color = extra1_list_color, alpha=alpha_set)
    ax.plot(s_extra0_list, label='min ways no extra miss', color = extra0_list_color,linewidth=1)
    ax.fill_between(x_val, full_ass_vals, s_extra0_list, color = extra0_list_color, alpha=alpha_set)
    ax.set_ylabel('needed ways')
    ax.set_ylim(0, 8)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xlabel('set idx (sorted by min 0miss ways)')
    ax.set_title(f'{workload_name}')
    if pos == (0,0):
        ax.legend(shadow=0, fontsize = 13, bbox_to_anchor=(-0.01,1.4), loc = 'upper left',  \
            borderaxespad=0.2, ncol = 1, columnspacing=0.5, labelspacing=0.1)
        # ax.legend(shadow=0, fontsize = 12, bbox_to_anchor=(-0.01,1.3,0,0), loc = 'upper left',  \
        #     borderaxespad=0.2, ncol = 10, columnspacing=0.5, labelspacing=0.1)


def draw_db_by_func(base_dir,n_rows,worksname_waydict,draw_one_func,fig_name):
    fig,ax = plt.subplots(n_rows,4)
    fig.set_size_inches(24, 4.5*n_rows+3)

    s_2 = re.compile(r'(\w+)-([\w\.]+)')

    for i,work in enumerate(worksname_waydict):
        full_ass = worksname_waydict[work]
        word_dir = os.path.join(base_dir,work)
        if not os.path.isdir(word_dir):
            continue
        fy = i % 4
        fx = i // 4
        ax_bar = ax[fx,fy]
        s_dicts = {}
        lru_hit_cnts = [np.zeros(full_ass) for _ in range(all_set)]
        set_hit_cnts = np.zeros(all_set)

        partsname = os.listdir(word_dir) #like l3-1
        for part in partsname:
            if not os.path.isdir(os.path.join(word_dir,part)):
                continue
            res = s_2.search(part)
            if not res:
                continue
            if res.group(1) != 'l3':
                continue
            ways = int(res.group(2))
            if ways != full_ass:
                continue

            new_base = os.path.join(word_dir,part)
            db_path = os.path.join(new_base,'hm.db')
            all_access_query = 'SELECT SETIDX,TAG,STAMP FROM HitMissTrace;'
            con = sqlite3.connect(db_path)
            cur = con.cursor()
            f = cur.execute(all_access_query)

            lru_states = [SortedList() for _ in range(all_set)]
            for idx,tag,stamp in f:
                idx = int(idx)
                tag = int(tag)
                stamp = int(stamp)
                ls = lru_states[idx]
                fi = filter(lambda x: x[1] == tag, ls)
                res = list(fi)
                if len(res) > 0:
                    #hit
                    set_hit_cnts[idx] += 1
                    get_pair = res[0]
                    lru_index = ls.index(get_pair)
                    hit_pos = len(ls) - lru_index - 1
                    lru_hit_cnts[idx][hit_pos] += 1
                    ls.remove(get_pair)
                    ls.add([stamp,tag])
                else:
                    #miss
                    if len(ls) >= full_ass:
                        ls.pop(0)
                    ls.add([stamp,tag])

            cur.close()
        s_dicts['min_ways_no_extra_miss'] = [full_ass for _ in range(all_set)]
        s_dicts['min_ways_1_extra_miss'] = [full_ass for _ in range(all_set)]
        s_dicts['min_ways_2_extra_miss'] = [full_ass for _ in range(all_set)]

        for idx in range(all_set):
            hitpos_cnts = lru_hit_cnts[idx]
            set_hit_cnt = set_hit_cnts[idx]
            sum_hit = 0
            for hitpos in range(full_ass):
                sum_hit += hitpos_cnts[hitpos]
                hit_loss =  set_hit_cnt - sum_hit
                if hit_loss == 0:
                    s_dicts['min_ways_no_extra_miss'][idx] = min(s_dicts['min_ways_no_extra_miss'][idx],hitpos+1)
                if hit_loss <= 1:
                    s_dicts['min_ways_1_extra_miss'][idx] = min(s_dicts['min_ways_1_extra_miss'][idx],hitpos+1)
                if hit_loss <= 2:
                    s_dicts['min_ways_2_extra_miss'][idx] = min(s_dicts['min_ways_2_extra_miss'][idx],hitpos+1)

        draw_one_func(ax_bar,s_dicts,work,full_ass,(fx,fy))


    plt.tight_layout()
    plt.savefig(fig_name,dpi=300)
    plt.clf()


if __name__ == '__main__':
    base_dir = '/nfs/home/zhangchuanqi/lvna/for_xs/catlog/single-profiling/'
    worksname = cache_work_names #like mcf
    # worksname = ['sphinx3','mcf'] #like mcf
    # worksname = os.listdir(base_dir)
    n_works = len(worksname)
    n_rows = math.ceil(n_works/4)
    draw_db_by_func(base_dir,n_rows,cache_work_90perfways,
        draw_one_func=draw_one_workload_atd_est,fig_name='set_analyze/pics/atd_est_90perf_dis.png')
    draw_db_by_func(base_dir,n_rows,cache_work_95perfways,
        draw_one_func=draw_one_workload_atd_est,fig_name='set_analyze/pics/atd_est_95perf_dis.png')
    draw_db_by_func(base_dir,n_rows,cache_work_fullways,
        draw_one_func=draw_one_workload_atd_est,fig_name='set_analyze/pics/atd_est_dis.png')
