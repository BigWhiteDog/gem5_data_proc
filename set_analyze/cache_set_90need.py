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


def draw_one_workload_way_need(ax,s_dicts,workload_name,full_ass,pos:tuple):
    label_s = ['min_ways_no_extra_miss','min_ways_1_extra_miss', 'min_ways_2_extra_miss']
    extra0_list = s_dicts[label_s[0]]
    extra1_list = s_dicts[label_s[1]]
    extra2_list = s_dicts[label_s[2]]
    sorted_setlist = sorted(zip(extra0_list,extra1_list,extra2_list))
    s_extra0_list,s_extra1_list,s_extra2_list = zip(*sorted_setlist)
    x_val = np.arange(all_set)
    full_ass_vals = np.full(all_set,full_ass)

    extra0_list_color = contrasting_orange[6]
    extra1_list_color = contrasting_orange[7]
    extra2_list_color = contrasting_orange[8]
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
        s_dicts['unique_blocks_number'] = [0 for _ in range(all_set)]
        s_dicts['unique_reused_blocks_number'] = [0 for _ in range(all_set)]
        s_dicts['ways_miss_cnt'] = {}

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
            if ways > full_ass:
                continue

            new_base = os.path.join(word_dir,part)
            db_path = os.path.join(new_base,'hm.db')
            all_access_query = 'SELECT SETIDX,sum(ISMISS) FROM HitMissTrace group by SETIDX,TAG;'
            con = sqlite3.connect(db_path)
            cur = con.cursor()
            f = cur.execute(all_access_query)

            s_dicts['ways_miss_cnt'][ways] = [0 for _ in range(all_set)]
            for setidx,msc in f:
                idx = int(setidx)
                msc = int(msc)
                s_dicts['ways_miss_cnt'][ways][idx] += msc

            cur.close()
        s_dicts['min_ways_no_extra_miss'] = [full_ass for _ in range(all_set)]
        s_dicts['min_ways_1_extra_miss'] = [full_ass for _ in range(all_set)]
        s_dicts['min_ways_2_extra_miss'] = [full_ass for _ in range(all_set)]
        fullass_miss_cnt = s_dicts['ways_miss_cnt'][full_ass]

        for ways in s_dicts['ways_miss_cnt']:
            my_way_miss_cnt = s_dicts['ways_miss_cnt'][ways]
            for idx in range(all_set):
                delta_miss = my_way_miss_cnt[idx] - fullass_miss_cnt[idx]
                if delta_miss <= 0:
                    s_dicts['min_ways_no_extra_miss'][idx] = min(s_dicts['min_ways_no_extra_miss'][idx],ways)
                if delta_miss <= 1:
                    s_dicts['min_ways_1_extra_miss'][idx] = min(s_dicts['min_ways_1_extra_miss'][idx],ways)
                if delta_miss <= 2:
                    s_dicts['min_ways_2_extra_miss'][idx] = min(s_dicts['min_ways_2_extra_miss'][idx],ways)
        draw_one_func(ax_bar,s_dicts,work,full_ass,(fx,fy))

    for i in range(len(worksname_waydict),n_rows*4):
        fx = i // 4
        fy = i % 4
        ax[fx,fy].remove()

    plt.tight_layout()
    plt.savefig(fig_name,dpi=300)
    plt.clf()


if __name__ == '__main__':
    # use_conf = conf_50M
    test_prefix = use_conf['test_prefix']
    base_dir = base_dir_format.format(test_prefix)
    pic_dir_path = f'set_analyze/{test_prefix}pics'
    os.makedirs(pic_dir_path, exist_ok=True)
    worksname = use_conf['cache_work_names'] #like mcf
    cache_work_90perfways = use_conf['cache_work_90perfways']
    cache_work_95perfways = use_conf['cache_work_95perfways']
    cache_work_fullways = use_conf['cache_work_fullways']

    n_works = len(worksname)
    n_rows = math.ceil(n_works/4)
    draw_db_by_func(base_dir,n_rows,cache_work_90perfways,
        draw_one_func=draw_one_workload_way_need,fig_name= os.path.join(pic_dir_path,'way_need_90perf_dis.png'))
    draw_db_by_func(base_dir,n_rows,cache_work_95perfways,
        draw_one_func=draw_one_workload_way_need,fig_name=os.path.join(pic_dir_path,'way_need_95perf_dis.png'))
    draw_db_by_func(base_dir,n_rows,cache_work_fullways,
        draw_one_func=draw_one_workload_way_need,fig_name=os.path.join(pic_dir_path,'way_need_dis.png'))
