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

from set_analyze.my_diff_color import *

all_set = 16384
tail_set = int(0.001*all_set)

def draw_one_workload_block_need(ax,s_dicts,workload_names,pos:tuple):
    label_s = ['unique_blocks_number', 'unique_reused_blocks_number']
    unique_list = s_dicts[label_s[0]]
    reused_list = s_dicts[label_s[1]]
    sorted_setlist = sorted(zip(unique_list,reused_list))
    s_unique_list,s_reused_list = zip(*sorted_setlist)
    x_val = np.arange(all_set)

    unique_list_color = contrasting_orange[0]
    reused_list_color = contrasting_orange[1]
    alpha_set = 0.8
    ax.plot(s_unique_list, label='unique blocks number',color = unique_list_color)
    ax.fill_between(x_val, s_unique_list, color = unique_list_color, alpha=alpha_set)
    ax.plot(s_reused_list, label='unique reused blocks number', color = reused_list_color)
    ax.fill_between(x_val, s_reused_list, color = reused_list_color, alpha=alpha_set)
    ax.set_ylabel('unique block numbers')
    # ax.set_ylim(0, 30)
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.set_xlabel('set idx (sorted by uniqueNumber)')
    ax.set_title(f'{workload_names}')
    if pos == (0,0):
        ax.legend(shadow=0, fontsize = 13, bbox_to_anchor=(-0.01,1.2,0,0), loc = 'upper left',  \
            borderaxespad=0.2, ncol = 1, columnspacing=0.5, labelspacing=0.1)
        # ax.legend(shadow=0, fontsize = 12, bbox_to_anchor=(-0.01,1.3,0,0), loc = 'upper left',  \
        #     borderaxespad=0.2, ncol = 10, columnspacing=0.5, labelspacing=0.1)


def draw_db_by_func(base_dir,n_rows,worksname,draw_one_func,fig_name):
    fig,ax = plt.subplots(n_rows,4)
    fig.set_size_inches(24, 4.5*n_rows+3)

    s_2 = re.compile(r'(\w+)-([\w\.]+)')

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
            res = s_2.search(part)
            if not res:
                continue
            if res.group(1) != 'l3':
                continue
            ways = res.group(2)
            if ways != '8':
                continue

            new_base = os.path.join(word_dir,part)
            db_path = os.path.join(new_base,'hm.db')
            all_access_query = 'SELECT SETIDX,TAG,count(*) FROM HitMissTrace group by SETIDX,TAG;'
            con = sqlite3.connect(db_path)
            cur = con.cursor()

            f = cur.execute(all_access_query)
            s_dicts['unique_blocks_number'] = [0 for _ in range(all_set)]
            s_dicts['unique_reused_blocks_number'] = [0 for _ in range(all_set)]

            for setidx,tag,cnt in f:
                idx = int(setidx)
                ucnt = int(cnt)
                s_dicts['unique_blocks_number'][idx] += 1
                if ucnt > 1:
                    s_dicts['unique_reused_blocks_number'][idx] += 1

            cur.close()
        draw_one_func(ax_bar,s_dicts,work,(fx,fy))


    plt.tight_layout()
    plt.savefig(fig_name,dpi=300)
    plt.clf()

from cache_sensitive_names import *

if __name__ == '__main__':
    base_dir = '/nfs/home/zhangchuanqi/lvna/for_xs/catlog/single-profiling/'
    worksname = cache_work_names #like mcf
    # worksname = ['sphinx3','mcf'] #like mcf
    # worksname = os.listdir(base_dir)
    n_works = len(worksname)
    n_rows = math.ceil(n_works/4)
    draw_db_by_func(base_dir,n_rows,worksname,
        draw_one_func=draw_one_workload_block_need,fig_name='set_analyze/set_need_dis.png')
