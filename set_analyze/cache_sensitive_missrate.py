from genericpath import isdir
import os
import re
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
from matplotlib.patches import Patch
import sqlite3


parser = argparse.ArgumentParser(description="options to get set stats")
# parser.add_argument('-d','--stats_dir', type=str,
#     help='stats dir to analyze',required=True)
# parser.add_argument('--ids',default=16,type=int)
# parser.add_argument('--nsamples',default=2,type=int)
# parser.add_argument('--l3_sets',default=4096,type=int)

opt = parser.parse_args()

from set_analyze.my_diff_color import swatch_colors

all_set = 16384
tail_set = int(0.001*all_set)

def draw_one_workload_miss_sbar(ax,s_dicts,workload_names,pos:tuple):
    sorted_keys = sorted(s_dicts.keys(),key=lambda x:int(x),reverse=True)
    miss_ranges = np.arange(0,1.05,0.1)
    width = 0.35
    new_colors = []
    for i,c in enumerate(swatch_colors):
        # if i % 2 == 0:
        #     new_colors.append(c)
        if i not in [6,7,8]:
            new_colors.append(c)
    for k in sorted_keys:
        sets_dict = s_dicts[k]
        miss_list = []
        sum_allcnt = 0
        sum_misscnt = 0
        for i in range(all_set):
            if i in sets_dict:
                allcnt,misscnt = sets_dict[i]
                sum_allcnt += allcnt
                sum_misscnt += misscnt
                miss_list.append(misscnt/allcnt)
            else:
                miss_list.append(0)
        hist, bin_edges = np.histogram(miss_list,bins=miss_ranges,density=True)
        hist = hist / hist.sum()
        # print(hist,bin_edges)
        x = int(k)
        bot = 0
        bars = []
        for i,h in enumerate(hist):
            bars += ax.bar(x,h,width=width,bottom=bot,
                color = new_colors[i],
                label=[f'{bin_edges[i]:.0%}~{bin_edges[i+1]:.0%}'])
            bot += h
    ax.set_ylabel('set portion')
    ax.set_ylim(0,1)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.set_xticks([int(k) for k in sorted_keys])
    ax.set_xlabel('number of L3 ways (1MB/way)')
    ax.set_title(f'{workload_names}')
    if pos == (0,0):
        # bars_labels = [b.get_label() for b in bars]
        # print(bars_labels)
        bar_patch = [Patch(color=new_colors[i],label=f'{miss_ranges[i]:.0%}~{miss_ranges[i+1]:.0%}')
                    for i in range(len(miss_ranges)-1)]
        ax.legend(handles=bar_patch, shadow=0, fontsize = 12, bbox_to_anchor=(-0.01,1.2,0,0), loc = 'upper left',  \
            borderaxespad=0.2, ncol = 10, columnspacing=0.5, labelspacing=0.1)
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

            new_base = os.path.join(word_dir,part)
            db_path = os.path.join(new_base,'hm.db')
            all_access_query = 'SELECT SETIDX,count(*),sum(ISMISS) FROM HitMissTrace group by SETIDX'
            con = sqlite3.connect(db_path)
            cur = con.cursor()

            f = cur.execute(all_access_query)
            one_out_dict = {}
            for setidx,allcnt,misscnt in f:
                idx = int(setidx)
                one_out_dict[int(setidx)] = (allcnt,misscnt)

            # ways as key
            s_dicts[ways] = one_out_dict
            cur.close()
        draw_one_func(ax_bar,s_dicts,work,(fx,fy))


    # plt.tight_layout()
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
        draw_one_func=draw_one_workload_miss_sbar,fig_name='set_analyze/missrate_dis.png')
