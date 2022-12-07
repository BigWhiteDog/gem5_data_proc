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
import sqlite3


parser = argparse.ArgumentParser(description="options to get set stats")
# parser.add_argument('-d','--stats_dir', type=str,
#     help='stats dir to analyze',required=True)
# parser.add_argument('--ids',default=16,type=int)
# parser.add_argument('--nsamples',default=2,type=int)
# parser.add_argument('--l3_sets',default=4096,type=int)

opt = parser.parse_args()

mycolor = ['#044E48','#06746B','#20876B','#6A9A48','#B5AC23','#E6B80B','#FACA3E','#FFDF80','#FFEBB0']
mycolor = ['#661900','#B22C00','#E6450F','#FF6500','#FF8C00','#FFB200','#FFCB33','#FFDF80','#FDEDBE']


all_set = 16384
tail_set = int(0.001*all_set)
def draw_one_workload_cnthist(ax,cnt_list,workload_names):
    cnt_list.sort()
    cnt_list = cnt_list[tail_set:-tail_set]

    # ax.set_ylabel('Norm IPC')
    # ax.set_ylim(0.5,1)
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.set_xlabel('number of access cnt of a set')
    ax.set_title(f'{workload_names}')
    
    ax.plot(sorted_keys,nipcs,marker='o',label='ipc')



def draw_db_by_func(base_dir,n_rows,worksname,draw_one_func,fig_name):
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
            if ways[1] != '8':
                continue

            new_base = os.path.join(word_dir,part)
            db_path = os.path.join(new_base,'hm.db')
            all_access_query = 'SELECT count(*) FROM HitMissTrace group by SETIDX'
            con = sqlite3.connect(db_path)
            cur = con.cursor()

            f = cur.execute(all_access_query)
            out_res = [x[0] for x in f]
            remain_0 = all_set - len(out_res)
            out_res.extend([0]*remain_0)
            # ways[1] as key

    plt.tight_layout()
    plt.savefig(fig_name,dpi=300)
    plt.clf()

from cache_sensitive_names import *

if __name__ == '__main__':
    base_dir = '/nfs/home/zhangchuanqi/lvna/for_xs/catlog/single-profiling/'
    # worksname = cache_work_names #like mcf
    worksname = os.listdir(base_dir)
    n_works = len(worksname)
    n_rows = math.ceil(n_works/4)
    draw_db_by_func(base_dir,n_rows,worksname,
        draw_one_func=draw_one_workload_ipc,fig_name='single_profiling_access_cnt.png')
