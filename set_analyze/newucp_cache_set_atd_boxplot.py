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
parser.add_argument('-j','--json', type=str,
    default=None)

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
    each_way_cnts = s_dicts['each_pos_data']
    x_val = np.arange(1, max_assoc+1)

    #draw boxplot
    bp = ax.boxplot(each_way_cnts, showfliers=False,
                    # patch_artist=True,
                    showmeans=True,
                    # meanline=True,
                capprops=dict(color='black', linewidth=2.5),
                whiskerprops=dict(color='black', linewidth=2.5),
                boxprops=dict(color='black', linewidth=2.5),
                meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor=contrasting_orange[1], markersize=10),
                # meanprops=dict(color='green', linewidth=2.5),
                medianprops=dict(color=contrasting_orange[0], linewidth=3.5),
                flierprops=dict(
                    marker='o',
                    markerfacecolor='black', linewidth=10, markersize=7, markeredgecolor='black'))

    # parts = ax.violinplot(each_way_cnts,x_val, widths = 1, points=256, showmeans=True,showextrema=True)

    
    if pos[1] == 0:
        ax.set_ylabel('hit cnts for each set')
    # ax.set_ylim(0, max_assoc)
    ax.set_xlim(0, max_assoc+1)
    # ax.xaxis.grid(False)
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xticks(x_val, [f'{x-1}' for x in x_val] )
    # ax.set_ylim(bottom=0)
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    # if pos[0] == 1:
    ax.set_xlabel('LRU hit pos')
    ax.set_title(f'{workload_name}')
    # if pos == (0,0):
    #     ax.legend(shadow=0, bbox_to_anchor=(-0.01,1.6), loc = 'upper left',  \
    #         borderaxespad=0.2, ncol = 1, columnspacing=0.5, labelspacing=0.1)
    #     # ax.legend(shadow=0, fontsize = 12, bbox_to_anchor=(-0.01,1.3,0,0), loc = 'upper left',  \
    #     #     borderaxespad=0.2, ncol = 10, columnspacing=0.5, labelspacing=0.1)


def draw_db_by_func(base_dir,n_rows,worksname_waydict,draw_one_func,fig_name,n_cols=6,
                    force_update_json = False):

    work_stats_dict = {}
    json_format = '/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/{}other/json/asplos23_cache_set_full_atd.json'
    json_path = json_format.format(test_prefix)
    
    #options of dict and json
    # force_update_json = True

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



    fig,ax = plt.subplots(n_rows,n_cols)
    
    fig.set_size_inches(n_cols*8, 6*n_rows)

    s_2 = re.compile(r'(\w+)-([\w\.]+)')

    for i,work in enumerate(worksname_waydict):
        work = worksname[i]
        # full_ass = max_assoc
        word_dir = os.path.join(base_dir,work)
        if not os.path.isdir(word_dir):
            continue
        fy = i % n_cols
        fx = i // n_cols
        ax_bar = ax[fx,fy]
        if dict_updated:
            s_dicts = work_stats_dict[work]
        else:
            s_dicts = {}
            # s_dicts['unique_blocks_number'] = [0 for _ in range(all_set)]
            lru_hit_cnts = [np.zeros(max_assoc) for _ in range(all_set)]

            new_base = os.path.join(word_dir,f'l3-{max_assoc}')
            db_path = os.path.join(new_base,'hm.db')
            all_access_query = 'SELECT SETIDX,METAS FROM HitPosTrace WHERE ISINS = 0 ORDER BY ID ASC;'
            con = sqlite3.connect(db_path)
            cur = con.cursor()
            f = cur.execute(all_access_query)

            for setidx,hitpos in f:
                idx = int(setidx)
                hitpos = int(hitpos)
                lru_hit_cnts[idx][hitpos] += 1

            cur.close()

            s_dicts['each_set_atd'] = [l.tolist() for l in lru_hit_cnts]
            each_pos_data = [ [] for _ in range(max_assoc)]
            for lru_pos in lru_hit_cnts:
                for cnts,datas in zip(lru_pos, each_pos_data):
                    datas.append(cnts)

            s_dicts['each_pos_data'] = each_pos_data

            work_stats_dict[work] = s_dicts

        draw_one_func(ax_bar,s_dicts,work,max_assoc,(fx,fy))

    for i in range(len(worksname_waydict),n_rows*n_cols):
        fx = i // n_cols
        fy = i % n_cols
        ax[fx,fy].remove()

    if not dict_updated or force_update_json:
        #save to json
        if json_path is not None:
            jdpath = os.path.dirname(json_path)
            os.makedirs(jdpath,exist_ok=True)
            with open(json_path,'w') as f:
                json.dump(work_stats_dict,f,indent=2)
    
    legends = [Patch(color=contrasting_orange[0],label=f'median val'),
                Patch(color=contrasting_orange[1],label=f'mean val'),
                ]
    fig.legend(handles = legends, loc = 'upper left', ncol = 2 )

    # plt.tight_layout()
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    # plt.tight_layout(pad=0.2)
    plt.savefig(fig_name,dpi=300)
    plt.clf()


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
    draw_db_by_func(base_dir,n_rows,cache_work_95perf_maxfull_ways,
                    n_cols=n_cols,
        draw_one_func=draw_one_workload_way_need,fig_name=os.path.join(pic_dir_path,'newucp_atd_hitcnt_boxplot.png'))


if __name__ == '__main__':
    # base_dir = '/nfs/home/zhangchuanqi/lvna/for_xs/catlog/single-profiling/'
    if opt.json:
        select_json = opt.json
        run_one_conf(select_json)
    else:
        for co in confs:
            select_json = co
            run_one_conf(select_json)