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
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Patch
import sqlite3


parser = argparse.ArgumentParser(description="options to get set stats")
parser.add_argument('-j','--json', type=str,
    default=None)

opt = parser.parse_args()

confs=[
    # "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_xs_tailbm50M.json",
    "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_goldencove_tailbm50M.json",
    # "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_goldencove24M_tailbm50M.json",
    # "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_goldencove48M_tailbm50M.json",
    # "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_oldinc_tailbm50M.json",
    # "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_skylake_tailbm50M.json",
]

from set_analyze.my_diff_color import *



def draw_one_workload_atd_est(ax,s_dicts,workload_name,full_ass,pos:tuple):
    sorted_tail_utility = np.array(s_dicts['sorted_tail_utility'])
    wakeup_seq_tail_utility = np.array(s_dicts['wakeup_seq_tail_utility'])
    plain_tail_utility = np.array(s_dicts['tail_utility'])
    cum_sorted_tail_utility = np.cumsum(sorted_tail_utility)
    cum_wakeup_seq_tail_utility = np.cumsum(wakeup_seq_tail_utility)
    cum_plain_tail_utility = np.cumsum(plain_tail_utility)
    
    x_val = np.arange(all_set)
    ax.plot(cum_sorted_tail_utility, label = 'Sorted by utility', color = contrasting_orange[2], linewidth = 3)    
    ax.plot(cum_wakeup_seq_tail_utility, label = 'Sorted by wakeup seq', color = contrasting_orange[3], linewidth = 3)
    ax.plot(cum_plain_tail_utility, label = 'unsorted set index', color = contrasting_orange[4], linewidth = 3)
    if pos[0] == 1:
        ax.set_xlabel('Set sequence number')
    ax.set_xlim(0,all_set)
    ax.set_xticks(range(0,all_set,5000))
    ax.set_ylim(0,None)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)
    ax.set_ylabel('Cum utility')
    ax.set_title(f'{workload_name}')
    # if pos == (0,0):
    #     ax.legend(shadow=0, fontsize = 13, bbox_to_anchor=(-0.01,1.4), loc = 'upper left',  \
    #         borderaxespad=0.2, ncol = 1, columnspacing=0.5, labelspacing=0.1)
        # ax.legend(shadow=0, fontsize = 12, bbox_to_anchor=(-0.01,1.3,0,0), loc = 'upper left',  \
        #     borderaxespad=0.2, ncol = 10, columnspacing=0.5, labelspacing=0.1)

def draw_one_workload_nolabel(ax,s_dicts,workload_name,full_ass,pos:tuple):
    sorted_tail_utility = np.array(s_dicts['sorted_tail_utility'])
    cum_sorted_tail_utility = np.cumsum(sorted_tail_utility)
    
    x_val = np.arange(all_set)
    ax.plot(cum_sorted_tail_utility, label = 'Sorted by utility', color = contrasting_orange[2], linewidth = 5)    
    # ax.set_xlabel('Number of Sets')
    ax.set_xlim(0,all_set)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0,None)
    # ax.set_ylabel('Cum utility')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.set_title(f'{workload_name}')
    # if pos == (0,0):
    #     ax.legend(shadow=0, fontsize = 13, bbox_to_anchor=(-0.01,1.4), loc = 'upper left',  \
    #         borderaxespad=0.2, ncol = 1, columnspacing=0.5, labelspacing=0.1)
        # ax.legend(shadow=0, fontsize = 12, bbox_to_anchor=(-0.01,1.3,0,0), loc = 'upper left',  \
        #     borderaxespad=0.2, ncol = 10, columnspacing=0.5, labelspacing=0.1)


def draw_db_by_func(base_dir,n_rows,worksname_waydict,draw_one_func,fig_name,n_cols=6,
                    force_update_json = False):

    work_stats_dict = {}
    json_format = '/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/{}other/json/europar24_tail_utility.json'
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



    # fig,ax = plt.subplots(n_rows,n_cols, sharex=True, sharey=True)
    fig,ax = plt.subplots(n_rows,n_cols, sharex=True)
    
    fig.set_size_inches(n_cols*6, 3.5*n_rows)

    s_2 = re.compile(r'(\w+)-([\w\.]+)')

    for i,work in enumerate(worksname_waydict):
        work = worksname[i]
        full_ass = worksname_waydict[work]
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
        
        lru_hit_cnts = np.zeros((all_set, full_ass))
        set_hit_cnts = np.zeros(all_set)

        tail_hited_flag = np.zeros(all_set, dtype=bool)
        tail_hited_seq = list()

        part = f'l3-{full_ass}'
        
        new_base = os.path.join(word_dir,part)
        db_path = os.path.join(new_base,'hm.db')
        all_access_query = 'SELECT SETIDX,TAG FROM HitMissTrace ORDER BY ID ASC;'
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        f = cur.execute(all_access_query)

        lru_states = [list() for _ in range(all_set)]
        for idx,tag in f:
            idx = int(idx)
            tag = int(tag)
            ls = lru_states[idx]
            if tag in ls:
                #hit
                set_hit_cnts[idx] += 1
                lru_index = ls.index(tag)
                hit_pos = len(ls) - lru_index - 1
                lru_hit_cnts[idx,hit_pos] += 1
                ls.pop(lru_index)
                ls.append(tag)
                if (hit_pos == full_ass-1):
                    if not tail_hited_flag[idx]:
                        tail_hited_flag[idx] = True
                        tail_hited_seq.append(idx)
            else:
                #miss
                if len(ls) >= full_ass:
                    ls.pop(0)
                ls.append(tag)

        cur.close()
        tail_utility = lru_hit_cnts[:,full_ass-1]
        s_dicts['tail_utility'] = tail_utility.tolist()
        s_dicts['wakeup_seq'] = tail_hited_seq
        s_dicts['sorted_tail_utility'] = np.sort(tail_utility)[::-1].tolist()
        wakeup_seq_tail_utility = np.zeros(all_set)
        for wi,sidx in enumerate(tail_hited_seq):
            wakeup_seq_tail_utility[wi] = tail_utility[sidx]
        s_dicts['wakeup_seq_tail_utility'] = wakeup_seq_tail_utility.tolist()

        draw_one_func(ax_bar,s_dicts,work,full_ass,(fx,fy))

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
    
    legends = [ Patch(color=contrasting_orange[2],label='Sorted by utility (desc)'),
                Patch(color=contrasting_orange[3],label=f'Sorted by wakeup seq'),
                Patch(color=contrasting_orange[4],label=f'Unsorted index'),
                ]
    fig.legend(handles = legends, loc = 'upper left', ncol = 3,borderaxespad=0, labelspacing=0, handlelength=0.5)

    plt.tight_layout(rect=(0, 0, 1, 0.93))
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
    worksname.remove('lbm')
    cache_work_95perf_maxfull_ways.pop('lbm')
    work_dict = {}
    # interest_list = [
    #     'cam4','omnetpp','parest','xalancbmk',
    #     'tc','imgdnn','sphinx','xapian'
    # ]
    # worksname = interest_list
    interest_list = worksname
    for w in interest_list:
        work_dict[w] = cache_work_95perf_maxfull_ways[w]


    n_works = len(work_dict)
    n_cols = 6
    # n_cols = 4
    n_rows = math.ceil(n_works/n_cols)
    # draw_db_by_func(base_dir,n_rows,cache_work_90perfways,
    #     draw_one_func=draw_one_workload_way_need,fig_name= os.path.join(pic_dir_path,'way_need_90perf_dis.png'))
    # draw_db_by_func(base_dir,n_rows,cache_work_95perfways,
    #     draw_one_func=draw_one_workload_way_need,fig_name=os.path.join(pic_dir_path,'way_need_95perf_dis.png'))
    # draw_db_by_func(base_dir,n_rows,cache_work_fullways,
    #     draw_one_func=draw_one_workload_way_need,fig_name=os.path.join(pic_dir_path,'way_need_dis.png'))
    draw_db_by_func(base_dir,n_rows,work_dict,
                    n_cols=n_cols,
        draw_one_func=draw_one_workload_atd_est,fig_name=os.path.join(pic_dir_path,'hpcc24_tail_utility_95perf.png'))
    # draw_db_by_func(base_dir,n_rows,cache_work_95perf_maxfull_ways,
    #                 n_cols=n_cols,
    #     draw_one_func=draw_one_workload_nolabel,fig_name=os.path.join(pic_dir_path,'europar24_tail_utility_nolabel.png'))



if __name__ == '__main__':
    # base_dir = '/nfs/home/zhangchuanqi/lvna/for_xs/catlog/single-profiling/'
    if opt.json:
        select_json = opt.json
        run_one_conf(select_json)
    else:
        for co in confs:
            select_json = co
            run_one_conf(select_json)