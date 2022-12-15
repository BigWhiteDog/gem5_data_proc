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


def draw_one_workload_len_est(ax,s_dicts,workload_name,full_ass,pos:tuple):
    label_s = ['min_ways_no_extra_miss','min_ways_1_extra_miss', 'min_ways_2_extra_miss',
                'no_extra_miss_hit_len','no_extra_miss_access_len','no_extra_miss_cycle']
    extra0_list = s_dicts[label_s[0]]
    extra1_list = s_dicts[label_s[1]]
    extra2_list = s_dicts[label_s[2]]
    hitlen_list = s_dicts[label_s[3]]
    accesslen_list = s_dicts[label_s[4]]
    cycle_list = s_dicts[label_s[5]]
    sorted_setlist = sorted(zip(extra0_list,extra1_list,extra2_list,hitlen_list,accesslen_list,cycle_list))
    cutlen = all_set
    # for i in range(all_set-1, -1 , -1):
    #     e0,e1,e2,_,_ = sorted_setlist[i]
    #     if e0 != full_ass:
    #         cutlen = i+1
    #         break
    # sorted_setlist = sorted_setlist[:cutlen]
    s_extra0_list,s_extra1_list,s_extra2_list,s_hitlen_list,s_accesslen_list,s_cycle_list = zip(*sorted_setlist)
    x_val = np.arange(cutlen)
    full_ass_vals = np.full(cutlen,full_ass)

    another_s_accesslen_list = sorted(s_accesslen_list)
    tail_accesslen_995 = another_s_accesslen_list[math.ceil(all_set * 0.995)]

    another_s_cycle_list = sorted(s_cycle_list)
    tail_cycle_995 = another_s_cycle_list[math.ceil(all_set * 0.995)]

    extra0_list_color = contrasting_orange[2]
    extra1_list_color = contrasting_orange[3]
    extra2_list_color = contrasting_orange[4]
    hitlen_list_color = contrasting_orange[6]
    accesslen_list_color = contrasting_orange[7]
    alpha_set = 0.8
    ax.plot(s_extra2_list, label='min ways 2 extra miss', color = extra2_list_color,linewidth=1)
    ax.fill_between(x_val,full_ass_vals, s_extra2_list, color = extra2_list_color, alpha=alpha_set)
    ax.plot(s_extra1_list, label='min ways 1 extra miss', color = extra1_list_color,linewidth=1)
    ax.fill_between(x_val, full_ass_vals, s_extra1_list, color = extra1_list_color, alpha=alpha_set)
    ax.plot(s_extra0_list, label='min ways no extra miss', color = extra0_list_color,linewidth=1)
    ax.fill_between(x_val, full_ass_vals, s_extra0_list, color = extra0_list_color, alpha=alpha_set)
    ax.plot(s_hitlen_list, label='hit len', color = hitlen_list_color,linewidth=2)
    ax.plot(s_accesslen_list, label='access len', color = accesslen_list_color,linewidth=2)

    ax.set_ylabel('needed ways/blocks')
    ax.set_ylim(0, tail_accesslen_995)
    ax.yaxis.set_major_locator(MaxNLocator('auto',integer=True))
    # ax.set_ylim(0, 8)
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    ax2 = ax.twinx()
    ax2.plot(s_cycle_list, label='cycle', color = contrasting_orange[8],linewidth=1)
    ax2.set_ylabel('needed cycle')
    ax2.set_ylim(0, tail_cycle_995)
    ax2.yaxis.set_major_locator(MaxNLocator('auto',integer=True))

    ax.set_xlabel('set idx (sorted by atd min 0miss ways)')
    ax.set_title(f'{workload_name}')
    if pos == (0,0):
        ax.legend(shadow=0, fontsize = 13, bbox_to_anchor=(-0.01,1.4), loc = 'upper left',  \
            borderaxespad=0.2, ncol = 1, columnspacing=0.5, labelspacing=0.1)
        # ax.legend(shadow=0, fontsize = 12, bbox_to_anchor=(-0.01,1.3,0,0), loc = 'upper left',  \
        #     borderaxespad=0.2, ncol = 10, columnspacing=0.5, labelspacing=0.1)

def draw_one_workload_len_est_hist(ax,s_dicts,workload_name,full_ass,pos:tuple):
    label_s = ['min_ways_no_extra_miss','min_ways_1_extra_miss', 'min_ways_2_extra_miss',
                'no_extra_miss_hit_len','no_extra_miss_access_len']
    hitlen_list = s_dicts[label_s[3]]
    accesslen_list = s_dicts[label_s[4]]

    hitlen_list_color = contrasting_orange[6]
    accesslen_list_color = contrasting_orange[7]

    hitlen_hist,hitlen_edges = np.histogram(hitlen_list, bins = 'auto',density=True)
    sum_density = np.sum(hitlen_hist)
    hitlen_hist = hitlen_hist/sum_density
    sum_h = 0
    for i,h in enumerate(hitlen_hist):
        sum_h += h
        if sum_h >= 0.995:
            hitlen_list = np.clip(hitlen_list,0,hitlen_edges[i])
            # hitlen_edges = np.delete(hitlen_edges, np.arange( i+1, len(hitlen_edges)-1 ) )
            break
    ax.hist(hitlen_list, bins = 'auto', label='hit len',histtype = 'step', 
            density=True, cumulative=True, color = hitlen_list_color,  linewidth=2)

    accesslen_hist,accesslen_edges = np.histogram(accesslen_list, bins = 'auto',density=True)
    sum_density = np.sum(accesslen_hist)
    accesslen_hist = accesslen_hist/sum_density
    sum_h = 0
    for i,h in enumerate(accesslen_hist):
        sum_h += h
        if sum_h >= 0.99:
            accesslen_list = np.clip(accesslen_list,0,accesslen_edges[i])
            # accesslen_edges = np.delete(accesslen_edges, np.arange( i+1, len(accesslen_edges)-1 ) )
            break
        
    ax.hist(accesslen_list, bins = 'auto', label='access len',histtype = 'step',
            density=True, cumulative=True, color = accesslen_list_color, linewidth=2)



    ax.set_ylabel('portion of sets to be est exactly')
    # ax.set_ylim(0, 8)
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    # ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.set_xlabel('needed blocks')
    ax.set_title(f'{workload_name}')
    if pos == (0,0):
        ax.legend(shadow=0, fontsize = 13, bbox_to_anchor=(-0.01,1.4), loc = 'upper left',  \
            borderaxespad=0.2, ncol = 1, columnspacing=0.5, labelspacing=0.1)
        # ax.legend(shadow=0, fontsize = 12, bbox_to_anchor=(-0.01,1.3,0,0), loc = 'upper left',  \
        #     borderaxespad=0.2, ncol = 10, columnspacing=0.5, labelspacing=0.1)

def draw_one_workload_cycle_hist(ax,s_dicts,workload_name,full_ass,pos:tuple):
    label_s = ['no_extra_miss_cycle']
    cyclelen_list = s_dicts[label_s[0]]

    cyclelen_list_color = contrasting_orange[9]
    s_cyclen_list = np.sort(cyclelen_list)
    tail_cycle_995 = s_cyclen_list[int(0.995*len(s_cyclen_list))]
    cyclelen_list = np.clip(cyclelen_list,0,tail_cycle_995)
    # cyclelen_hist,cyclelen_edges = np.histogram(cyclelen_list, bins = 'auto',density=True)
    # sum_density = np.sum(cyclelen_hist)
    # cyclelen_hist = cyclelen_hist/sum_density
    # sum_h = 0
    # for i,h in enumerate(cyclelen_hist):
    #     sum_h += h
    #     if sum_h >= 0.995:
    #         cyclelen_list = np.clip(cyclelen_list,0,cyclelen_edges[i])
    #         # hitlen_edges = np.delete(hitlen_edges, np.arange( i+1, len(hitlen_edges)-1 ) )
    #         break
    ax.hist(cyclelen_list, bins = 'auto', label='cycle len',histtype = 'step', 
            density=True, cumulative=True, color = cyclelen_list_color,  linewidth=2)

    ax.set_ylabel('portion of sets to be est exactly')
    # ax.set_ylim(0, 8)
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    # ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.set_xlabel('needed cycles')
    ax.set_title(f'{workload_name}')
    if pos == (0,0):
        ax.legend(shadow=0, fontsize = 13, bbox_to_anchor=(-0.01,1.4), loc = 'upper left',  \
            borderaxespad=0.2, ncol = 1, columnspacing=0.5, labelspacing=0.1)
        # ax.legend(shadow=0, fontsize = 12, bbox_to_anchor=(-0.01,1.3,0,0), loc = 'upper left',  \
        #     borderaxespad=0.2, ncol = 10, columnspacing=0.5, labelspacing=0.1)


def analyze_workload_len_est(work_stats_dict,work,work_dir,full_ass):
    if work in work_stats_dict:
        return
    s_2 = re.compile(r'(\w+)-([\w\.]+)')
    s_dicts = {}
    lru_hit_cnts = [np.zeros(full_ass) for _ in range(all_set)]
    hit_len_hit_cnts = [np.zeros(full_ass) for _ in range(all_set)]
    hit_len_access_cnts = [np.zeros(full_ass) for _ in range(all_set)]
    reach_hit_cycles = [np.zeros(full_ass) for _ in range(all_set)]
    set_hit_cnts = np.zeros(all_set)
    set_access_cnts = np.zeros(all_set)

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

        lru_states = [SortedList() for _ in range(all_set)]
        stamp0 = 0
        for idx,tag,stamp in f:
            idx = int(idx)
            tag = int(tag)
            stamp = int(stamp)
            if stamp0 == 0:
                stamp0 = stamp
            delta_stamp = stamp - stamp0
            ls = lru_states[idx]
            fi = filter(lambda x: x[1] == tag, ls)
            res = list(fi)
            set_access_cnts[idx] += 1
            if len(res) > 0:
                #hit
                set_hit_cnts[idx] += 1
                #record hit pos
                get_pair = res[0]
                lru_index = ls.index(get_pair)
                hit_pos = len(ls) - lru_index - 1
                if lru_hit_cnts[idx][hit_pos] == 0:
                    #first reach a hit pos
                    hit_len_hit_cnts[idx][hit_pos] = set_hit_cnts[idx]
                    hit_len_access_cnts[idx][hit_pos] = set_access_cnts[idx]
                    reach_hit_cycles[idx][hit_pos] = delta_stamp
                lru_hit_cnts[idx][hit_pos] += 1
                #modify lru
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
    s_dicts['no_extra_miss_hit_len'] = [0 for _ in range(all_set)]
    s_dicts['no_extra_miss_access_len'] = [0 for _ in range(all_set)]
    s_dicts['no_extra_miss_cycle'] = [0 for _ in range(all_set)]
    

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

        min_ways_0 = s_dicts['min_ways_no_extra_miss'][idx]
        s_dicts['no_extra_miss_hit_len'][idx] = hit_len_hit_cnts[idx][min_ways_0-1]
        s_dicts['no_extra_miss_access_len'][idx] = hit_len_access_cnts[idx][min_ways_0-1]
        s_dicts['no_extra_miss_cycle'][idx] = reach_hit_cycles[idx][min_ways_0-1]

    work_stats_dict[work] = s_dicts

def draw_db_by_func(base_dir,n_rows,worksname_waydict,analyze_func,draw_one_func,fig_name,input_stats_dict=None):
    fig,ax = plt.subplots(n_rows,4)
    fig.set_size_inches(24, 4.5*n_rows+3)

    work_stats_dict = {}
    if input_stats_dict is not None:
        work_stats_dict = input_stats_dict

    for i,work in enumerate(worksname_waydict):
        full_ass = worksname_waydict[work]
        work_dir = os.path.join(base_dir,work)
        if not os.path.isdir(work_dir):
            continue
        fy = i % 4
        fx = i // 4
        ax_bar = ax[fx,fy]
        analyze_func(work_stats_dict,work,work_dir,full_ass)
        s_dicts = work_stats_dict[work]
        draw_one_func(ax_bar,s_dicts,work,full_ass,(fx,fy))     


    plt.tight_layout()
    plt.savefig(fig_name,dpi=300)
    plt.clf()

    return work_stats_dict


if __name__ == '__main__':
    base_dir = '/nfs/home/zhangchuanqi/lvna/for_xs/catlog/single-profiling/'
    worksname = cache_work_names #like mcf
    # worksname = ['sphinx3','mcf'] #like mcf
    # worksname = os.listdir(base_dir)
    n_works = len(worksname)
    n_rows = math.ceil(n_works/4)

    w_dict_90 = draw_db_by_func(base_dir,n_rows,cache_work_90perfways,
        analyze_func=analyze_workload_len_est,
        draw_one_func=draw_one_workload_len_est,fig_name='set_analyze/pics/est_justlen_90perf_dis.png')
    draw_db_by_func(base_dir,n_rows,cache_work_90perfways,
        analyze_func=analyze_workload_len_est,
        draw_one_func=draw_one_workload_len_est_hist,
        fig_name='set_analyze/pics/est_lencdf_90perf_dis.png',
        input_stats_dict=w_dict_90)
    draw_db_by_func(base_dir,n_rows,cache_work_90perfways,
        analyze_func=analyze_workload_len_est,
        draw_one_func=draw_one_workload_cycle_hist,
        fig_name='set_analyze/pics/est_cyclecdf_90perf_dis.png',
        input_stats_dict=w_dict_90)

    w_dict_95 = draw_db_by_func(base_dir,n_rows,cache_work_95perfways,
        analyze_func=analyze_workload_len_est,
        draw_one_func=draw_one_workload_len_est,fig_name='set_analyze/pics/est_justlen_95perf_dis.png')
    draw_db_by_func(base_dir,n_rows,cache_work_95perfways,
        analyze_func=analyze_workload_len_est,
        draw_one_func=draw_one_workload_len_est_hist,
        fig_name='set_analyze/pics/est_lencdf_95perf_dis.png',
        input_stats_dict=w_dict_95)
    draw_db_by_func(base_dir,n_rows,cache_work_95perfways,
        analyze_func=analyze_workload_len_est,
        draw_one_func=draw_one_workload_cycle_hist,
        fig_name='set_analyze/pics/est_cyclecdf_95perf_dis.png',
        input_stats_dict=w_dict_95)

    w_dict_full = draw_db_by_func(base_dir,n_rows,cache_work_fullways,
        analyze_func=analyze_workload_len_est,
        draw_one_func=draw_one_workload_len_est,fig_name='set_analyze/pics/est_justlen_dis.png')
    draw_db_by_func(base_dir,n_rows,cache_work_fullways,
        analyze_func=analyze_workload_len_est,
        draw_one_func=draw_one_workload_len_est_hist,
        fig_name='set_analyze/pics/est_lencdf_dis.png',
        input_stats_dict=w_dict_full)
    draw_db_by_func(base_dir,n_rows,cache_work_fullways,
        analyze_func=analyze_workload_len_est,
        draw_one_func=draw_one_workload_cycle_hist,
        fig_name='set_analyze/pics/est_cyclecdf_dis.png',
        input_stats_dict=w_dict_full)
