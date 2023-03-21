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

def draw_one_workload_pn_need(ax,s_dicts,workload_name,full_ass,pos:tuple):
    extra0_list = s_dicts['est_used_ways']
    s_extra0_list = sorted(extra0_list)
    x_val = np.arange(all_set)
    full_ass_vals = np.full(all_set,full_ass)

    extra0_list_color = contrasting_orange[6]
    alpha_set = 0.8
    ax.plot(s_extra0_list, label='min est ways', color = extra0_list_color,linewidth=1)
    ax.fill_between(x_val, full_ass_vals, s_extra0_list, color = extra0_list_color, alpha=alpha_set)
    ax.set_ylabel('needed ways')
    ax.set_ylim(0, 8)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xlabel('set number(sorted by min used ways)')
    ax.set_title(f'{workload_name}')
    if pos == (0,0):
        ax.legend(shadow=0, fontsize = 13, bbox_to_anchor=(-0.01,1.4), loc = 'upper left',  \
            borderaxespad=0.2, ncol = 1, columnspacing=0.5, labelspacing=0.1)
        # ax.legend(shadow=0, fontsize = 12, bbox_to_anchor=(-0.01,1.3,0,0), loc = 'upper left',  \
        #     borderaxespad=0.2, ncol = 10, columnspacing=0.5, labelspacing=0.1)

def draw_one_workload_origin_pn_half(ax,s_dicts,workload_name,full_ass,pos:tuple):
    labels = ['min_ways_no_extra_miss','est_used_ways','half_access_est_ways','half_cycle_est_ways']
    # zip and sort
    full_lists = [s_dicts[k] for k in labels]
    sorted_zip_setlist = sorted(zip(*full_lists))
    sorted_setlists = list(zip(*sorted_zip_setlist))

    x_val = np.arange(all_set)
    full_ass_vals = np.full(all_set,full_ass)
    extra0_list_color = contrasting_orange[6]
    alpha_set = 0.8
    s_extra0_list = sorted_setlists[0]
    ax.plot(s_extra0_list, label='min real est ways', color = extra0_list_color,linewidth=1)
    ax.fill_between(x_val, full_ass_vals, s_extra0_list, color = extra0_list_color, alpha=alpha_set)

    for i in range(1,4):
        extra_i_list = sorted_setlists[i]
        extra_i_color = contrasting_orange[6+i]
        ax.plot(extra_i_list, label=labels[i], color=extra_i_color,linewidth=1.5)
    

    ax.set_ylabel('needed ways')
    ax.set_ylim(0, 8)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xlabel('set number(sorted by min used ways)')
    ax.set_title(f'{workload_name}')
    if pos == (0,0):
        ax.legend(shadow=0, fontsize = 13, bbox_to_anchor=(-0.01,1.4), loc = 'upper left',  \
            borderaxespad=0.2, ncol = 1, columnspacing=0.5, labelspacing=0.1)
        # ax.legend(shadow=0, fontsize = 12, bbox_to_anchor=(-0.01,1.3,0,0), loc = 'upper left',  \
        #     borderaxespad=0.2, ncol = 10, columnspacing=0.5, labelspacing=0.1)


def draw_one_workload_pn_hitlen(ax,s_dicts,workload_name,full_ass,pos:tuple):
    ways_len_dicts = s_dicts['est_way_hitlen']
    # zip and sort
    # full_first_len_list = []
    # for i in range(full_ass,0,-1):
        # full_first_len_list.append( ways_len_dicts[i] )
    # sorted_zip_setlist = sorted(zip(*full_first_len_list))
    # sorted_setlists = list(zip(*sorted_zip_setlist))
    # sorted_setlists.reverse()

    #sort w/o zip
    sorted_setlists = []
    for i in range(1,full_ass+1):
        sorted_setlists.append( sorted(ways_len_dicts[i]) )

    tail_len_995 = sorted_setlists[-1][math.ceil(all_set * 0.995)]
    x_val = np.arange(all_set)

    alpha_set = 1
    for i in range(full_ass):
        est_way = i+1
        my_color = leaf_yellow[-(i+1)]
        ax.plot(sorted_setlists[i], label=f'est {est_way}ways',color = my_color,linewidth=1)
        if i > 0:
            ax.fill_between(x_val, sorted_setlists[i-1], sorted_setlists[i], color = my_color, alpha=alpha_set)
        else:
            ax.fill_between(x_val, np.zeros(all_set), sorted_setlists[i], color = my_color, alpha=alpha_set)
    ax.set_ylabel('needed hit blocks accesing the set')
    # ax.set_ylim(0, tail_len_995)
    ax.yaxis.set_major_locator(MaxNLocator('auto',integer=True))
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # ax.set_xlabel('set number(sorted by est needed hit blocks)')
    ax.set_xlabel('set number(sorted)')
    ax.set_title(f'{workload_name}')
    if pos == (0,0):
        ax.legend(shadow=0, fontsize = 13, bbox_to_anchor=(-0.01,1.4), loc = 'upper left',  \
            borderaxespad=0.2, ncol = 3, columnspacing=0.5, labelspacing=0.1)
        # ax.legend(shadow=0, fontsize = 12, bbox_to_anchor=(-0.01,1.3,0,0), loc = 'upper left',  \
        #     borderaxespad=0.2, ncol = 10, columnspacing=0.5, labelspacing=0.1)

def draw_one_workload_pn_blocklen(ax,s_dicts,workload_name,full_ass,pos:tuple):
    ways_len_dicts = s_dicts['est_way_blocklen']
    # zip and sort
    # full_first_len_list = []
    # for i in range(full_ass,0,-1):
        # full_first_len_list.append( ways_len_dicts[i] )
    # sorted_zip_setlist = sorted(zip(*full_first_len_list))
    # sorted_setlists = list(zip(*sorted_zip_setlist))
    # sorted_setlists.reverse()

    #sort w/o zip
    sorted_setlists = []
    for i in range(1,full_ass+1):
        sorted_setlists.append( sorted(ways_len_dicts[i]) )

    tail_len_995 = sorted_setlists[-1][math.ceil(all_set * 0.995)]
    x_val = np.arange(all_set)

    alpha_set = 1
    for i in range(full_ass):
        est_way = i+1
        my_color = sunset_oranges[-(i+1)]
        ax.plot(sorted_setlists[i], label=f'est {est_way}ways',color = my_color,linewidth=1)
        if i > 0:
            ax.fill_between(x_val, sorted_setlists[i-1], sorted_setlists[i], color = my_color, alpha=alpha_set)
        else:
            ax.fill_between(x_val, np.zeros(all_set), sorted_setlists[i], color = my_color, alpha=alpha_set)
    ax.set_ylabel('needed blocks accesing the set')
    # ax.set_ylim(0, tail_len_995)
    ax.yaxis.set_major_locator(MaxNLocator('auto',integer=True))
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # ax.set_xlabel('set number(sorted by est needed blocks)')
    ax.set_xlabel('set number(sorted)')
    ax.set_title(f'{workload_name}')
    if pos == (0,0):
        ax.legend(shadow=0, fontsize = 13, bbox_to_anchor=(-0.01,1.4), loc = 'upper left',  \
            borderaxespad=0.2, ncol = 2, columnspacing=0.5, labelspacing=0.1)
        # ax.legend(shadow=0, fontsize = 12, bbox_to_anchor=(-0.01,1.3,0,0), loc = 'upper left',  \
        #     borderaxespad=0.2, ncol = 10, columnspacing=0.5, labelspacing=0.1)

def draw_one_workload_pn_cyclelen(ax,s_dicts,workload_name,full_ass,pos:tuple):
    ways_len_dicts = s_dicts['est_way_cyclelen']
    # zip and sort
    # full_first_len_list = []
    # for i in range(full_ass,0,-1):
        # full_first_len_list.append( ways_len_dicts[i] )
    # sorted_zip_setlist = sorted(zip(*full_first_len_list))
    # sorted_setlists = list(zip(*sorted_zip_setlist))
    # sorted_setlists.reverse()

    #sort w/o zip
    sorted_setlists = []
    for i in range(1,full_ass+1):
        sorted_setlists.append( sorted(ways_len_dicts[i]) )

    tail_len_995 = sorted_setlists[-1][math.ceil(all_set * 0.995)]
    x_val = np.arange(all_set)

    alpha_set = 1
    for i in range(full_ass):
        est_way = i+1
        my_color = geek_blue[-(i+1)]
        ax.plot(sorted_setlists[i], label=f'est {est_way}ways',color = my_color,linewidth=1)
        if i > 0:
            ax.fill_between(x_val, sorted_setlists[i-1], sorted_setlists[i], color = my_color, alpha=alpha_set)
        else:
            ax.fill_between(x_val, np.zeros(all_set), sorted_setlists[i], color = my_color, alpha=alpha_set)
    ax.set_ylabel('needed cycles')
    # ax.set_ylim(0, tail_len_995)
    # ax.yaxis.set_yscale('log')
    ax.yaxis.set_major_locator(MaxNLocator('auto',integer=True))
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # ax.set_xlabel('set number(sorted by est needed cycles)')
    ax.set_xlabel('set number(sorted)')
    ax.set_title(f'{workload_name}')
    if pos == (0,0):
        ax.legend(shadow=0, fontsize = 13, bbox_to_anchor=(-0.01,1.4), loc = 'upper left',  \
            borderaxespad=0.2, ncol = 2, columnspacing=0.5, labelspacing=0.1)
        # ax.legend(shadow=0, fontsize = 12, bbox_to_anchor=(-0.01,1.3,0,0), loc = 'upper left',  \
        #     borderaxespad=0.2, ncol = 10, columnspacing=0.5, labelspacing=0.1)

def draw_one_workload_pn_total_hitlen(ax,s_dicts,workload_name,full_ass,pos:tuple):
    ways_len_dicts = s_dicts['est_way_total_hitlen']
    sorted_setlists = []
    for i in range(1,full_ass+1):
        sorted_setlists.append( sorted(ways_len_dicts[i]) )
    tail_len_995 = sorted_setlists[-1][math.ceil(all_set * 0.995)]
    x_val = np.arange(all_set)

    alpha_set = 1
    for i in range(full_ass):
        est_way = i+1
        my_color = leaf_yellow[-(i+1)]
        ax.plot(sorted_setlists[i], label=f'est {est_way}ways',color = my_color,linewidth=1)
        if i > 0:
            ax.fill_between(x_val, sorted_setlists[i-1], sorted_setlists[i], color = my_color, alpha=alpha_set)
        else:
            ax.fill_between(x_val, np.zeros(all_set), sorted_setlists[i], color = my_color, alpha=alpha_set)
    ax.set_ylabel('needed total hit blocks')
    # ax.set_ylim(0, tail_len_995)
    ax.yaxis.set_major_locator(MaxNLocator('auto',integer=True))
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # ax.set_xlabel('set number(sorted by est needed blocks)')
    ax.set_xlabel('set number(sorted)')
    ax.set_title(f'{workload_name}')
    if pos == (0,0):
        ax.legend(shadow=0, fontsize = 13, bbox_to_anchor=(-0.01,1.4), loc = 'upper left',  \
            borderaxespad=0.2, ncol = 2, columnspacing=0.5, labelspacing=0.1)

def draw_one_workload_pn_total_blocklen(ax,s_dicts,workload_name,full_ass,pos:tuple):
    ways_len_dicts = s_dicts['est_way_total_blocklen']

    #sort w/o zip
    sorted_setlists = []
    for i in range(1,full_ass+1):
        sorted_setlists.append( sorted(ways_len_dicts[i]) )

    tail_len_995 = sorted_setlists[-1][math.ceil(all_set * 0.995)]
    x_val = np.arange(all_set)

    alpha_set = 1
    for i in range(full_ass):
        est_way = i+1
        my_color = sunset_oranges[-(i+1)]
        ax.plot(sorted_setlists[i], label=f'est {est_way}ways',color = my_color,linewidth=1)
        if i > 0:
            ax.fill_between(x_val, sorted_setlists[i-1], sorted_setlists[i], color = my_color, alpha=alpha_set)
        else:
            ax.fill_between(x_val, np.zeros(all_set), sorted_setlists[i], color = my_color, alpha=alpha_set)
    ax.set_ylabel('needed total blocks')
    # ax.set_ylim(0, tail_len_995)
    ax.yaxis.set_major_locator(MaxNLocator('auto',integer=True))
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # ax.set_xlabel('set number(sorted by est needed blocks)')
    ax.set_xlabel('set number(sorted)')
    ax.set_title(f'{workload_name}')
    if pos == (0,0):
        ax.legend(shadow=0, fontsize = 13, bbox_to_anchor=(-0.01,1.4), loc = 'upper left',  \
            borderaxespad=0.2, ncol = 2, columnspacing=0.5, labelspacing=0.1)

class SetPositiveState:
    def __init__(self, set_id, full_ass, meta_bits=2):
        self.set_id = set_id
        self.full_ass = full_ass
        self.meta_bits = meta_bits
        self.positive_bits = np.full(full_ass, False)
        self.positive_num = 0
        self.positive_hitlen_record = np.zeros(full_ass+1)
        self.positive_blocklen_record = np.zeros(full_ass+1)
        self.positive_cyclelen_record = np.zeros(full_ass+1)
        self.positive_total_hitlen = np.zeros(full_ass+1)
        self.positive_total_blocklen = np.zeros(full_ass+1)
        self.hitlen = 0
        self.blocklen = 0

        self.meta_mask = (1 << meta_bits) - 1

    def newHit(self, way_id, delta_stamp, total_blocklen, total_hitlen):
        self.blocklen += 1
        self.hitlen += 1
        if not self.positive_bits[way_id]:
            #when hit a negative block, there will be new infeciton
            self.positive_bits[way_id] = True
            self.positive_num += 1
            self.positive_hitlen_record[self.positive_num] = self.hitlen
            self.positive_blocklen_record[self.positive_num] = self.blocklen
            self.positive_cyclelen_record[self.positive_num] = delta_stamp
            self.positive_total_hitlen[self.positive_num] = total_hitlen
            self.positive_total_blocklen[self.positive_num] = total_blocklen

    def newBlock(self, way_id, meta_datas):
        self.blocklen += 1
        max_positive_idx = -1
        max_positive_value = -1
        if not self.positive_bits[way_id]:
            for i in range(self.full_ass):
                #find max metadata positive block
                if self.positive_bits[i]:
                    tmp_meta = meta_datas & self.meta_mask
                    if tmp_meta > max_positive_value:
                        max_positive_value = tmp_meta
                        max_positive_idx = i
                meta_datas >>= self.meta_bits
            if max_positive_idx >= 0:
                self.positive_bits[max_positive_idx] = False
            self.positive_bits[way_id] = True
                    

            

def analyze_pn_lencycle_est(work_stats_dict,work,work_dir,full_ass):
    if work in work_stats_dict:
        return
    s_2 = re.compile(r'(\w+)-([\w\.]+)')
    s_dicts = {}

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
        all_access_query = 'SELECT SETIDX,WAYIDX,ISINS,METAS,STAMP FROM HitPosTrace;'
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        f = cur.execute(all_access_query)

        pn_states = [SetPositiveState(i,full_ass) for i in range(all_set)]
        stamp0 = 0
        total_hit_len = 0
        totla_block_len = 0
        for setidx,wayidx,isins,metas,stamp in f:
            setidx = int(setidx)
            wayidx = int(wayidx)
            isins = bool(int(isins))
            metas = int(metas)
            stamp = int(stamp)
            if stamp0 == 0:
                stamp0 = stamp
            delta_stamp = stamp - stamp0
            totla_block_len += 1
            if isins:
                #insert block
                pn_states[setidx].newBlock(wayidx,metas)
            else:
                #hit block
                total_hit_len += 1
                pn_states[setidx].newHit(wayidx,delta_stamp,
                                totla_block_len,total_hit_len)

        cur.close()
    s_dicts['est_used_ways'] = [1 for _ in range(all_set)]
    s_dicts['half_cycle_est_ways'] = [1 for _ in range(all_set)]
    s_dicts['half_access_est_ways'] = [1 for _ in range(all_set)]
    s_dicts['est_way_hitlen'] = {}
    s_dicts['est_way_blocklen'] = {}
    s_dicts['est_way_cyclelen'] = {}
    s_dicts['est_way_total_hitlen'] = {}
    s_dicts['est_way_total_blocklen'] = {}
    for w in range(1,full_ass+1):
        s_dicts['est_way_hitlen'][w] = np.zeros(all_set)
        s_dicts['est_way_blocklen'][w] = np.zeros(all_set)
        s_dicts['est_way_cyclelen'][w] = np.zeros(all_set)
        s_dicts['est_way_total_hitlen'][w] = np.zeros(all_set)
        s_dicts['est_way_total_blocklen'][w] = np.zeros(all_set)


    for idx in range(all_set):
        set_pn_state = pn_states[idx]
        s_dicts['est_used_ways'][idx] = max(set_pn_state.positive_num,1)
        for pos_num in range(1,s_dicts['est_used_ways'][idx]):
            if set_pn_state.positive_total_blocklen[pos_num] <= totla_block_len / 2:
                s_dicts['half_access_est_ways'][idx] = pos_num
            if set_pn_state.positive_cyclelen_record[pos_num] <= delta_stamp / 2:
                s_dicts['half_cycle_est_ways'][idx] = pos_num
        for pos_num in range(1,full_ass+1):
            used_pos_num = min(pos_num,set_pn_state.positive_num)
            s_dicts['est_way_hitlen'][pos_num][idx] = set_pn_state.positive_hitlen_record[used_pos_num]
            s_dicts['est_way_blocklen'][pos_num][idx] = set_pn_state.positive_blocklen_record[used_pos_num]
            s_dicts['est_way_cyclelen'][pos_num][idx] = set_pn_state.positive_cyclelen_record[used_pos_num]
            s_dicts['est_way_total_hitlen'][pos_num][idx] = set_pn_state.positive_total_hitlen[used_pos_num]
            s_dicts['est_way_total_blocklen'][pos_num][idx] = set_pn_state.positive_total_blocklen[used_pos_num]

    work_stats_dict[work] = s_dicts

def draw_db_by_func(base_dir,n_rows,worksname_waydict,analyze_func,draw_one_func,fig_name,
    csv_top_dir=None,
    input_stats_dict=None):
    fig,ax = plt.subplots(n_rows,4)
    fig.set_size_inches(24, 4.5*n_rows+3)

    # work_stats_dict = {}
    # if input_stats_dict is not None:
    #     work_stats_dict = input_stats_dict
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
        s_dicts['min_ways_no_extra_miss'] = []
        csv_file = os.path.join(csv_top_dir,f'{work}.csv')
        with open(csv_file,'r') as f:
            for i in range(all_set):
                s_dicts['min_ways_no_extra_miss'].append(int(f.readline().strip()))
        draw_one_func(ax_bar,s_dicts,work,full_ass,(fx,fy))     

    for i in range(len(worksname_waydict),n_rows*4):
        fx = i // 4
        fy = i % 4
        ax[fx,fy].remove()

    plt.tight_layout()
    plt.savefig(fig_name,dpi=300)
    plt.clf()

    return work_stats_dict


if __name__ == '__main__':
    use_conf = conf_50M
    test_prefix = use_conf['test_prefix']
    base_dir = base_dir_format.format(test_prefix)
    csv_dir_path = f'set_analyze/{test_prefix}other/csv'
    pic_dir_path = f'set_analyze/{test_prefix}pics'
    os.makedirs(pic_dir_path, exist_ok=True)
    worksname = use_conf['cache_work_names'] #like mcf

    n_works = len(worksname)
    n_rows = math.ceil(n_works/4)

    waydict_format = 'cache_work_{}ways'
    perf_prefixs = ['90perf','95perf','full']
    draw_picformat = [
        # (draw_one_workload_pn_need,'pn_est_wayneed_{}.png',os.path.join(csv_dir_path,'min0way_{}')),
        # (draw_one_workload_pn_hitlen,'pn_est_hitlen_contour_{}.png',os.path.join(csv_dir_path,'min0way_{}')),
        # (draw_one_workload_pn_blocklen,'pn_est_blocklen_contour_{}.png',os.path.join(csv_dir_path,'min0way_{}')),
        # (draw_one_workload_pn_cyclelen,'pn_est_cyclelen_contour_{}.png',os.path.join(csv_dir_path,'min0way_{}')),
        # (draw_one_workload_pn_total_hitlen,'pn_est_total_hitlen_contour_{}.png',os.path.join(csv_dir_path,'min0way_{}')),
        # (draw_one_workload_pn_total_blocklen,'pn_est_total_blocklen_contour_{}.png',os.path.join(csv_dir_path,'min0way_{}')),
        (draw_one_workload_origin_pn_half,'pn_est_half_{}.png',os.path.join(csv_dir_path,'min0way_{}')),
    ]

    for perf_prefix in perf_prefixs:
        waydict_name = waydict_format.format(perf_prefix)
        waydict = use_conf[waydict_name]
        ret_dict = {}
        for draw_func,pic_name_format,csv_dir_format in draw_picformat:
            csv_dir = csv_dir_format.format(perf_prefix)
            draw_db_by_func(base_dir,n_rows,waydict,
                analyze_func=analyze_pn_lencycle_est,
                draw_one_func=draw_func,
                fig_name=os.path.join(pic_dir_path,pic_name_format.format(perf_prefix)),
                csv_top_dir=csv_dir,
                input_stats_dict=ret_dict)