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

class SetPositiveState:
    def __init__(self, set_id, full_ass, meta_bits=2, start_postive=1, decrease_f=0.5):
        self.set_id = set_id
        self.full_ass = full_ass
        self.meta_bits = meta_bits
        self.positive_bits = np.full(full_ass, False)
        self.positive_cyclelen_record = {start_postive:0}
        self.positive_total_blocklen = {start_postive:0}
        self.positive_num = start_postive
        for i in range(start_postive):
            self.positive_bits[i] = True
        self.decrease_f = decrease_f
        self.hitlen = 0
        self.blocklen = 0

        self.meta_mask = (1 << meta_bits) - 1

    def newHit(self, way_id, delta_stamp, total_blocklen, total_hitlen):
        self.blocklen += 1
        self.hitlen += 1
        if not self.positive_bits[way_id]:
            #when hit a negative block, there will be new infeciton
            total_less_ways = self.full_ass - self.positive_num
            need_grow_ways = math.ceil(total_less_ways * self.decrease_f)
            self.positive_bits[way_id] = True
            self.positive_num += 1
            need_grow_ways -= 1
            for i in range(self.full_ass):
                if need_grow_ways <= 0:
                    break
                if not self.positive_bits[i]:
                    self.positive_bits[i] = True
                    self.positive_num += 1
                    need_grow_ways -= 1
            self.positive_cyclelen_record[self.positive_num] = delta_stamp
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

        pn_states = [SetPositiveState(i,full_ass,
            start_postive=math.ceil(full_ass/2),
            decrease_f=0.0625,
            ) for i in range(all_set)]
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

    for idx in range(all_set):
        set_pn_state = pn_states[idx]
        s_dicts['est_used_ways'][idx] = max(set_pn_state.positive_num,1)
        pcycle_dict = set_pn_state.positive_cyclelen_record
        s_dicts['half_cycle_est_ways'][idx] = max(
            filter(lambda k: pcycle_dict[k] <= delta_stamp/2 , pcycle_dict)
        )
        pblocklen_dict = set_pn_state.positive_total_blocklen
        s_dicts['half_access_est_ways'][idx] = max(
            filter(lambda k: pblocklen_dict[k] <= totla_block_len/2 , pblocklen_dict)
        )

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
        (draw_one_workload_origin_pn_half,'pn_AIMD_est_half_{}.png',os.path.join(csv_dir_path,'min0way_{}')),
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