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


def draw_growlist(ax,s_dicts,workload_name,full_ass,pos:tuple):
    # s_dicts['set_grow_lists'] = [lru_states[s].way_grow_list for s in range(all_set)]
    # s_dicts['min_ways_no_extra_miss'] = [full_ass for _ in range(all_set)]
    set_grow_lists = s_dicts['set_grow_lists']
    ax.set_title(f'{workload_name}')
    if pos == (0,0):
        ax.legend(shadow=0, fontsize = 13, bbox_to_anchor=(-0.01,1.4), loc = 'upper left',  \
            borderaxespad=0.2, ncol = 1, columnspacing=0.5, labelspacing=0.1)
        # ax.legend(shadow=0, fontsize = 12, bbox_to_anchor=(-0.01,1.3,0,0), loc = 'upper left',  \
        #     borderaxespad=0.2, ncol = 10, columnspacing=0.5, labelspacing=0.1)

class SetLRUStates:
    def __init__(self, set_id, full_ass):
        self.set_id = set_id
        self.full_ass = full_ass
        self.continuous_mru = 0
        self.mru_hit_cnts = [0 for _ in range(full_ass)]
        self.hit_len_hit_cnts = [0 for _ in range(full_ass)]
        self.hit_len_access_cnts = [0 for _ in range(full_ass)]
        self.reach_hit_cycles = [0 for _ in range(full_ass)]
        self.hit_cnts = 0
        self.access_cnts = 0
        self.lru_states = SortedList()
        self.way_grow_list = []

    def newcome(self, tag, stamp):
        self.access_cnts += 1
        ls = self.lru_states
        fi = filter(lambda x: x[1] == tag, ls)
        res = list(fi)
        if len(res) > 0:
            #hit
            self.hit_cnts += 1
            #record hit pos
            get_pair = res[0]
            lru_index = ls.index(get_pair)
            hit_pos = len(ls) - lru_index - 1
            if self.mru_hit_cnts[hit_pos] == 0:
                #first reach a hit pos
                self.hit_len_hit_cnts[hit_pos] = self.hit_cnts
                self.hit_len_access_cnts[hit_pos] = self.access_cnts
                self.reach_hit_cycles[hit_pos] = stamp
                self.way_grow_list.append(hit_pos)
            self.mru_hit_cnts[hit_pos] += 1
            #modify lru
            ls.remove(get_pair)
            ls.add([stamp,tag])
        else:
            #miss
            if len(ls) >= self.full_ass:
                ls.pop(0)
            ls.add([stamp,tag])

def analyze_workload_len_est(work_stats_dict,work,work_dir,full_ass):
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
        all_access_query = 'SELECT SETIDX,TAG,STAMP FROM HitMissTrace;'
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        f = cur.execute(all_access_query)

        lru_states = [SetLRUStates(se,full_ass) for se in range(all_set)]
        stamp0 = 0
        for idx,tag,stamp in f:
            idx = int(idx)
            tag = int(tag)
            stamp = int(stamp)
            if stamp0 == 0:
                stamp0 = stamp
            delta_stamp = stamp - stamp0
            lru_states[idx].newcome(tag,delta_stamp)

        cur.close()
    s_dicts['min_ways_no_extra_miss'] = [full_ass for _ in range(all_set)]
    s_dicts['min_ways_1_extra_miss'] = [full_ass for _ in range(all_set)]
    s_dicts['min_ways_2_extra_miss'] = [full_ass for _ in range(all_set)]
    s_dicts['no_extra_miss_hit_len'] = [0 for _ in range(all_set)]
    s_dicts['no_extra_miss_access_len'] = [0 for _ in range(all_set)]
    s_dicts['no_extra_miss_cycle'] = [0 for _ in range(all_set)]

    s_dicts['set_grow_lists'] = [lru_states[s].way_grow_list for s in range(all_set)]

    for idx in range(all_set):
        hitpos_cnts = lru_states[idx].mru_hit_cnts
        set_hit_cnt = lru_states[idx].hit_cnts
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
        s_dicts['no_extra_miss_hit_len'][idx] = lru_states[idx].hit_len_hit_cnts[min_ways_0-1]
        s_dicts['no_extra_miss_access_len'][idx] = lru_states[idx].hit_len_access_cnts[min_ways_0-1]
        s_dicts['no_extra_miss_cycle'][idx] = lru_states[idx].reach_hit_cycles[min_ways_0-1]

    work_stats_dict[work] = s_dicts

def draw_db_by_func(base_dir,n_rows,worksname_waydict,analyze_func,draw_one_func,fig_name,input_stats_dict=None,
    json_path=None,force_update_json = False):
    fig,ax = plt.subplots(n_rows,4)
    fig.set_size_inches(24, 4.5*n_rows+3)

    work_stats_dict = {}
    dict_updated = False
    if input_stats_dict is not None:
        work_stats_dict = input_stats_dict
        if len(input_stats_dict) > 0:
            #it has data
            dict_updated = True

    if not dict_updated:
        #try load from json
        if json_path is not None and os.path.isfile(json_path) and not force_update_json:
            with open(json_path,'r') as f:
                json_dict = json.load(f)
                if len(json_dict) > 0:
                    #it has data
                    work_stats_dict.update(json_dict)
                    dict_updated = True

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

    for i in range(len(worksname_waydict),n_rows*4):
        fx = i // 4
        fy = i % 4
        ax[fx,fy].remove()

    plt.tight_layout()
    plt.savefig(fig_name,dpi=300)
    plt.clf()

    if not dict_updated or force_update_json:
        #save to json
        if json_path is not None:
            with open(json_path,'w') as f:
                json.dump(work_stats_dict,f)

    return work_stats_dict


if __name__ == '__main__':
    use_conf = conf_50M
    test_prefix = use_conf['test_prefix']
    base_dir = base_dir_format.format(test_prefix)
    pic_dir_path = f'set_analyze/{test_prefix}pics'
    json_dir_path = f'set_analyze/{test_prefix}other/json'
    os.makedirs(pic_dir_path, exist_ok=True)
    os.makedirs(json_dir_path, exist_ok=True)

    worksname = use_conf['cache_work_names'] #like mcf

    n_works = len(worksname)
    n_rows = math.ceil(n_works/4)

    waydict_format = 'cache_work_{}ways'
    perf_prefixs = ['90perf','95perf','full']
    draw_picformat_jsonformat = [
        (draw_growlist,'atd_est_growlist_{}.png','atd_est_growlist_{}.json'),
        # (draw_one_workload_pn_blocklen,'pn_est_blocklen_contour_{}.png'),
        # (draw_one_workload_pn_cyclelen,'pn_est_cyclelen_contour_{}.png'),
    ]

    for perf_prefix in perf_prefixs:
        waydict_name = waydict_format.format(perf_prefix)
        waydict = use_conf[waydict_name]
        ret_dict = {}
        for draw_func,pic_name_format,json_name_format in draw_picformat_jsonformat:
            draw_db_by_func(base_dir,n_rows,waydict,
                analyze_func=analyze_workload_len_est,
                draw_one_func=draw_func,
                fig_name=os.path.join(pic_dir_path,pic_name_format.format(perf_prefix)),
                json_path=os.path.join(json_dir_path,json_name_format.format(perf_prefix)),
                # force_update_json=True,
                input_stats_dict=ret_dict)