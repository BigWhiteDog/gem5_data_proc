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
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib import ticker
from matplotlib.patches import Patch
import sqlite3

import itertools
import functools
import operator

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

def draw_lru_farhit_stampsdist(ax,s_dicts,workload_name,full_ass,pos:tuple):
    rt = s_dicts['farhit_reusetime_list']
    rt_array = np.array(rt)
    rtm_array = rt_array/1_000_000
    max_time = s_dicts['delta_stamp_last']
    maxM_time = max_time/1_000_000

    ax.hist(rtm_array.tolist(), bins = 'auto', label='farthest reref Mcycle',histtype = 'bar', 
            color =  contrasting_orange[0],  linewidth=2)

    # ax.set_xlim(0,maxM_time)

    # hitlen_hist,hitlen_edges = np.histogram(rt, bins = 'auto')
    # sum_density = np.sum(hitlen_hist)
    # hitlen_hist = hitlen_hist/sum_density
    # ax.hist(hit)
    # ax.hist(hitlen_list, bins = 'auto', label='hit len',histtype = 'step', 
    #         density=True, cumulative=True, color = hitlen_list_color,  linewidth=2)


    ax.set_ylabel('farthest reuse counts')
    ax.set_xlabel('reuse cycle (Mcycle)')
    ax.set_title(f'{workload_name}')
    if pos == (0,0):
        ax.legend(shadow=0, fontsize = 13, bbox_to_anchor=(-0.01,1.4), loc = 'upper left',  \
            borderaxespad=0.2, ncol = 1, columnspacing=0.5, labelspacing=0.1)

def draw_lru_firstfarhit_stampsdist(ax,s_dicts,workload_name,full_ass,pos:tuple):
    ffht = s_dicts['first_farhit_stamp_list']
    ffht_array = np.array(ffht)
    ffhtm_array = ffht_array/1_000_000
    max_time = s_dicts['delta_stamp_last']
    maxM_time = max_time/1_000_000

    # ax.hist(ffhtm_array.tolist(), bins = 'auto', label='first far farhit happen',histtype = 'bar', 
    #         density=True, cumulative=True,
    #         color =  contrasting_orange[1],  linewidth=2)

    ax.set_xlim(0,maxM_time)

    hitlen_hist,hitlen_edges = np.histogram(ffhtm_array.tolist(), bins = 'auto')
    # sum_density = np.sum(hitlen_hist)
    hitlen_hist = hitlen_hist/all_set
    cum_hitlen_hist = np.cumsum(hitlen_hist)
    ax.stairs(cum_hitlen_hist, hitlen_edges, label='farthest reuse happened sets',
        color =  contrasting_orange[1],  linewidth=2 , fill=True)
    
    ax.set_ylim(0,1)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

    ax.set_ylabel('portion of sets with farthest reuse')
    ax.set_xlabel('timeline (Mcycle)')
    ax.set_title(f'{workload_name}')
    if pos == (0,0):
        ax.legend(shadow=0, fontsize = 13, bbox_to_anchor=(-0.01,1.4), loc = 'upper left',  \
            borderaxespad=0.2, ncol = 1, columnspacing=0.5, labelspacing=0.1)



class SetLRUStates:
    def __init__(self, set_id:int, full_ass:int, target_ass:int):
        self.set_id = set_id
        self.full_ass = full_ass
        #when hit pos >= target_ass, it is a tail hit
        self.target_ass = target_ass
        self.mru_list = list()
        self.mru_last_stamps = list()
        #record stats for far hit
        self.farhit_delta_stamp_list = []
        self.first_farhit_stamp = None

    def record_sign_hit(self, hit_pos,stamp):
        lstamp = self.mru_last_stamps
        if hit_pos >= self.target_ass:
            #far hit
            last_stamp = lstamp[hit_pos]
            delta_stamp = stamp - last_stamp
            self.farhit_delta_stamp_list.append(delta_stamp)
            if self.first_farhit_stamp is None:
                self.first_farhit_stamp = stamp

    def newcome(self, tag, stamp):
        ls = self.mru_list
        lstamp = self.mru_last_stamps
        if tag in ls:
            #hit
            #record hit pos
            hit_pos = ls.index(tag)
            #record hit pos stamp
            self.record_sign_hit(hit_pos,stamp)
            #modify lru
            ls.pop(hit_pos)
            ls.insert(0,tag)
            #modify lru stamp
            lstamp.pop(hit_pos)
            lstamp.insert(0,stamp)
        else:
            #miss
            if len(ls) >= self.full_ass:
                evict_tag = ls.pop()
                evict_stamp = lstamp.pop()
            ls.insert(0,tag)
            lstamp.insert(0,stamp)

def analyze_workload_reuse(work_stats_dict,work,work_dir,full_ass):
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
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        stamp0_query = 'SELECT min(STAMP),max(STAMP)-min(STAMP) from HitMissTrace;'
        f = cur.execute(stamp0_query)
        stamp0,delta_stamp_last = f.fetchone()

        all_access_query = 'SELECT SETIDX,TAG,STAMP FROM HitMissTrace ORDER BY ID;'
        f = cur.execute(all_access_query)

        lru_states = [SetLRUStates(se,full_ass,full_ass-1) for se in range(all_set)]
        for idx,tag,stamp in f:
            idx = int(idx)
            tag = int(tag)
            stamp = int(stamp) - stamp0
            lru_states[idx].newcome(tag,stamp)

        cur.close()

    s_dicts['delta_stamp_last'] = delta_stamp_last

    s_dicts['farhit_reusetime_list'] = []
    s_dicts['first_farhit_stamp_list'] = []
    for reref_state in lru_states:
        s_dicts['farhit_reusetime_list'].extend(reref_state.farhit_delta_stamp_list)
        if reref_state.first_farhit_stamp is not None:
            s_dicts['first_farhit_stamp_list'].append(reref_state.first_farhit_stamp)

    work_stats_dict[work] = s_dicts

def draw_db_by_func(base_dir,n_rows,worksname_waydict,
    analyze_func,draw_one_func,
    fig_name,
    csv_one_func=None,
    csv_summary_path=None,
    input_stats_dict=None,
    json_path=None,
    force_update_json = False):
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
                
    mypd = pd.DataFrame()

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
        if csv_one_func is not None:
            mypd = csv_one_func(mypd,s_dicts,work,full_ass)
        draw_one_func(ax_bar,s_dicts,work,full_ass,(fx,fy))     

    for i in range(len(worksname_waydict),n_rows*4):
        fx = i // 4
        fy = i % 4
        ax[fx,fy].remove()

    plt.tight_layout()
    if 'nothing' not in fig_name:
        plt.savefig(fig_name,dpi=300)
    plt.clf()

    if not dict_updated or force_update_json:
        #save to json
        if json_path is not None:
            with open(json_path,'w') as f:
                json.dump(work_stats_dict,f,indent=2)
    
    if csv_summary_path is not None:
        if len(mypd) > 0:
            mypd.style.format(precision=5)
            print(mypd)
            mypd.to_csv(csv_summary_path,index=False,float_format='%.5f')

    return work_stats_dict


if __name__ == '__main__':
    use_conf = conf_50M
    test_prefix = use_conf['test_prefix']
    base_dir = base_dir_format.format(test_prefix)
    pic_dir_path = f'set_analyze/{test_prefix}pics'
    json_dir_path = f'set_analyze/{test_prefix}other/json'
    csv_summary_dir_path = f'set_analyze/{test_prefix}other/csv_summary'
    os.makedirs(pic_dir_path, exist_ok=True)
    os.makedirs(json_dir_path, exist_ok=True)
    os.makedirs(csv_summary_dir_path, exist_ok=True)

    worksname = use_conf['cache_work_names'] #like mcf

    n_works = len(worksname)
    n_rows = math.ceil(n_works/4)

    waydict_format = 'cache_work_{}ways'
    perf_prefixs = ['90perf','95perf','full']
    drawF_picf_jsonf_csvF_csvsumf = [
        # (draw_memsign_bar,'sign_mem_reref_bar_{}.png','sign_reref_{}.json',None,None),
        # (draw_pcsign_bar,'sign_pc_reref_bar_{}.png','sign_reref_{}.json',None,None),
        (draw_lru_farhit_stampsdist,'lru_farhit_stampsdist_{}.png','lrufarhit_reref_stamp_{}.json',None,None),
        (draw_lru_firstfarhit_stampsdist,'lru_firstfarhit_stampsdist_{}.png','lrufarhit_reref_stamp_{}.json',None,None),
    ]

    for perf_prefix in perf_prefixs:
        waydict_name = waydict_format.format(perf_prefix)
        waydict = use_conf[waydict_name]
        ret_dict = {}
        for draw_func,pic_name_format,json_name_format,csv_func,csv_name_format in drawF_picf_jsonf_csvF_csvsumf:
            if json_name_format is None:
                this_json_path = None
            else:
                this_json_path = os.path.join(json_dir_path,json_name_format.format(perf_prefix))
            if csv_name_format is None:
                this_csv_summary_path = None
            else:
                this_csv_summary_path = os.path.join(csv_summary_dir_path,csv_name_format.format(perf_prefix))
            draw_db_by_func(base_dir,n_rows,waydict,
                analyze_func=analyze_workload_reuse,
                draw_one_func=draw_func,
                csv_one_func=csv_func,
                fig_name=os.path.join(pic_dir_path,pic_name_format.format(perf_prefix)),
                json_path=this_json_path,
                csv_summary_path=this_csv_summary_path,
                force_update_json=False,
                input_stats_dict=ret_dict)