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

def draw_reref_cycle_hist(ax,s_dicts,workload_name,full_ass,pos:tuple):
    all_s_it = itertools.chain.from_iterable(s_dicts['valid_reref_delta_stamp'])
    ax.hist(list(all_s_it), bins = 'auto', label='cycle len',histtype = 'bar', 
            density=True, color = contrasting_orange[0],  linewidth=2)

    ax.set_ylabel('portion of reref')
    # ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
    # ax.set_ylim(0,0.1)
    ax.set_xlim(0,25_000_000)
    ax.set_xlabel('delta cycles')
    ax.set_title(f'{workload_name}')
    if pos == (0,0):
        ax.legend(shadow=0, fontsize = 13, bbox_to_anchor=(-0.01,1.4), loc = 'upper left',  \
            borderaxespad=0.2, ncol = 1, columnspacing=0.5, labelspacing=0.1)
def draw_reref_cycle_hist_cdf(ax,s_dicts,workload_name,full_ass,pos:tuple):
    all_s_it = itertools.chain.from_iterable(s_dicts['valid_reref_delta_stamp'])
    ax.hist(list(all_s_it), bins = 'auto', label='cycle len',histtype = 'bar', 
            density=True, cumulative=True,color = contrasting_orange[0],  linewidth=2)

    ax.set_ylabel('portion of reref')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.set_xlim(0,25_000_000)
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
    # ax.set_ylim(0,0.1)
    ax.set_xlabel('delta cycles')
    ax.set_title(f'{workload_name}')
    if pos == (0,0):
        ax.legend(shadow=0, fontsize = 13, bbox_to_anchor=(-0.01,1.4), loc = 'upper left',  \
            borderaxespad=0.2, ncol = 1, columnspacing=0.5, labelspacing=0.1)

def draw_reref_access_hist(ax,s_dicts,workload_name,full_ass,pos:tuple):
    all_s_it = itertools.chain.from_iterable(s_dicts['valid_reref_delta_access'])
    ax.hist(list(all_s_it), bins = 'auto', label='cycle len',histtype = 'bar', 
            density=True, color = contrasting_orange[1],  linewidth=2)

    ax.set_ylabel('portion of reref')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
    # ax.set_ylim(0,0.1)
    ax.set_xlabel('delta access')
    ax.set_title(f'{workload_name}')
    if pos == (0,0):
        ax.legend(shadow=0, fontsize = 13, bbox_to_anchor=(-0.01,1.4), loc = 'upper left',  \
            borderaxespad=0.2, ncol = 1, columnspacing=0.5, labelspacing=0.1)
def draw_reref_access_hist_cdf(ax,s_dicts,workload_name,full_ass,pos:tuple):
    all_s_it = itertools.chain.from_iterable(s_dicts['valid_reref_delta_access'])
    ax.hist(list(all_s_it), bins = 'auto', label='cycle len',histtype = 'bar', 
            density=True, cumulative=True, color = contrasting_orange[1],  linewidth=2)

    ax.set_ylabel('portion of reref')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
    # ax.set_ylim(0,0.1)
    ax.set_xlabel('delta access')
    ax.set_title(f'{workload_name}')
    if pos == (0,0):
        ax.legend(shadow=0, fontsize = 13, bbox_to_anchor=(-0.01,1.4), loc = 'upper left',  \
            borderaxespad=0.2, ncol = 1, columnspacing=0.5, labelspacing=0.1)

class SetValidRerefStates:
    #record reref distance only when the last time is hit
    def __init__(self, set_id, full_ass):
        self.set_id = set_id
        self.full_ass = full_ass
        #stamp,cnt,ismiss
        self.ongoing_tags = {}
        self.access_cnts = 0

        self.valid_reref_deltastamp = []
        self.valid_reref_deltaaccess = []

    def newcome(self, tag, stamp, ismiss):
        self.access_cnts += 1
        if ismiss:
            #miss, just update tag status
            self.ongoing_tags[tag] = [stamp, self.access_cnts, ismiss]
        else:
            #hit, check if tag is ongoing
            if tag in self.ongoing_tags:
                #ongoing tag, record reref distance
                last_stamp, last_access, last_ismiss = self.ongoing_tags[tag]
                self.valid_reref_deltastamp.append(stamp - last_stamp)
                self.valid_reref_deltaaccess.append(self.access_cnts - last_access)
            #update ongoing tag
            self.ongoing_tags[tag] = [stamp, self.access_cnts, ismiss]


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
        all_access_query = 'SELECT SETIDX,TAG,STAMP,ISMISS FROM HitMissTrace;'
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        f = cur.execute(all_access_query)

        reref_states = [SetValidRerefStates(se,full_ass) for se in range(all_set)]
        stamp0 = 0
        for idx,tag,stamp,ismiss in f:
            idx = int(idx)
            tag = int(tag)
            stamp = int(stamp)
            ismiss = bool(int(ismiss))
            if stamp0 == 0:
                stamp0 = stamp
            delta_stamp = stamp - stamp0
            reref_states[idx].newcome(tag,delta_stamp,ismiss)

        cur.close()
    s_dicts['last_delta_stamp'] = delta_stamp
    s_dicts['valid_reref_delta_stamp'] = [r.valid_reref_deltastamp for r in reref_states]
    s_dicts['valid_reref_delta_access'] = [r.valid_reref_deltaaccess for r in reref_states]

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
        (draw_reref_cycle_hist,'valid_reref_cycle_hist_{}.png','valid_reref_{}.json',None,None),
        (draw_reref_cycle_hist_cdf,'valid_reref_cycle_hist_cdf_{}.png','valid_reref_{}.json',None,None),
        (draw_reref_access_hist,'valid_reref_access_hist_{}.png','valid_reref_{}.json',None,None),
        (draw_reref_access_hist_cdf,'valid_reref_access_hist_cdf_{}.png','valid_reref_{}.json',None,None),
        # (draw_one_workload_pn_blocklen,'pn_est_blocklen_contour_{}.png'),
        # (draw_one_workload_pn_cyclelen,'pn_est_cyclelen_contour_{}.png'),
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
                analyze_func=analyze_workload_len_est,
                draw_one_func=draw_func,
                csv_one_func=csv_func,
                fig_name=os.path.join(pic_dir_path,pic_name_format.format(perf_prefix)),
                json_path=this_json_path,
                csv_summary_path=this_csv_summary_path,
                force_update_json=False,
                input_stats_dict=ret_dict)