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

def draw_memsign_bar(ax,s_dicts,workload_name,full_ass,pos:tuple):
    all_signs_view = s_dicts['access_mem_signs'].values()
    s_signs = sorted(all_signs_view,key=lambda x:x[3],reverse=True)

    lenx = len(s_signs)
    xval = np.arange(1,lenx+1,1)
    near_hit_cnts = [x[0] for x in s_signs]
    far_hit_cnts = [x[1] for x in s_signs]
    miss_cnts = [x[2] for x in s_signs]

    barwidth = 0.5
    ax.bar(xval,near_hit_cnts, label='near hit', color = 'tab:blue', width = barwidth)
    ax.bar(xval,far_hit_cnts, label='far hit', color = 'tab:orange', width = barwidth, bottom = near_hit_cnts)
    ax.bar(xval,miss_cnts, label='miss', color = 'tab:red', width = barwidth, bottom = np.array(near_hit_cnts)+np.array(far_hit_cnts))

    ax.set_ylabel('reref counts')
    ax.set_xlabel('mem signatures (sorted by full reref cnt)')
    ax.set_title(f'{workload_name}')
    if pos == (0,0):
        ax.legend(shadow=0, fontsize = 13, bbox_to_anchor=(-0.01,1.4), loc = 'upper left',  \
            borderaxespad=0.2, ncol = 1, columnspacing=0.5, labelspacing=0.1)

def draw_pcsign_bar(ax,s_dicts,workload_name,full_ass,pos:tuple):
    all_signs_view = s_dicts['access_pc_signs'].values()
    s_signs = sorted(all_signs_view,key=lambda x:x[3],reverse=True)

    lenx = len(s_signs)
    xval = np.arange(1,lenx+1,1)
    near_hit_cnts = [x[0] for x in s_signs]
    far_hit_cnts = [x[1] for x in s_signs]
    miss_cnts = [x[2] for x in s_signs]

    barwidth = 0.5
    ax.bar(xval,near_hit_cnts, label='near hit', color = 'tab:blue', width = barwidth)
    ax.bar(xval,far_hit_cnts, label='far hit', color = 'tab:orange', width = barwidth, bottom = near_hit_cnts)
    ax.bar(xval,miss_cnts, label='miss', color = 'tab:red', width = barwidth, bottom = np.array(near_hit_cnts)+np.array(far_hit_cnts))

    ax.set_ylabel('reref counts')
    ax.set_xlabel('pc signatures (sorted by full reref cnt)')
    ax.set_title(f'{workload_name}')
    if pos == (0,0):
        ax.legend(shadow=0, fontsize = 13, bbox_to_anchor=(-0.01,1.4), loc = 'upper left',  \
            borderaxespad=0.2, ncol = 1, columnspacing=0.5, labelspacing=0.1)

def draw_memsign_maynothit_bar(ax,s_dicts,workload_name,full_ass,pos:tuple):
    all_signs_view = s_dicts['access_mem_signs'].values()
    #miss rate over 40%
    notjudge_hit_it = filter(lambda x:x[2]/x[3] > 0.5,all_signs_view)
    s_signs = sorted(notjudge_hit_it,key=lambda x:x[3],reverse=True)

    lenx = len(s_signs)
    xval = np.arange(1,lenx+1,1)
    near_hit_cnts = [x[0] for x in s_signs]
    far_hit_cnts = [x[1] for x in s_signs]
    miss_cnts = [x[2] for x in s_signs]

    barwidth = 0.5
    ax.bar(xval,near_hit_cnts, label='near hit', color = 'tab:blue', width = barwidth)
    ax.bar(xval,far_hit_cnts, label='far hit', color = 'tab:orange', width = barwidth, bottom = near_hit_cnts)
    ax.bar(xval,miss_cnts, label='miss', color = 'tab:red', width = barwidth, bottom = np.array(near_hit_cnts)+np.array(far_hit_cnts))

    ax.set_ylabel('reref counts')
    ax.set_xlabel('mem signatures (sorted by full reref cnt)')
    ax.set_title(f'{workload_name}')
    if pos == (0,0):
        ax.legend(shadow=0, fontsize = 13, bbox_to_anchor=(-0.01,1.4), loc = 'upper left',  \
            borderaxespad=0.2, ncol = 1, columnspacing=0.5, labelspacing=0.1)
def draw_pcsign_maynothit_bar(ax,s_dicts,workload_name,full_ass,pos:tuple):
    all_signs_view = s_dicts['access_pc_signs'].values()
    #miss rate over 40%
    notjudge_hit_it = filter(lambda x:x[2]/x[3] > 0.5,all_signs_view)
    s_signs = sorted(notjudge_hit_it,key=lambda x:x[3],reverse=True)

    lenx = len(s_signs)
    xval = np.arange(1,lenx+1,1)
    near_hit_cnts = [x[0] for x in s_signs]
    far_hit_cnts = [x[1] for x in s_signs]
    miss_cnts = [x[2] for x in s_signs]

    barwidth = 0.5
    ax.bar(xval,near_hit_cnts, label='near hit', color = 'tab:blue', width = barwidth)
    ax.bar(xval,far_hit_cnts, label='far hit', color = 'tab:orange', width = barwidth, bottom = near_hit_cnts)
    ax.bar(xval,miss_cnts, label='miss', color = 'tab:red', width = barwidth, bottom = np.array(near_hit_cnts)+np.array(far_hit_cnts))

    ax.set_ylabel('reref counts')
    ax.set_xlabel('pc signatures (sorted by full reref cnt)')
    ax.set_title(f'{workload_name}')
    if pos == (0,0):
        ax.legend(shadow=0, fontsize = 13, bbox_to_anchor=(-0.01,1.4), loc = 'upper left',  \
            borderaxespad=0.2, ncol = 1, columnspacing=0.5, labelspacing=0.1)

def report_csvsum_memsign_growhit(usepd,s_dicts,workload_name,full_ass):
    hitpos_set_records = s_dicts['hitpos_stamp_record']

    new_dict = {}

    new_dict['workload_name'] = workload_name
    # new_dict['max_ways'] = full_ass
    tmp_pd = pd.DataFrame(new_dict,index=[0])
    return pd.concat([usepd,tmp_pd],ignore_index=True)

def draw_pcsign_growhit_hist(ax,s_dicts,workload_name,full_ass,pos:tuple):
    phit_hist, phit_bin_edges = s_dicts['pc_predict_hit_num_when_growhit']
    phit_hist = np.array(phit_hist)
    phit_bin_edges = np.array(phit_bin_edges)
    pnhit_hist, pnhit_bin_edges = s_dicts['pc_predict_nearhit_num_when_growhit']
    pnhit_hist = np.array(pnhit_hist)
    pnhit_bin_edges = np.array(pnhit_bin_edges)

    xval = np.arange(0,full_ass+1,1)

    bwidth = 0.2
    hist, bin_edges = phit_hist, phit_bin_edges
    newh = hist * np.diff(bin_edges)
    ax.bar( xval[:-1] - bwidth/2 , newh, label='pc predict hit nums', color = contrasting_orange[0], width = bwidth)

    hist, bin_edges = pnhit_hist, pnhit_bin_edges
    newh = hist * np.diff(bin_edges)
    ax.bar( xval[:-1] + bwidth/2 , hist, label='pc predict nearhit nums', color = contrasting_orange[1], width = bwidth)

    ax.set_ylabel('portion of growhits')
    ax.set_xlabel('number of blocks in a set')
    ax.set_title(f'{workload_name}')
    if pos == (0,0):
        ax.legend(shadow=0, fontsize = 13, bbox_to_anchor=(-0.01,1.4), loc = 'upper left',  \
            borderaxespad=0.2, ncol = 1, columnspacing=0.5, labelspacing=0.1)

def draw_memsign_growhit_hist(ax,s_dicts,workload_name,full_ass,pos:tuple):
    phit_hist, phit_bin_edges = s_dicts['mem_predict_hit_num_when_growhit']
    phit_hist = np.array(phit_hist)
    phit_bin_edges = np.array(phit_bin_edges)
    pnhit_hist, pnhit_bin_edges = s_dicts['mem_predict_nearhit_num_when_growhit']
    pnhit_hist = np.array(pnhit_hist)
    pnhit_bin_edges = np.array(pnhit_bin_edges)

    xval = np.arange(0,full_ass+1,1)

    bwidth = 0.2
    hist, bin_edges = phit_hist, phit_bin_edges
    newh = hist * np.diff(bin_edges)
    ax.bar( xval[:-1] - bwidth/2 , newh, label='mem predict hit nums', color = contrasting_orange[3], width = bwidth)

    hist, bin_edges = pnhit_hist, pnhit_bin_edges
    newh = hist * np.diff(bin_edges)
    ax.bar( xval[:-1] + bwidth/2 , hist, label='mem predict nearhit nums', color = contrasting_orange[4], width = bwidth)

    ax.set_ylabel('portion of growhits')
    ax.set_xlabel('number of blocks in a set')
    ax.set_title(f'{workload_name}')
    if pos == (0,0):
        ax.legend(shadow=0, fontsize = 13, bbox_to_anchor=(-0.01,1.4), loc = 'upper left',  \
            borderaxespad=0.2, ncol = 1, columnspacing=0.5, labelspacing=0.1)

class SetLRUStates:
    def __init__(self, set_id:int, full_ass:int, pc_sign_dict:dict, mem_sign_dict:dict):
        self.set_id = set_id
        self.full_ass = full_ass
        #when hit pos >= now_ass, it is a grow reref
        self.now_ass = 1
        self.pc_sign_dict = pc_sign_dict
        self.mem_sign_dict = mem_sign_dict
        self.mru_list = list()
        #associate singature status with tag
        self.tag_sign_bits_dict = dict()
        #record access to judge reref
        self.reached_tags = set()
        #record stats for far hit
        self.pc_predict_hit_diff_ingrow_when_growhit = []
        self.pc_predict_nearhit_diff_ingrow_when_growhit = []
        self.mem_predict_hit_diff_ingrow_when_growhit = []
        self.mem_predict_nearhit_diff_ingrow_when_growhit = []
    
    def build_mem_sign(self, tag:int):
        #build mem signature
        return tag << 6 | self.set_id >> 8

    def check_sign_morehit(self, target_dict, sign):
        if sign in target_dict:
            # 2 * miss < total refcnt
            if target_dict[sign][2] * 2 < target_dict[sign][3]:
                return True
        return False
    def check_sign_morenearhit(self, target_dict, sign):
        if sign in target_dict:
            # 2 * nearhit >= total refcnt
            if target_dict[sign][0] * 2 >= target_dict[sign][3]:
                return True
        return False

    def insert_sign_dict(self, target_dict, sign, value_index):
        if sign in target_dict:
            target_dict[sign][value_index] += 1
            target_dict[sign][3] += 1
        else:
            # near hit, grow hit, miss, total refcnt
            target_dict[sign] = [0,0,0,1]
            target_dict[sign][value_index] = 1


    def record_sign_hit(self, tag, pc, hitpos):
        mem_sign = self.build_mem_sign(tag)
        if hitpos >= self.now_ass:
            #grow hit
            self.insert_sign_dict(self.mem_sign_dict, mem_sign, 1)
            self.insert_sign_dict(self.pc_sign_dict, pc, 1)
            
            #record stats for far hit
            total_pc_hit = 0
            total_pc_nearhit = 0
            total_mem_hit = 0
            total_mem_nearhit = 0
            for i in range(self.now_ass):
                t = self.mru_list[i]
                pc_hit,pc_nearhit,mem_hit,mem_nearhit = self.tag_sign_bits_dict[t]
                total_pc_hit += int(pc_hit)
                total_pc_nearhit += int(pc_nearhit)
                total_mem_hit += int(mem_hit)
                total_mem_nearhit += int(mem_nearhit)
            diff_pc_hit = self.now_ass - total_pc_hit
            diff_pc_nearhit = self.now_ass - total_pc_nearhit
            diff_mem_hit = self.now_ass - total_mem_hit
            diff_mem_nearhit = self.now_ass - total_mem_nearhit
            self.pc_predict_hit_diff_ingrow_when_growhit.append(diff_pc_hit)
            self.pc_predict_nearhit_diff_ingrow_when_growhit.append(diff_pc_nearhit)
            self.mem_predict_hit_diff_ingrow_when_growhit.append(diff_mem_hit)
            self.mem_predict_nearhit_diff_ingrow_when_growhit.append(diff_mem_nearhit)
        else:
            #near hit
            self.insert_sign_dict(self.mem_sign_dict, mem_sign, 0)
            self.insert_sign_dict(self.pc_sign_dict, pc, 0)

    def record_sign_miss(self, tag, pc):
        mem_sign = self.build_mem_sign(tag)
        self.insert_sign_dict(self.mem_sign_dict, mem_sign, 2)
        self.insert_sign_dict(self.pc_sign_dict, pc, 2)

    def find_sign_bits(self, tag, pc):
        pc_sign_predict_hit = self.check_sign_morehit(self.pc_sign_dict, pc)
        pc_sign_predict_nearhit = self.check_sign_morenearhit(self.pc_sign_dict, pc)
        mem_sign = self.build_mem_sign(tag)
        mem_sign_predict_hit = self.check_sign_morehit(self.mem_sign_dict, mem_sign)
        mem_sign_predict_nearhit = self.check_sign_morenearhit(self.mem_sign_dict, mem_sign)
        return (pc_sign_predict_hit, pc_sign_predict_nearhit, mem_sign_predict_hit, mem_sign_predict_nearhit)

    def newcome(self, tag, pc):
        ls = self.mru_list
        if tag in ls:
            #hit
            #record hit pos
            hit_pos = ls.index(tag)
            #record hit pos stamp
            self.record_sign_hit(tag, pc, hit_pos)
            #grow hit if hit pos far
            if hit_pos >= self.now_ass:
                self.now_ass += 1
            #modify lru
            ls.remove(tag)
            ls.insert(0,tag)
        else:
            #miss
            if len(ls) >= self.full_ass:
                evict_tag = ls.pop()
                self.tag_sign_bits_dict.pop(evict_tag)
            ls.insert(0,tag)
            '''
            if tag in self.reached_tags:
                #reref, record miss
                self.record_sign_miss(tag, pc)
            else:
                #first time
                self.reached_tags.add(tag)
            '''
            self.record_sign_miss(tag, pc)
        #update tag sign bits
        self.tag_sign_bits_dict[tag] = self.find_sign_bits(tag, pc)

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
        all_access_query = 'SELECT REQID,SETIDX,TAG FROM HitMissTrace ORDER BY ID;'
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        f = cur.execute(all_access_query)

        access_pc_signs = {}
        access_mem_signs = {}
        reref_states = [SetLRUStates(se,full_ass,access_pc_signs,access_mem_signs) for se in range(all_set)]
        for reqid,idx,tag in f:
            pc = int(reqid)
            idx = int(idx)
            tag = int(tag)
            reref_states[idx].newcome(pc,tag)

        cur.close()

    s_dicts['access_pc_signs'] = access_pc_signs
    s_dicts['access_mem_signs'] = access_mem_signs

    xval = np.arange(0,full_ass+1,1)

    fit = functools.reduce(operator.concat, map(lambda x:x.pc_predict_hit_diff_ingrow_when_growhit, reref_states), [])
    s_dicts['pc_predict_hit_num_when_growhit'] = [a.tolist() for a in np.histogram(list(fit),bins=xval,density=True)]
    fit = functools.reduce(operator.concat, map(lambda x:x.pc_predict_nearhit_diff_ingrow_when_growhit, reref_states), [])
    s_dicts['pc_predict_nearhit_num_when_growhit'] = [a.tolist() for a in np.histogram(list(fit),bins=xval,density=True)]
    fit = functools.reduce(operator.concat, map(lambda x:x.mem_predict_hit_diff_ingrow_when_growhit, reref_states), [])
    s_dicts['mem_predict_hit_num_when_growhit'] = [a.tolist() for a in np.histogram(list(fit),bins=xval,density=True)]
    fit = functools.reduce(operator.concat, map(lambda x:x.mem_predict_nearhit_diff_ingrow_when_growhit, reref_states), [])
    s_dicts['mem_predict_nearhit_num_when_growhit'] = [a.tolist() for a in np.histogram(list(fit),bins=xval,density=True)]

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
        # (draw_memsign_bar,'sign_mem_reref_bar_{}.png','sign_reref_grow_{}.json',None,None),
        # (draw_pcsign_bar,'sign_pc_reref_bar_{}.png','sign_reref_grow_{}.json',None,None),
        (draw_memsign_bar,'sign_mem_reref_bar_{}.png','sign_reref_grow_{}.json',None,None),
        (draw_pcsign_bar,'sign_pc_reref_bar_{}.png','sign_reref_grow_{}.json',None,None),
        (draw_memsign_growhit_hist,'sign_mem_growhit_hist_{}.png','sign_reref_grow_{}.json',None,None),
        (draw_pcsign_growhit_hist,'sign_pc_growhit_hist_{}.png','sign_reref_grow_{}.json',None,None),
        (draw_memsign_maynothit_bar,'sign_mem_maynothit_bar_{}.png','sign_reref_grow_{}.json',None,None),
        (draw_pcsign_maynothit_bar,'sign_pc_maynothit_bar_{}.png','sign_reref_grow_{}.json',None,None),
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
                force_update_json=True,
                input_stats_dict=ret_dict)