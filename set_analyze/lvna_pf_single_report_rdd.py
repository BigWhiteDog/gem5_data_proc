import os
import shutil
import numpy as np
import utils.common as c
from utils.common import extract_newgem_raw_json
import utils.target_stats as t
import csv
import numpy as np
import argparse

import json
import numpy as np
import pandas as pd
import re
import sqlite3

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib import ticker

from set_analyze.my_diff_color import *

parser = argparse.ArgumentParser(description="options to get set stats")
# parser.add_argument('-d','--stats_dir', type=str,
#     help='stats dir to analyze',required=True)
# parser.add_argument('--ids',default=16,type=int)
# parser.add_argument('--nsamples',default=2,type=int)
# parser.add_argument('--l3_sets',default=4096,type=int)
parser.add_argument('-n','--ncore',type=int,default=1)
parser.add_argument('-j','--json', type=str,
    default=None)
parser.add_argument('--bench-choice',
		    choices=['short','long'],
			default='long')

opt = parser.parse_args()

confs=[
    "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_lvnapf_50M.json",
]

skip_res = [
    re.compile(r'PrefetchSingleCorePolicy'),
    re.compile(r'PageUCPPolicy'),
]
interested_res = [
    # re.compile(r'IpcSample'),
    # re.compile(r'UCP'),
    # re.compile(r'FairPrefUCPPolicy-2M'),
]
xlsx_drop_res = [
    re.compile(r'nopart'),
]

def analyze_one_workload_dict(base_dir,workload_names,ncore=2):
    s_dicts = {'workload':workload_names}
    
    new_base = os.path.join(base_dir,'l3-nopart')
        
    db_path = os.path.join(new_base,'hm.db')
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # all_demand_query = 'SELECT REQID,TAG,SETIDX,HITMISSINS FROM L3XsTrace WHERE HITMISSINS != 3 ORDER BY ID ASC;'
    all_demand_query = 'SELECT REQID,TAG,SETIDX,HITMISSINS FROM L3XsTrace ORDER BY ID ASC;'
    f = cur.execute(all_demand_query)
    set_lru_state = [list() for _ in range(all_set)]
    set_tag_mapid = [dict() for _ in range(all_set)]
    hitcnt = 0 #hit count
    stat_pfnone_hitcnt = 0
    stat_pf_hitcnt = 0
    misscnt = 0
    stat_pfnone_misscnt = 0
    stat_pf_misscnt = 0

    d2d_cnt = 0 #normal demand
    d2p_cnt = 0 #prefetch after demand
    p2d_cnt = 0 #useful prefetch
    p2p_cnt = 0 #prefetch reuse
    cross_ref_cnt = 0 #cross reference
    pref_cross_ref_cnt = 0 #prefetch cross reference
    id_hitcnt = 0 #hit cnt with id
    for reqid,tag,setidx,hitmissins in f:
        reqid = int(reqid)
        setidx = int(setidx)
        tag = int(tag)
        hitmissins = int(hitmissins)
        map_id = set_tag_mapid[setidx]
        if hitmissins == 2:
            #insert
            map_id[tag] = reqid
        elif hitmissins == 0:
            #hit
            hitcnt += 1
            if reqid == 0:
                stat_pfnone_hitcnt += 1
            else:
                stat_pf_hitcnt += 1
            if tag not in map_id:
                continue
            id_hitcnt += 1
            old_id = map_id[tag]
            if old_id == 0:
                if reqid == 0:
                    d2d_cnt += 1
                else:
                    d2p_cnt += 1
                    cross_ref_cnt += 1
            else:
                if reqid == 0:
                    p2d_cnt += 1
                    cross_ref_cnt += 1
                else:
                    p2p_cnt += 1
                    if old_id != reqid:
                        pref_cross_ref_cnt += 1
                        cross_ref_cnt += 1
        elif hitmissins == 1:
            #miss
            misscnt += 1
            if reqid == 0:
                stat_pfnone_misscnt += 1
            else:
                stat_pf_misscnt += 1

    s_dicts['hitcnt'] = hitcnt
    s_dicts['stat_pfnone_hitcnt'] = stat_pfnone_hitcnt
    s_dicts['stat_pf_hitcnt'] = stat_pf_hitcnt
    s_dicts['d2d_cnt'] = d2d_cnt
    s_dicts['d2p_cnt'] = d2p_cnt
    s_dicts['p2d_cnt'] = p2d_cnt
    s_dicts['p2p_cnt'] = p2p_cnt
    s_dicts['cross_ref_cnt'] = cross_ref_cnt
    s_dicts['pref_cross_ref_cnt'] = pref_cross_ref_cnt
    s_dicts['id_hitcnt'] = id_hitcnt

    s_dicts['misscnt'] = misscnt
    s_dicts['stat_pfnone_misscnt'] = stat_pfnone_misscnt
    s_dicts['stat_pf_misscnt'] = stat_pf_misscnt

    s_dicts['total_cnt'] = hitcnt + misscnt
    s_dicts['total_pf_cnt'] = stat_pf_hitcnt + stat_pf_misscnt
    s_dicts['total_pfnone_cnt'] = stat_pfnone_hitcnt + stat_pfnone_misscnt

    s_dicts['hit_rate'] = hitcnt / (hitcnt + misscnt)

    s_dicts['pf_rate'] = s_dicts['total_pf_cnt'] / s_dicts['total_cnt']

    s_dicts['pf_hit_intotal_rate'] = s_dicts['stat_pf_hitcnt'] / s_dicts['total_cnt']
    if s_dicts['total_pf_cnt'] == 0:
        s_dicts['pf_hit_rate'] = 0
    else:
        s_dicts['pf_hit_rate'] = s_dicts['stat_pf_hitcnt'] / s_dicts['total_pf_cnt']
    
    s_dicts['demand_hit_intotal_rate'] = s_dicts['stat_pfnone_hitcnt'] / s_dicts['total_cnt']
    if s_dicts['total_pfnone_cnt'] == 0:
        s_dicts['demand_hit_rate'] = 0
    else:
        s_dicts['demand_hit_rate'] = s_dicts['stat_pfnone_hitcnt'] / s_dicts['total_pfnone_cnt']
    
    if (id_hitcnt == 0):
        s_dicts['d2d_ratio'] = 0
        s_dicts['d2p_ratio'] = 0
        s_dicts['p2d_ratio'] = 0
        s_dicts['p2p_ratio'] = 0
        s_dicts['cross_ref_ratio'] = 0
        s_dicts['pref_cross_ref_ratio'] = 0
    else:
        s_dicts['d2d_ratio'] = d2d_cnt / id_hitcnt
        s_dicts['d2p_ratio'] = d2p_cnt / id_hitcnt
        s_dicts['p2d_ratio'] = p2d_cnt / id_hitcnt
        s_dicts['p2p_ratio'] = p2p_cnt / id_hitcnt
        s_dicts['cross_ref_ratio'] = cross_ref_cnt / id_hitcnt
        s_dicts['pref_cross_ref_ratio'] = pref_cross_ref_cnt / id_hitcnt

    
    return s_dicts


def run_one_conf(select_json:str):
    with open(select_json,'r') as f:
        global use_conf
        use_conf = json.load(f)
    if use_conf is None:
        exit(255)

    global test_prefix
    test_prefix = use_conf['test_prefix']
    global max_assoc
    max_assoc = use_conf['max_assoc']
    global all_set
    all_set = use_conf['all_set']

    base_dir_format = use_conf['base_dir_format']
    test_prefix = use_conf['test_prefix']
    base_dir = base_dir_format.format(test_prefix)

    excel_dir_path = f'set_analyze/{test_prefix}xls'
    os.makedirs(excel_dir_path, exist_ok=True)

    sub_excel_dir_path = os.path.join(excel_dir_path,'nopart')
    os.makedirs(sub_excel_dir_path, exist_ok=True)

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    twriter = pd.ExcelWriter(os.path.join(sub_excel_dir_path, f'id-reuse.xlsx'), engine='xlsxwriter')


    worksname = os.listdir(base_dir) #like omnetpp-xalancbmk
    pd_dict_list = []
    for i,work in enumerate(worksname):
        word_dir = os.path.join(base_dir,work)
        if not os.path.isdir(word_dir):
            continue
        pd_dict = analyze_one_workload_dict(word_dir,work)
        pd_dict['workload'] = work
        pd_dict_list.append(pd_dict)
 
    df = pd.DataFrame(pd_dict_list)
    df.to_excel(twriter, sheet_name='id-reuse', index=False)
    twriter.close()

    # with open(select_json,'w') as f:
    #     json.dump(use_conf,f,indent=4)

if __name__ == '__main__':
    # base_dir = '/nfs/home/zhangchuanqi/lvna/for_xs/catlog/single-profiling/'
    if opt.json:
        select_json = opt.json
        run_one_conf(select_json)
    else:
        for co in confs:
            select_json = co
            run_one_conf(select_json)