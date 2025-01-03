import copy
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
    s_dicts = {}
    partsname = os.listdir(base_dir) #like l3-1,l3-nopoart,l3-1-csv
    
    interval_num = 0
    
    for part in partsname:
        if not os.path.isdir(os.path.join(base_dir,part)):
            continue
        ways = part.split('-')
        if ways[0] != 'l3':
            continue
        if ways[1].isnumeric(): # l3-2 or l3-2-pf
            continue #skip
        new_base = os.path.join(base_dir,part)
        # ways[-1] as key
        dkey = re.split('-',part, maxsplit=1)[1]
        
        jump_flag = True
        for res in interested_res:
            if res.search(dkey):
                jump_flag = False
            
        if jump_flag:
            continue

        db_path = os.path.join(new_base,'hm.db')
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        max_interval_query = 'SELECT MAX(INTERVAL) FROM UCPLookahead;'
        f_max = cur.execute(max_interval_query)
        
        for mval in f_max:
            interval_num = int(mval[0]) + 1
        cur.close()

        n_hid = 13
        global_ucp_decision_array = np.zeros((interval_num,n_hid),dtype=int)
        global_hitcnt_array = np.zeros((interval_num,n_hid,max_assoc),dtype=int)
        global_hitcnt_diff_array = np.zeros((interval_num,n_hid,max_assoc),dtype=int)
        
        global_insertion_array = np.zeros((interval_num,n_hid),dtype=int)
        
        for tti in range(interval_num):
            # record global
            cur = conn.cursor()
            global_hitcnt_query = f'SELECT HITCNTS,ALLOCATIONS FROM UCPLookahead WHERE INTERVAL = {tti} AND SETIDX = {all_set};'
            f = cur.execute(global_hitcnt_query)
            for idhitcnts, allocs in f:
                #record hitcnts of each cpu
                idhitcnts = idhitcnts.strip().split(' ')
                for cpu,hitcnts in enumerate(idhitcnts):
                    hitcnts = hitcnts.strip().split('-')
                    hitcnts = [int(x) for x in hitcnts]
                    global_hitcnt_array[tti,cpu,:] = hitcnts
                    if (tti == 0):
                        global_hitcnt_diff_array[tti,cpu,:] = hitcnts
                    else:
                        global_hitcnt_diff_array[tti,cpu,:] = hitcnts - np.right_shift(global_hitcnt_array[tti-1,cpu,:],1)
                #record allocation decision
                allocs = allocs.strip().split(' ')
                allocs = [int(x) for x in allocs]
                global_ucp_decision_array[tti,:] = allocs

            cur.close()

            cur = conn.cursor()
            global_insertion_query = f'SELECT INSERTIONS FROM L3InsertTrace WHERE INTERVAL = {tti};'
            f = cur.execute(global_insertion_query)
            for ins in f:
                ins = ins[0].strip().split(' ')
                ins = [int(x) for x in ins]
                global_insertion_array[tti,:] = ins

        conn.close()

    hitcnt_dict_list = []
    decision_dict_list = []
    insertion_dict_list = []
    for intveral in range(interval_num):
        hitcnt_dict = {}
        for i in range(n_hid):
            for j in range(max_assoc):
                hitcnt_dict[f'hitcnt_diff_{i}_{j}'] = global_hitcnt_diff_array[intveral,i,j]
        hitcnt_dict_list.append(hitcnt_dict)

        decision_dict = {}
        for i in range(n_hid):
            decision_dict[f'ucp_decision_{i}'] = global_ucp_decision_array[intveral,i]
        decision_dict_list.append(decision_dict)

        insertion_dict = {}
        for i in range(n_hid):
            insertion_dict[f'insertion_{i}'] = global_insertion_array[intveral,i]
        insertion_dict_list.append(insertion_dict)

    all_pd_dict = {}
    all_pd_dict['hitcnt'] = hitcnt_dict_list
    all_pd_dict['decision'] = decision_dict_list
    all_pd_dict['insertion'] = insertion_dict_list

    return all_pd_dict


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

    int_len = '2M'
    interested_res.append(re.compile(r'FairPrefUCPPolicy-2M'))
    sub_excel_dir_path = os.path.join(excel_dir_path,'FairPrefUCPPolicy-2M')
    os.makedirs(sub_excel_dir_path, exist_ok=True)

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    twriter = pd.ExcelWriter(os.path.join(sub_excel_dir_path, f'atd-{int_len}.xlsx'), engine='xlsxwriter')


    worksname = os.listdir(base_dir) #like omnetpp-xalancbmk
    for i,work in enumerate(worksname):
        word_dir = os.path.join(base_dir,work)
        if not os.path.isdir(word_dir):
            continue
        pd_dict = analyze_one_workload_dict(word_dir,work)
        df = pd.DataFrame(pd_dict['hitcnt'])
        df.to_excel(twriter, sheet_name=f'{work}_hitcnt',index=True)
        df = pd.DataFrame(pd_dict['decision'])
        df.to_excel(twriter, sheet_name=f'{work}_decision',index=True)
        df = pd.DataFrame(pd_dict['insertion'])
        df.to_excel(twriter, sheet_name=f'{work}_insertion',index=True)
    
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