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

class SaturatedInteger:
    def __init__(self, val, lo, hi):
        self.real, self.lo, self.hi = val, lo, hi

    def __add__(self, other):
        return min(self.real + other.real, self.hi)

    def __sub__(self, other):
        return max(self.real - other.real, self.lo)
    
    def __iadd__(self, other):
        self.real = min(self.real + other.real, self.hi)
        return self

    def __isub__(self, other):
        self.real = max(self.real - other.real, self.lo)
        return self

    def isSaturated(self):
        return self.real == self.hi
    def isLowSaturated(self):
        return self.real == self.lo
    
    def half(self):
        self.real = self.real//2

    def isHighHalf(self):
        return self.real > (self.hi + self.lo)/2
    def idLowHalf(self):
        return self.real <= (self.hi + self.lo)/2

    def isHighPortion(self,portion):
        return self.real > self.hi*portion + self.lo*(1-portion)
    def isLowPortion(self,portion):
        return self.real <= self.hi*portion + self.lo*(1-portion)

    # def __get__(self):
    #     return self.real
    def __set__(self, val):
        if val > self.hi:
            self.real = self.hi
        elif val < self.lo:
            self.real = self.lo
        else:
            self.real = val

def analyze_one_workload_dict(base_dir,workload_names):
    s_dicts = {'workload':workload_names}
    
    new_base = os.path.join(base_dir,'l3-nopart')
        
    db_path = os.path.join(new_base,'hm.db')
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    all_demand_query = 'SELECT REQID,TAG,SETIDX,HITMISSINS,PREFAGE FROM L3XsTrace ORDER BY ID ASC;'
    f = cur.execute(all_demand_query)
    # set_tag_mapid = [dict() for _ in range(all_set)]
    tag_pfage_map = {}
    for reqid,tag,setidx,hitmissins,pfage in f:
        reqid = int(reqid)
        setidx = int(setidx)
        tag = int(tag)
        tag = tag & 0xFFFF
        hitmissins = int(hitmissins)
        pfage = int(pfage)
        if hitmissins == 2:
            if tag not in tag_pfage_map:
                tag_pfage_map[tag] = [pfage]
            else:
                tag_pfage_map[tag].append(pfage)
        # map_id = set_tag_mapid[setidx]
        # if hitmissins == 2:
        #     #insert
        #     map_id[tag] = reqid
        # elif hitmissins == 0:
        #     #hit
        #     hitcnt += 1
        #     if reqid == 0:
        #         stat_pfnone_hitcnt += 1
        #     else:
        #         stat_pf_hitcnt += 1
        #     if tag not in map_id:
        #         continue
        #     id_hitcnt += 1
        #     old_id = map_id[tag]
        #     if old_id == 0:
        #         if reqid == 0:
        #             d2d_cnt += 1
        #         else:
        #             d2p_cnt += 1
        #             cross_ref_cnt += 1
        #     else:
        #         if reqid == 0:
        #             p2d_cnt += 1
        #             cross_ref_cnt += 1
        #         else:
        #             p2p_cnt += 1
        #             if old_id != reqid:
        #                 pref_cross_ref_cnt += 1
        #                 cross_ref_cnt += 1
    
    intervals = [250,500,1000,2000]
    sc_maxs = [3,7,15]
    for sc_max in sc_maxs:
        for interval in intervals:
            total_cnt = 0
            total_sat = 0
            correct_sat = 0
            for t,pfages in tag_pfage_map.items():
                tag_sc = SaturatedInteger(0,0,sc_max)
                for pfage in pfages:
                    total_cnt += 1
                    if tag_sc.isSaturated():
                        total_sat += 1
                    if pfage >= interval:
                        if tag_sc.isSaturated():
                            correct_sat += 1
                        else:
                            tag_sc += 1
                    else:
                        tag_sc -= 1
            
            # s_dicts[f'sc-{sc_max}-{interval}-total-cnt'] = total_cnt
            # s_dicts[f'sc-{sc_max}-{interval}-total-sat'] = total_sat
            s_dicts[f'sc-{sc_max}-{interval}-total-sat-rate'] = total_sat/total_cnt
            if total_sat != 0:
                s_dicts[f'sc-{sc_max}-{interval}-correct-sat-rate'] = correct_sat/total_sat
            else:
                s_dicts[f'sc-{sc_max}-{interval}-correct-sat-rate'] = 0
    
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
    twriter = pd.ExcelWriter(os.path.join(sub_excel_dir_path, f'id-pfage-16bitTag.xlsx'), engine='xlsxwriter')


    worksname = os.listdir(base_dir) #like omnetpp-xalancbmk
    pf_works = [
		'GemsFDTD.06',		'astar.06',		'bwaves.06',		'bwaves.17',
		'bzip2.06',		'cactuBSSN.17',		'cactusADM.06',		'cam4.17',
		'cc_sv',		'dealII.06',		'fotonik3d.17',		'gcc.06',
		'gromacs.06',		'lbm.06',		'lbm.17',		'leslie3d.06',
		'libquantum.06',		'mcf.06',		'mcf.17',		'milc.06',
		'moses',		'nab.17',		'namd.17',		'omnetpp.06',
		'omnetpp.17',		'parest.17',		'perlbench.17',		'pr_spmv',
		'roms.17',		'soplex.06',		'sphinx',		'sphinx3.06',
		'sssp',		'tc',		'xalancbmk.06',		'xalancbmk.17',
		'xapian',		'xz.17',		'zeusmp.06',
	]
    worksname = pf_works
    pd_dict_list = []
    for i,work in enumerate(worksname):
        word_dir = os.path.join(base_dir,work)
        if not os.path.isdir(word_dir):
            continue
        pd_dict = analyze_one_workload_dict(word_dir,work)
        pd_dict['workload'] = work
        pd_dict_list.append(pd_dict)
 
    df = pd.DataFrame(pd_dict_list)
    df.to_excel(twriter, sheet_name='id-pfage', index=False)
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