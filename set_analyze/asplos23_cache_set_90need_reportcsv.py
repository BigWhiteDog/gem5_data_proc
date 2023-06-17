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
parser.add_argument('-j','--json', type=str,
    default=None)

opt = parser.parse_args()

confs=[
    # "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_xs_tailbm50M.json",
    "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_goldencove_tailbm50M.json",
    "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_oldinc_tailbm50M.json",
    "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_skylake_tailbm50M.json",
]


from set_analyze.my_diff_color import *


def outputcsv_minway_fromdb(csv_top_dir,work,s_dicts):
    csv_file = os.path.join(csv_top_dir,f'{work}.csv')
    with open(csv_file,'w') as f:
        for setneed in s_dicts['min_ways_no_extra_miss']:
            f.write(f'{setneed}\n')

def analyze_minway_fromdb(work_stats_dict,work,work_dir,full_ass):
    # if work in work_stats_dict:
    #     return
    s_2 = re.compile(r'(\w+)-([\w\.]+)')

    s_dicts = {}
    s_dicts['unique_blocks_number'] = [0 for _ in range(all_set)]
    s_dicts['unique_reused_blocks_number'] = [0 for _ in range(all_set)]
    s_dicts['ways_miss_cnt'] = {}
    s_dicts['ways_miss_rate'] = {}

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
        if ways > full_ass:
            continue

        new_base = os.path.join(work_dir,part)
        db_path = os.path.join(new_base,'hm.db')
        all_access_query = 'SELECT SETIDX,sum(ISMISS),count(*) FROM HitMissTrace group by SETIDX;'
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        f = cur.execute(all_access_query)

        s_dicts['ways_miss_cnt'][ways] = [0 for _ in range(all_set)]
        s_dicts['ways_miss_rate'][ways] = [0 for _ in range(all_set)]
        for setidx,msc,allaccess in f:
            idx = int(setidx)
            msc = int(msc)
            allaccess = int(allaccess)
            s_dicts['ways_miss_cnt'][ways][idx] = msc
            s_dicts['ways_miss_rate'][ways][idx] = msc/allaccess

        cur.close()
    s_dicts['min_ways_no_extra_miss'] = [full_ass for _ in range(all_set)]
    s_dicts['min_ways_1_extra_miss'] = [full_ass for _ in range(all_set)]
    s_dicts['min_ways_2_extra_miss'] = [full_ass for _ in range(all_set)]
    fullass_miss_cnt = s_dicts['ways_miss_cnt'][full_ass]

    for ways in s_dicts['ways_miss_cnt']:
        my_way_miss_cnt = s_dicts['ways_miss_cnt'][ways]
        for idx in range(all_set):
            delta_miss = my_way_miss_cnt[idx] - fullass_miss_cnt[idx]
            if delta_miss <= 0:
                s_dicts['min_ways_no_extra_miss'][idx] = min(s_dicts['min_ways_no_extra_miss'][idx],ways)
            if delta_miss <= 1:
                s_dicts['min_ways_1_extra_miss'][idx] = min(s_dicts['min_ways_1_extra_miss'][idx],ways)
            if delta_miss <= 2:
                s_dicts['min_ways_2_extra_miss'][idx] = min(s_dicts['min_ways_2_extra_miss'][idx],ways)

    work_stats_dict[work] = s_dicts


def outputcsv_by_func(base_dir,worksname_waydict,analyze_func,output_csv_func,csv_top_dir,input_stats_dict):

    work_stats_dict = input_stats_dict

    for i,work in enumerate(worksname_waydict):
        full_ass = worksname_waydict[work]
        work_dir = os.path.join(base_dir,work)
        if not os.path.isdir(work_dir):
            continue
        analyze_func(work_stats_dict,work,work_dir,full_ass)
        s_dicts = work_stats_dict[work]
        output_csv_func(csv_top_dir,work,s_dicts)

    return work_stats_dict

def run_one_conf(select_json:str):
    with open(select_json,'r') as f:
        global use_conf
        use_conf = json.load(f)
    if use_conf is None:
        exit(255)

    global test_prefix
    test_prefix = use_conf['test_prefix']
    base_dir_format = use_conf['base_dir_format']
    base_dir = base_dir_format.format(test_prefix)
    csv_dir_path = f'set_analyze/{test_prefix}other/csv'
    os.makedirs(csv_dir_path, exist_ok=True)
    worksname = use_conf['cache_work_names'] #like mcf

    n_works = len(worksname)
    n_rows = math.ceil(n_works/4)

    global all_set
    all_set = use_conf['all_set']
    global max_assoc
    max_assoc = use_conf['max_assoc']

    waydict_format = 'cache_work_{}ways'
    # perf_prefixs = ['90perf','95perf','full']
    perf_prefixs = ['95perf']
    csvfunc_dir_pair = [
        (outputcsv_minway_fromdb,os.path.join(csv_dir_path,'min0way_{}')),
        # (draw_one_workload_pn_blocklen,'pn_est_blocklen_contour_{}.png'),
        # (draw_one_workload_pn_cyclelen,'pn_est_cyclelen_contour_{}.png'),
    ]

    for perf_prefix in perf_prefixs:
        waydict_name = waydict_format.format(perf_prefix)
        waydict = use_conf[waydict_name]
        ret_dict = {}
        for csvfunc,csv_dir_format in csvfunc_dir_pair:
            csv_dir = csv_dir_format.format(perf_prefix)
            os.makedirs(csv_dir, exist_ok=True)
            outputcsv_by_func(base_dir,waydict,
            analyze_func=analyze_minway_fromdb,
            output_csv_func=csvfunc,
            csv_top_dir=csv_dir,
            input_stats_dict=ret_dict)

if __name__ == '__main__':
    # base_dir = '/nfs/home/zhangchuanqi/lvna/for_xs/catlog/single-profiling/'
    if opt.json:
        select_json = opt.json
        run_one_conf(select_json)
    else:
        for co in confs:
            select_json = co
            run_one_conf(select_json)