from genericpath import isdir
import os
import numpy as np
import utils.common as c
from utils.common import extract_newgem_raw_json
import utils.target_stats as t
import csv
import numpy as np
import argparse
import math
import itertools
import random

import json

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib import ticker

# from cache_sensitive_names import *

parser = argparse.ArgumentParser(description="options to get set stats")
# parser.add_argument('-d','--stats_dir', type=str,
#     help='stats dir to analyze',required=True)
# parser.add_argument('--ids',default=16,type=int)
# parser.add_argument('--nsamples',default=2,type=int)
# parser.add_argument('--l3_sets',default=4096,type=int)
parser.add_argument('-j','--json', type=str,
    default=None)
parser.add_argument('-l',choices=['short','long'],default='long')
parser.add_argument('-n',type=int,default=4)

opt = parser.parse_args()

confs=[
    # "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_xs_tailbm50M.json",
    # "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_goldencove_tailbm50M.json",
    # "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_goldencoveLRU_tailbm50M.json",
    # "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_goldencoveLRU_tailbm250M.json",
    "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_oldincLRU_tailbm50M.json",
    "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_oldincLRU_tailbm250M.json",
    # "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_oldinc_tailbm50M.json",
    # "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_skylake_tailbm50M.json",
]

def run_one_conf(select_json:str):
    with open(select_json,'r') as f:
        global use_conf
        use_conf = json.load(f)
    if use_conf is None:
        exit(255)
    base_dir_format = use_conf['base_dir_format']
    test_prefix = use_conf['test_prefix']
    base_dir = base_dir_format.format(test_prefix)

    need_work_names = use_conf["cache_work_names"]
    num_works = len(need_work_names)

    #generate work lists
    work_lists = set()

    # for hp in need_work_names:
    one_sample_num = opt.n
    if opt.l == 'long':
        iter_num = 16 * opt.n
    else:
        iter_num = opt.n
    remain_work_set = set()

    assert(iter_num % one_sample_num == 0)

    for _ in range(iter_num):
        iter_work_name_set = set(need_work_names)
        # iter_work_name_set.remove(hp)
        #handle remain
        remain_num = len(remain_work_set)
        if remain_num > 0:
            need_num = one_sample_num - remain_num
            next_candidate_set = iter_work_name_set - remain_work_set
            sampled_items = random.sample(next_candidate_set, need_num)
            for item in sampled_items:
                iter_work_name_set.remove(item)
            # new_work_combine = [hp]
            new_work_combine = []
            new_work_combine.extend(remain_work_set)
            new_work_combine.extend(sampled_items)
            remain_work_set.clear()
            work_lists.add('-'.join(new_work_combine))
            # work_lists.append('-'.join(new_work_combine))
        
        #handle other iter_work_name_set
        while len(iter_work_name_set) >= one_sample_num:
            sampled_items = random.sample(iter_work_name_set, one_sample_num)
            for item in sampled_items:
                iter_work_name_set.remove(item)
            # new_work_combine = [hp]
            new_work_combine = []
            new_work_combine.extend(sampled_items)
            work_lists.add('-'.join(new_work_combine))
            # work_lists.append('-'.join(new_work_combine))

        #store remain
        for item in iter_work_name_set:
            remain_work_set.add(item)
        iter_work_name_set.clear()

    json_dir_path = os.path.dirname(select_json)
    old_json_name = os.path.basename(select_json)
    new_json_name = old_json_name.replace("conf",f"{opt.l}_benchs_{opt.n}")
    new_json_path = os.path.join(json_dir_path,new_json_name)

    final_dict = {}
    final_dict[f'{opt.n}bench'] = list(work_lists)
    with open(new_json_path,'w') as f:
        json.dump(final_dict,f,indent=4)

if __name__ == '__main__':
    if opt.json:
        select_json = opt.json
        run_one_conf(select_json)
    else:
        for co in confs:
            select_json = co
            run_one_conf(select_json)
