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
import time

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
# parser.add_argument('-n',type=int,default=4)

opt = parser.parse_args()

confs=[
    "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_lvnapf_50M.json",
]

def gen_balance_pair_list(list_of_worklist:list,reorder_in_pair=True):
    pair_set = set()
    out_list = []
    len_onelist = len(list_of_worklist[0])
    for wl in list_of_worklist:
        assert(len(wl) == len_onelist)
    
    try_ins_time = 0
    for _ in range(len_onelist):
        find_flag = False
        while find_flag == False:
            one_pair = []
            sort_pair = []
            for wl in list_of_worklist:
                tmp_work = wl.pop()
                while tmp_work in one_pair:
                    wl.insert(0,tmp_work)
                    tmp_work = wl.pop()
                one_pair.append(tmp_work)
                sort_pair.append(tmp_work)
            sort_pair.sort()
            pair_key = '-'.join(sort_pair)
            if pair_key not in pair_set:
                pair_set.add(pair_key)
                find_flag = True
                #shuffle one_pair to make workpair random
                if reorder_in_pair:
                    random.shuffle(one_pair)
                out_list.append('-'.join(one_pair))
            else:
                try_ins_time += 1
                for wl in list_of_worklist:
                    wl.insert(0,one_pair.pop(0))
                    random.shuffle(wl)
            if try_ins_time > 1000:
                print(list_of_worklist)
                print(f"Error: try insert time > 1000")
                exit(255)
    return out_list

def run_one_conf(select_json:str):
    with open(select_json,'r') as f:
        global use_conf
        use_conf = json.load(f)
    if use_conf is None:
        exit(255)
    base_dir_format = use_conf['base_dir_format']
    test_prefix = use_conf['test_prefix']
    base_dir = base_dir_format.format(test_prefix)

    cap_sense_names = set(use_conf["cache_cap_sense_names"])
    pf_sense_names = set(use_conf["cache_pf_sense_names"])

    only_cap_sense_names = cap_sense_names - pf_sense_names
    only_pf_sense_names = pf_sense_names - cap_sense_names
    both_sense_names = cap_sense_names & pf_sense_names
    print(f"only_cap_sense_names:{only_cap_sense_names} len:{len(only_cap_sense_names)}")
    print(f"only_pf_sense_names:{only_pf_sense_names} len:{len(only_pf_sense_names)}")
    print(f"both_sense_names:{both_sense_names} len:{len(both_sense_names)}")

    use_conf['cs_works'] = list(only_cap_sense_names)
    use_conf['ps_works'] = list(only_pf_sense_names)
    use_conf['cs-ps_works'] = list(both_sense_names)

    # need_work_names = use_conf["cache_work_names"]
    # num_works = len(need_work_names)


    json_dir_path = os.path.dirname(select_json)
    old_json_name = os.path.basename(select_json)
    new_json_name = old_json_name.replace("conf",f"mixpf_bm_4")
    new_json_path = os.path.join(json_dir_path,new_json_name)

    final_dict = {}
    
    with open(new_json_path,'r') as f:
        json_data = json.load(f)
        #update dict
        final_dict.update(json_data)
    
    random.seed(time.time())
    #len of lists
    # CS-4 PS-16 CS-PS-12
    ncore = 4
    cs_n144 = list(only_cap_sense_names) * 36
    ps_n144 = list(only_pf_sense_names) * 9
    cs_ps_n144 = list(both_sense_names) * 12
    # #gen 0 0 4
    # list004 = []
    # for _ in range(ncore):
    #     tmp_list = cs_ps_n144.copy()
    #     random.shuffle(tmp_list)
    #     list004.append(tmp_list)
    # final_dict[f'004_bench'] = gen_balance_pair_list(list004)
    # #gen 2 2 0
    # list220 = [
    #     cs_n144.copy(),
    #     cs_n144.copy(),
    #     ps_n144.copy(),
    #     ps_n144.copy()
    # ]
    # for wl in list220:
    #     random.shuffle(wl)
    # final_dict[f'220_bench'] = gen_balance_pair_list(list220)
    # #gen 1 3 0
    # list130 = [
    #     cs_n144.copy(),
    #     ps_n144.copy(),
    #     ps_n144.copy(),
    #     ps_n144.copy()
    # ]
    # for wl in list130:
    #     random.shuffle(wl)
    # final_dict[f'130_bench'] = gen_balance_pair_list(list130)
    # #gen 0 3 1
    # list031 = [
    #     ps_n144.copy(),
    #     ps_n144.copy(),
    #     ps_n144.copy(),
    #     cs_ps_n144.copy()
    # ]
    # for wl in list031:
    #     random.shuffle(wl)
    # final_dict[f'031_bench'] = gen_balance_pair_list(list031)
    # #gen 0 2 2
    # list022 = [
    #     ps_n144.copy(),
    #     ps_n144.copy(),
    #     cs_ps_n144.copy(),
    #     cs_ps_n144.copy()
    # ]
    # for wl in list022:
    #     random.shuffle(wl)
    # final_dict[f'022_bench'] = gen_balance_pair_list(list022)
    # #gen 1 0 3
    # list103 = [
    #     cs_n144.copy(),
    #     cs_ps_n144.copy(),
    #     cs_ps_n144.copy(),
    #     cs_ps_n144.copy()
    # ]
    # for wl in list103:
    #     random.shuffle(wl)
    # final_dict[f'103_bench'] = gen_balance_pair_list(list103)
    # #gen 2 0 2
    # list202 = [
    #     cs_n144.copy(),
    #     cs_n144.copy(),
    #     cs_ps_n144.copy(),
    #     cs_ps_n144.copy()
    # ]
    # for wl in list202:
    #     random.shuffle(wl)
    # final_dict[f'202_bench'] = gen_balance_pair_list(list202)
    #gen 1 0 3 no reorder
    list103 = [
        cs_n144.copy(),
        cs_ps_n144.copy(),
        cs_ps_n144.copy(),
        cs_ps_n144.copy()
    ]
    for wl in list103:
        random.shuffle(wl)
    final_dict[f'fix103_bench'] = gen_balance_pair_list(list103,reorder_in_pair=False)

    with open(new_json_path,'w') as f:
        json.dump(final_dict,f,indent=4)
    
    # with open(select_json,'w') as f:
    #     json.dump(use_conf,f,indent=4)

if __name__ == '__main__':
    if opt.json:
        select_json = opt.json
        run_one_conf(select_json)
    else:
        for co in confs:
            select_json = co
            run_one_conf(select_json)
