import copy
from genericpath import isdir
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
force_update_res = [
    # re.compile(r'IpcSample'),
    # re.compile(r'UCP'),
    # re.compile(r'PrefUCP'),
]
xlsx_drop_res = [
    re.compile(r'nopart'),
]

rm_res = [
    # re.compile(r'IdPassLRU'),
]

def analyze_one_workload_dict(base_dir,workload_names,ncore=2):
    s_dicts = {}
    path_dicts = {}
    partsname = os.listdir(base_dir) #like l3-1,l3-nopoart,l3-1-csv
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
        
        force_update = False
        rmflag = False
        jump_flag = False
        for res in skip_res:
            if res.search(dkey):
                jump_flag = True
        for res in force_update_res:
            if res.search(dkey):
                force_update = True
        for res in rm_res:
            if res.search(dkey):
                rmflag = True
            
        if rmflag:
            shutil.rmtree(new_base)
            continue
        if jump_flag:
            continue

        last_nsamples=1

        if force_update:
            one_dict = extract_newgem_raw_json(new_base,ncores=ncore,last_nsamples=last_nsamples)
        elif not os.path.exists(os.path.join(new_base,f'{last_nsamples}period.json')):
            one_dict = extract_newgem_raw_json(new_base,ncores=ncore,last_nsamples=last_nsamples)
        else:
            with open(os.path.join(new_base,f'{last_nsamples}period.json'),'r') as f:
                one_dict = json.load(f)
        
        #special case for single core
        if ncore == 1:
            one_dict['cpu0.ipc'] = one_dict['cpu.ipc']

        if ways[1].isnumeric(): # l3-2 or l3-2-pf
            continue #skip
            s_dicts['core0_setway'] = dkey
            s_dicts['static_waymask'] = one_dict
        else:
            s_dicts[dkey] = one_dict
            path_dicts[dkey] = new_base

    pd_dict = {}
    pd_dict[f'workload'] = workload_names
    # for i,wn in enumerate(workload_names.split('-')):
    #     pd_dict[f'workload{i}'] = wn

    baseline_cpuipcs = [ s_dicts['nopart'][f'cpu{c}.ipc'][0] for c in range(ncore) ]

    for k in sorted(s_dicts.keys()):
        ipc_sum = 0
        speedup_sum = 0
        for c in range(ncore):
            newipc = s_dicts[k][f'cpu{c}.ipc'][0]
            pd_dict[f'{k}_relperf{c}'] = newipc/baseline_cpuipcs[c]
            ipc_sum += newipc
            speedup_sum += newipc/baseline_cpuipcs[c]

        pd_dict[f'{k}_ipcsum'] = ipc_sum
        pd_dict[f'{k}_speedupsum'] = speedup_sum/ncore - 1

    return pd_dict


def run_one_conf(select_json:str):
    with open(select_json,'r') as f:
        global use_conf
        use_conf = json.load(f)
    if use_conf is None:
        exit(255)

    global test_prefix
    test_prefix = use_conf['test_prefix']

    base_dir_format = use_conf['base_dir_format']
    test_prefix = use_conf['test_prefix']
    base_dir = base_dir_format.format(test_prefix)

    ncore = opt.ncore

    wm_len = test_prefix.split('_')[1].strip('-')

    worksname = os.listdir(base_dir) #like omnetpp-xalancbmk
    pd_dict_list = []
    speedup_sum_list = []
    for i,work in enumerate(worksname):
        word_dir = os.path.join(base_dir,work)
        if not os.path.isdir(word_dir):
            continue
        pd_dict = analyze_one_workload_dict(word_dir,work,ncore=ncore)
        if ncore > 1:
            w0 = work.split('-',1)[0]
            other_w = work.split('-',1)[1]

        #calculate speedupsum
        speedupsum_keys = [ k for k in pd_dict.keys() if k.endswith('_speedupsum') ]
        speedupsum_dict = { 'workload': work }
        for k in speedupsum_keys:
            new_key = k.rsplit('_',1)[0]
            drop_flag = False
            for res in xlsx_drop_res:
                if res.search(new_key):
                    drop_flag = True
                
            if drop_flag:
                pd_dict.pop(k)
            else:
                speedupsum_dict[new_key] = pd_dict.pop(k)

        # intervals = ['2M','5M','10M']
        intervals = ['10M']
        #find best page out of pages_xM
        # for inter in intervals:
        #     page_keys = [ k for k in speedupsum_dict.keys() if k.startswith('PageUCPPolicy') and k.endswith(inter) ]
        #     max_key = max(page_keys,key=lambda x: speedupsum_dict[x])
        #     speedupsum_dict[f'best_page_{inter}'] = speedupsum_dict[max_key]
        #     speedupsum_dict[f'best_page_{inter}_choice'] = max_key.split('-')[1]
        speedup_sum_list.append(speedupsum_dict)


        pd_dict_list.append(pd_dict)

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    twriter = pd.ExcelWriter(f'set_analyze/lvna_pf-mix{ncore}-allpolicy.xlsx', engine='xlsxwriter')

    new_pd_list = copy.deepcopy(pd_dict_list)
            
    df = pd.DataFrame(new_pd_list)
    df.to_excel(twriter, sheet_name=f'speedup_over_nopart',index=False,float_format="%.5f")

    df = pd.DataFrame(speedup_sum_list)
    df.to_excel(twriter, sheet_name=f'weightedspeedup-gain',index=False,float_format="%.5f")

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