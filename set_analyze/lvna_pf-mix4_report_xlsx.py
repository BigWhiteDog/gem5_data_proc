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
import openpyxl

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib import ticker

from set_analyze.my_diff_color import *

all_bench_list = [
    '004','220','130','031','022','103','202'
]

parser = argparse.ArgumentParser(description="options to get set stats")
# parser.add_argument('-d','--stats_dir', type=str,
#     help='stats dir to analyze',required=True)
# parser.add_argument('--ids',default=16,type=int)
# parser.add_argument('--nsamples',default=2,type=int)
# parser.add_argument('--l3_sets',default=4096,type=int)
parser.add_argument('-n','--ncore',type=int,default=4)
parser.add_argument('-j','--json', type=str,
    default=None)
parser.add_argument('--bench-choice',
		    choices=all_bench_list,
			)

opt = parser.parse_args()

confs=[
    # "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_oldincLRU_tailbm250M.json",
    # "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_oldincLRU16M_tailbm250M.json",
    # "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_goldencoveLRU_tailbm250M.json",
    "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_lvnapf_50M.json",
]

skip_res = [
    # re.compile(r'PrefetchSingleCorePolicy'),
    # re.compile(r'PageUCPPolicy'),
    # re.compile(r'CBP.*Policy'),
    # re.compile(r'Ctrl.*'),
]
interested_res = [
    # re.compile(r'IpcSample'),
    # re.compile(r'UCP'),
    # re.compile(r'FairPrefUCPPolicy-2M'),
]
xlsx_drop_res = [
    re.compile(r'nopart'),
]
force_update_res = [
    # re.compile(r'cppf.*UnusePF'),
]

xlsx_postfix = 'LabelPFALLCtrlLvP6'

def analyze_one_workload_dict(base_dir,workload_names,ncore=4):
    s_dicts = {}
    path_dicts = {}
    partsname = os.listdir(base_dir) #like l3-1,l3-nopoart,l3-1-csv
    for part in partsname:
        if not os.path.isdir(os.path.join(base_dir,part)):
            continue
        ways = part.split('-')
        if ways[0] != 'l3':
            continue
        new_base = os.path.join(base_dir,part)
        # ways[-1] as key
        dkey = re.split('-',part, maxsplit=1)[1]
        # if re.search(r'OneLessPageUCPPolicy', dkey):
        #     shutil.rmtree(new_base)
        #     continue
        force_update = False
        # force_update = True
        rmflag = False
        jump_flag = True
        if len(interested_res) > 0:
            for res in interested_res:
                if res.search(dkey):
                    jump_flag = False
                    break
        else:
            jump_flag = False
        #always jump skip_res
        for jres in skip_res:
            if jres.search(dkey):
                jump_flag = True
        
        for fres in force_update_res:
            if fres.search(dkey):
                force_update = True
                break

        if rmflag:
            shutil.rmtree(new_base)
            continue
        if jump_flag:
            continue

        last_nsamples=1

        if force_update:
            one_dict = extract_newgem_raw_json(new_base,ncores=ncore,last_nsamples=last_nsamples,need_hits=False)
        elif not os.path.exists(os.path.join(new_base,f'{last_nsamples}period.json')):
            one_dict = extract_newgem_raw_json(new_base,ncores=ncore,last_nsamples=last_nsamples,need_hits=False)
        else:
            with open(os.path.join(new_base,f'{last_nsamples}period.json'),'r') as f:
                one_dict = json.load(f)
        if dkey.isnumeric(): # l3-2
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
    s_dicts.pop('nopart')

    ipc_sum = 0
    for k in sorted(s_dicts.keys()):
        speedup_sum = 0
        for c in range(ncore):
            newipc = s_dicts[k][f'cpu{c}.ipc'][0]
            pd_dict[f'{k}_relperf{c}'] = newipc/baseline_cpuipcs[c]
            ipc_sum += newipc
            speedup_sum += newipc/baseline_cpuipcs[c]

        # pd_dict[f'{k}_ipcsum'] = ipc_sum
        pd_dict[f'{k}_speedupsum'] = speedup_sum/ncore - 1

    return pd_dict


def run_one_conf_one_bench(select_json:str,bench_choice:str):
    with open(select_json,'r') as f:
        global use_conf
        use_conf = json.load(f)
    if use_conf is None:
        exit(255)

    global test_prefix
    test_prefix = use_conf['test_prefix']

    ncore = opt.ncore

    wm_len = test_prefix.split('_')[1].strip('-')

    base_dir = f'/nfs/home/zhangchuanqi/lvna/for_xs/catlog/lvnapf-mix-test/mix{ncore}-{wm_len}-bench{bench_choice}'
    worksname = os.listdir(base_dir) #like omnetpp-xalancbmk
    pd_dict_list = []
    speedup_sum_list = []
    for i,work in enumerate(worksname):
        word_dir = os.path.join(base_dir,work)
        if not os.path.isdir(word_dir):
            continue
        pd_dict = analyze_one_workload_dict(word_dir,work,ncore=ncore)

        #calculate speedupsum
        speedupsum_keys = [ k for k in pd_dict.keys() if k.endswith('_speedupsum') ]
        speedupsum_dict = { 'workload': work }
        for k in speedupsum_keys:
            new_key = k.rsplit('_',1)[0]
            drop_flag = False
            for dre in xlsx_drop_res:
                if dre.search(new_key):
                    drop_flag = True
                    break
                
            if drop_flag:
                pd_dict.pop(k)
            else:
                speedupsum_dict[new_key] = pd_dict.pop(k)
        speedup_sum_list.append(speedupsum_dict)

        pd_dict_list.append(pd_dict)

    xlsx_path = f'set_analyze/{test_prefix}xls/mix{ncore}-{wm_len}-{xlsx_postfix}.xlsx'
    if not os.path.exists(xlsx_path):
        writer = pd.ExcelWriter(xlsx_path,mode='w',engine='openpyxl')
    else:
        writer = pd.ExcelWriter(xlsx_path,mode='a',if_sheet_exists='overlay',engine='openpyxl')

    new_pd_list = copy.deepcopy(pd_dict_list)

    # Write the dataframe to a new sheet
    df = pd.DataFrame(new_pd_list)
    df.to_excel(writer, sheet_name=f'{bench_choice}-speedup_over_nopart',
                index=False, float_format="%.5f")

    df = pd.DataFrame(speedup_sum_list)
    df.to_excel(writer, sheet_name=f'{bench_choice}-weightedspeedup-gain',
                index=False, float_format="%.5f")

    # Save the workbook
    writer.close()

def run_one_conf(select_json:str):
    if opt.bench_choice is None:
        for bench_choice in all_bench_list:
            run_one_conf_one_bench(select_json,bench_choice)
    else:
        run_one_conf_one_bench(select_json,opt.bench_choice)


if __name__ == '__main__':
    # base_dir = '/nfs/home/zhangchuanqi/lvna/for_xs/catlog/single-profiling/'
    if opt.json:
        select_json = opt.json
        run_one_conf(select_json)
    else:
        for co in confs:
            select_json = co
            run_one_conf(select_json)