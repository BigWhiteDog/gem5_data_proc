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
parser.add_argument('-n','--ncore',type=int,default=4)
parser.add_argument('-j','--json', type=str,
    default=None)
parser.add_argument('--bench-choice',
		    choices=['short','long'],
			default='short')

opt = parser.parse_args()

confs=[
    "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_oldincLRU_tailbm250M.json",
    # "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_goldencoveLRU_tailbm250M.json",
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
        new_base = os.path.join(base_dir,part)
        # ways[-1] as key
        dkey = re.split('-',part, maxsplit=1)[1]
        # if re.search(r'OneLessPageUCPPolicy', dkey):
        #     shutil.rmtree(new_base)
        #     continue
        rmflag = False
        if re.search(r'LocalUCPPolicy-[2,5]M', dkey):
            rmflag = True
        elif re.search(r'BaseUCPPolicy-2M', dkey):
            rmflag = True
        elif re.search(r'LessUCPPolicy-2M', dkey):
            rmflag = True
        elif re.search(r'PageUCPPolicy', dkey):
            # banlist = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
            # banlist = [16, 32, 64, 128,]
            banlist = [16, 32]
            # banlist = [16, 32, 128, 512, 2048]
            if any([ f'{w}' in dkey for w in banlist ]):
                continue
            
        if rmflag:
            shutil.rmtree(new_base)
            continue

        last_nsamples=1
        force_update = True

        if force_update:
            one_dict = extract_newgem_raw_json(new_base,ncores=ncore,last_nsamples=last_nsamples)
        elif not os.path.exists(os.path.join(new_base,f'{last_nsamples}period.json')):
            one_dict = extract_newgem_raw_json(new_base,ncores=ncore,last_nsamples=last_nsamples)
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
    ucp_baseline_cpuipcs = [ s_dicts['BaseUCPPolicy-10M'][f'cpu{c}.ipc'][0] for c in range(ncore) ]
    # s_dicts.pop('nopart')
    # get_policy_re = re.compile(r'(.*)Policy(.*)')
    # get_train_re = re.compile(r'(\d+)MTrain')
    # get_test_re = re.compile(r'(\d+)MTest')
    # get_grow_re = re.compile(r'grow(\d+)in(\d+)')
    # get_fullgrow_re = re.compile(r'fullgrow(\d+)in(\d+)')

    for k in sorted(s_dicts.keys()):
        # get_policy_res = get_policy_re.match(k)
        ipc_sum = 0
        speedup_sum = 0
        ucp_base_speedup_sum = 0
        for c in range(ncore):
            # pd_dict[f'{new_key}_ipc{c}'] = s_dicts[k][f'cpu{c}.ipc'][0]
            newipc = s_dicts[k][f'cpu{c}.ipc'][0]
            pd_dict[f'{k}_relperf{c}'] = newipc/baseline_cpuipcs[c]
            ipc_sum += newipc
            speedup_sum += newipc/baseline_cpuipcs[c]

            pd_dict[f'{k}_relperf{c}_ucp'] = newipc/ucp_baseline_cpuipcs[c]
            ucp_base_speedup_sum += newipc/ucp_baseline_cpuipcs[c]


        pd_dict[f'{k}_ipcsum'] = ipc_sum
        pd_dict[f'{k}_speedupsum'] = speedup_sum/ncore - 1
        pd_dict[f'{k}_speedupsum_ucp'] = ucp_base_speedup_sum/ncore - 1

    return pd_dict


def run_one_conf(select_json:str):
    with open(select_json,'r') as f:
        global use_conf
        use_conf = json.load(f)
    if use_conf is None:
        exit(255)

    global test_prefix
    test_prefix = use_conf['test_prefix']
    perf_prefix = '95perf'

    ncore = opt.ncore

    full_grow_dict = {}
    # work0names = use_conf['cache_work_names']
    # for w in work0names:
    #     full_grow_dict[w] = {}

    cache_type = test_prefix.split('_')[1]
    wm_len = test_prefix.split('_')[2].strip('-')

    base_dir = f'/nfs/home/zhangchuanqi/lvna/for_xs/catlog/newucp-mix{ncore}-{opt.bench_choice}-{cache_type}-{wm_len}'
    worksname = os.listdir(base_dir) #like omnetpp-xalancbmk
    pd_dict_list = []
    ipc_sum_list = []
    speedup_sum_list = []
    speedup_sum_ucp_list = []
    for i,work in enumerate(worksname):
        word_dir = os.path.join(base_dir,work)
        if not os.path.isdir(word_dir):
            continue
        pd_dict = analyze_one_workload_dict(word_dir,work,ncore=ncore)
        w0 = work.split('-',1)[0]
        other_w = work.split('-',1)[1]
        # gd = pd_dict['fullgrow_dict']
        # full_grow_dict[w0][other_w] = gd

        #calculate speedupsum
        speedupsum_keys = [ k for k in pd_dict.keys() if k.endswith('_speedupsum') ]
        speedupsum_dict = { 'workload': work }
        for k in speedupsum_keys:
            new_key = k.rsplit('_',1)[0]
            drop_flag = False
            if 'nopart' in new_key:
                drop_flag = True
            elif 'OneLess' in new_key or 'TwoLess' in new_key:
                if not "DynGroup" in new_key:
                    drop_flag = True
            elif re.search(r'[2,5]M', new_key):
                drop_flag = True
                
            if drop_flag:
                pd_dict.pop(k)
            else:
                speedupsum_dict[new_key] = pd_dict.pop(k)

        # intervals = ['2M','5M','10M']
        intervals = ['10M']
        #find best page out of pages_xM
        for inter in intervals:
            page_keys = [ k for k in speedupsum_dict.keys() if k.startswith('PageUCPPolicy') and k.endswith(inter) ]
            max_key = max(page_keys,key=lambda x: speedupsum_dict[x])
            speedupsum_dict[f'best_page_{inter}'] = speedupsum_dict[max_key]
            speedupsum_dict[f'best_page_{inter}_choice'] = max_key.split('-')[1]
        speedup_sum_list.append(speedupsum_dict)

        #calculate ipcsum
        ipcsum_keys = [ k for k in pd_dict.keys() if k.endswith('_ipcsum') ]
        ipcsum_dict = { 'workload': work }
        for k in ipcsum_keys:
            new_key = k.rsplit('_',1)[0]
            drop_flag = False
            if 'nopart' in new_key:
                drop_flag = True
            elif 'OneLess' in new_key or 'TwoLess' in new_key:
                if not "DynGroup" in new_key:
                    drop_flag = True
            if drop_flag:
                pd_dict.pop(k)
            else:
                ipcsum_dict[new_key] = pd_dict.pop(k)
        #find best page out of pages_10M
        for inter in intervals:
            page_keys = [ k for k in ipcsum_dict.keys() if k.startswith('PageUCPPolicy') and k.endswith(inter) ]
            max_key = max(page_keys,key=lambda x: ipcsum_dict[x])
            ipcsum_dict[f'best_page_{inter}'] = ipcsum_dict[max_key]
            ipcsum_dict[f'best_page_{inter}_choice'] = max_key.split('-')[1]
            # for k in page_keys:
            #     ipcsum_dict.pop(k)
        ipc_sum_list.append(ipcsum_dict)

        #calculate speedupsum_ucp
        speedupsum_ucp_keys = [ k for k in pd_dict.keys() if k.endswith('_speedupsum_ucp') ]
        speedupsum_ucp_dict = { 'workload': work }
        for k in speedupsum_ucp_keys:
            new_key = k.rsplit('_',2)[0]
            drop_flag = False
            if 'nopart' in new_key:
                drop_flag = True
            elif 'OneLess' in new_key or 'TwoLess' in new_key:
                if not "DynGroup" in new_key:
                    drop_flag = True
            elif re.search(r'[2,5]M', new_key):
                drop_flag = True
                
            if drop_flag:
                pd_dict.pop(k)
            else:
                speedupsum_ucp_dict[new_key] = pd_dict.pop(k)
        #find best page out of pages_xM
        for inter in intervals:
            page_keys = [ k for k in speedupsum_ucp_dict.keys() if k.startswith('PageUCPPolicy') and k.endswith(inter) ]
            max_key = max(page_keys,key=lambda x: speedupsum_ucp_dict[x])
            speedupsum_ucp_dict[f'best_page_{inter}'] = speedupsum_ucp_dict[max_key]
            speedupsum_ucp_dict[f'best_page_{inter}_choice'] = max_key.split('-')[1]
        speedup_sum_ucp_list.append(speedupsum_ucp_dict)


        pd_dict_list.append(pd_dict)

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    twriter = pd.ExcelWriter(f'set_analyze/newucp-mix{ncore}-{opt.bench_choice}-{cache_type}-{wm_len}-allpolicy.xlsx', engine='xlsxwriter')

    new_pd_list = copy.deepcopy(pd_dict_list)
            
    df = pd.DataFrame(new_pd_list)
    df.to_excel(twriter, sheet_name=f'speedup_over_nopart',index=False,float_format="%.5f")

    df = pd.DataFrame(ipc_sum_list)
    df.to_excel(twriter, sheet_name=f'ipcsum',index=False,float_format="%.5f")

    df = pd.DataFrame(speedup_sum_list)
    df.to_excel(twriter, sheet_name=f'weightedspeedup-gain',index=False,float_format="%.5f")

    df = pd.DataFrame(speedup_sum_ucp_list)
    df.to_excel(twriter, sheet_name=f'weightedspeedup-gain-ucp',index=False,float_format="%.5f")

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