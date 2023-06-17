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
parser.add_argument('-j','--json', type=str,
    default=None)

opt = parser.parse_args()

confs=[
    # "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_xs_tailbm50M.json",
    "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_goldencove_tailbm50M.json",
    # "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_oldinc_tailbm50M.json",
    # "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_skylake_tailbm50M.json",
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
        dkey = ways[-1]
        if dkey == 'BaseAtdPolicy':
            # print(new_base)
            # print('11')
            # shutil.rmtree(new_base)
            continue
        if re.search(r'fullgrow(\d+)in16',dkey):
            print(new_base)
            print('22')
            # shutil.rmtree(new_base)
            continue
        last_nsamples=1
        # one_dict = extract_newgem_raw_json(new_base,ncores=ncore,last_nsamples=last_nsamples)
        with open(os.path.join(new_base,f'{last_nsamples}period.json'),'r') as f:
            one_dict = json.load(f)
        if dkey.isnumeric():
            s_dicts['core0_setway'] = dkey
            s_dicts['static_waymask'] = one_dict
        else:
            s_dicts[dkey] = one_dict
            path_dicts[dkey] = new_base

    pd_dict = {}
    for i,wn in enumerate(workload_names.split('-')):
        pd_dict[f'workload{i}'] = wn

    # pd_dict['core0_setway'] = s_dicts['core0_setway']
    static_cpuipcs = [ s_dicts['static_waymask'][f'cpu{c}.ipc'][0] for c in range(ncore) ]
    # for c in range(ncore):
    #     static_cpuipc = static_cpuipcs[c]
    #     # pd_dict[f'static_cpu{c}.ipc'] = static_cpuipc
    #     csv_cpuipc = s_dicts['csv'][f'cpu{c}.ipc'][0]
    #     pd_dict[f'csv_speedup_cpu{c}'] = csv_cpuipc/static_cpuipc
    for c in range(ncore):
        pd_dict[f'nopart_speedup_cpu{c}'] = s_dicts['nopart'][f'cpu{c}.ipc'][0]/static_cpuipcs[c]
    get_policy_re = re.compile(r'(.*)Policy(.*)')
    get_train_re = re.compile(r'(\d+)MTrain')
    get_test_re = re.compile(r'(\d+)MTest')
    get_grow_re = re.compile(r'grow(\d+)in(\d+)')
    get_fullgrow_re = re.compile(r'fullgrow(\d+)in(\d+)')

    pd_dict['fullgrow_dict'] = {}

    for k in sorted(s_dicts.keys()):
        get_policy_res = get_policy_re.match(k)
        if get_policy_res:
            policy = get_policy_res.group(1)
            tt_s = get_policy_res.group(2)
            get_train_res = get_train_re.search(tt_s)
            new_key = policy
            if policy in ['OneLessGrowTarget']:
                continue
            if get_train_res:
                train_size = get_train_res.group(1)
                new_key += f'_{train_size}M_train'
                #skip train
                continue
            get_test_res = get_test_re.search(tt_s)
            if get_test_res:
                test_size = get_test_res.group(1)
                new_key += f'_{test_size}M_test'
            get_grow_res = get_grow_re.match(tt_s)
            if get_grow_res:
                new_key += f'_grow{get_grow_res.group(1)}in{get_grow_res.group(2)}'
                #skip part grow
                continue
            get_fullgrow_res = get_fullgrow_re.match(tt_s)
            if get_fullgrow_res:
                full_target = int(get_fullgrow_res.group(2))
                if full_target != 64:
                    continue
                grow_target = int(get_fullgrow_res.group(1))
                # pd_tmp_key = new_key + 'fullgrow_dict'
                pd_tmp_key = 'fullgrow_dict'
                pd_dict[pd_tmp_key][grow_target] = {}
                for c in range(ncore):
                    newipc = s_dicts[k][f'cpu{c}.ipc'][0]
                    pd_dict[pd_tmp_key][grow_target][c] = newipc/static_cpuipcs[c]
                new_key += f'_fullgrow{get_fullgrow_res.group(1)}in{get_fullgrow_res.group(2)}'
                continue
            # if k in ['HitPreventMaskCsvPolicy'] or policy in ['FullAtd']:
            #     shutil.rmtree(path_dicts[k])
            for c in range(ncore):
                # pd_dict[f'{new_key}_ipc{c}'] = s_dicts[k][f'cpu{c}.ipc'][0]
                newipc = s_dicts[k][f'cpu{c}.ipc'][0]
                pd_dict[f'{new_key}_speedup{c}'] = newipc/static_cpuipcs[c]

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

    ncore = 4

    full_grow_dict = {}
    work0names = use_conf['cache_work_names']
    for w in work0names:
        full_grow_dict[w] = {}

    base_dir = f'/nfs/home/zhangchuanqi/lvna/for_xs/catlog/mix{ncore}-qosfromstart-core0-{test_prefix}{perf_prefix}'
    worksname = os.listdir(base_dir) #like omnetpp-xalancbmk
    pd_dict_list = []
    for i,work in enumerate(worksname):
        word_dir = os.path.join(base_dir,work)
        if not os.path.isdir(word_dir):
            continue
        pd_dict = analyze_one_workload_dict(word_dir,work,ncore=ncore)
        w0 = work.split('-',1)[0]
        other_w = work.split('-',1)[1]
        gd = pd_dict['fullgrow_dict']
        full_grow_dict[w0][other_w] = gd

        pd_dict_list.append(pd_dict)

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    twriter = pd.ExcelWriter(f'set_analyze/asplos23-{test_prefix}allpolicy.xlsx', engine='xlsxwriter')

    dflist = []
    tstep_list = [1,2,4,8]
    for tstep in tstep_list:
        new_pd_list = copy.deepcopy(pd_dict_list)
        target_perf_list = [0.97,0.975,0.98,0.985,0.99]
        maxbuckets = 64//tstep
        for perf in target_perf_list:
            find_grow_target = {}
            print(f'perf:{perf}')
            for w0 in full_grow_dict:
                #for every w0 in full_grow_dict, find least grow target
                res_dicts = full_grow_dict[w0]
                target_max_s1 = {}
                for t in range(tstep,65,tstep):
                    satisfy = True
                    s1_max = 0
                    s1_max_t = 0
                    for w1 in res_dicts:
                        s0= res_dicts[w1][t][0]
                        other_speeds = [res_dicts[w1][t][i] for i in range(1,ncore)]
                        s1= np.average(other_speeds)
                        if s0 < perf:
                            satisfy = False
                            break
                        if s1 > s1_max:
                            s1_max = s1
                            s1_max_t = t
                    if satisfy:
                        #t is the min satisfy grow target
                        # find_grow_target[w0] = t
                        # break
                        target_max_s1[t] = s1_max
                final_t = max(target_max_s1, key=lambda k:target_max_s1[k])
                find_grow_target[w0] = final_t
                print(f'w0:{w0} find_{maxbuckets}_grow_target:{find_grow_target[w0]//tstep}')
            
            grow_df = pd.DataFrame(find_grow_target,index=[0])
            grow_df.to_excel(twriter, sheet_name=f'{maxbuckets}_target_{perf}_in64')
            
            for p in new_pd_list:
                w0 = p['workload0']
                w0t = find_grow_target[w0]
                for c in range(ncore):
                    p[f'realOneWithTarget{perf}_speedup{c}'] = p['fullgrow_dict'][w0t][c]
        for p in new_pd_list:
            p.pop('fullgrow_dict')

        mycolumns = ['workload','core0_setway']

        df = pd.DataFrame(new_pd_list)
        df.to_csv(f'set_analyze/mix{ncore}_oneless_report{maxbuckets}.csv',index=False,float_format="%.5f")
        df.to_excel(twriter, sheet_name=f'{maxbuckets}bucket')

    twriter.close()

if __name__ == '__main__':
    # base_dir = '/nfs/home/zhangchuanqi/lvna/for_xs/catlog/single-profiling/'
    if opt.json:
        select_json = opt.json
        run_one_conf(select_json)
    else:
        for co in confs:
            select_json = co
            run_one_conf(select_json)