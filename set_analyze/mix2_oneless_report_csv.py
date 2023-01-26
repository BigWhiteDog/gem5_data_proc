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

from cache_sensitive_names import *
from set_analyze.my_diff_color import *

parser = argparse.ArgumentParser(description="options to get set stats")
# parser.add_argument('-d','--stats_dir', type=str,
#     help='stats dir to analyze',required=True)
# parser.add_argument('--ids',default=16,type=int)
# parser.add_argument('--nsamples',default=2,type=int)
# parser.add_argument('--l3_sets',default=4096,type=int)

opt = parser.parse_args()

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
    pd_dict['workload0'] = workload_names.split('-')[0]
    pd_dict['workload1'] = workload_names.split('-')[1]

    # pd_dict['core0_setway'] = s_dicts['core0_setway']
    static_cpuipcs = [ s_dicts['static_waymask'][f'cpu{c}.ipc'][0] for c in range(ncore) ]
    for c in range(ncore):
        static_cpuipc = static_cpuipcs[c]
        # pd_dict[f'static_cpu{c}.ipc'] = static_cpuipc
        csv_cpuipc = s_dicts['csv'][f'cpu{c}.ipc'][0]
        pd_dict[f'csv_speedup_cpu{c}'] = csv_cpuipc/static_cpuipc
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
                if full_target != 32:
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


if __name__ == '__main__':
    ncore = 2
    use_conf = conf_50M
    test_prefix = use_conf['test_prefix']
    perf_prefix = '95perf'

    full_grow_dict = {}
    work0names = use_conf['cache_work_names']
    for w in work0names:
        full_grow_dict[w] = {}

    base_dir = f'/nfs/home/zhangchuanqi/lvna/for_xs/catlog/mix2-qosfromstart-core0-{test_prefix}{perf_prefix}'
    worksname = os.listdir(base_dir) #like omnetpp-xalancbmk
    pd_dict_list = []
    for i,work in enumerate(worksname):
        word_dir = os.path.join(base_dir,work)
        if not os.path.isdir(word_dir):
            continue
        pd_dict = analyze_one_workload_dict(word_dir,work,ncore=ncore)
        w0 = pd_dict['workload0']
        w1 = pd_dict['workload1']
        gd = pd_dict['fullgrow_dict']
        full_grow_dict[w0][w1] = gd

        pd_dict_list.append(pd_dict)

    target_perf_list = [0.97,0.975,0.98,0.985,0.99]
    for perf in target_perf_list:
        find_grow_target = {}
        print(f'perf:{perf}')
        for w0 in full_grow_dict:
            #for every w0 in full_grow_dict, find least grow target
            res_dicts = full_grow_dict[w0]
            for t in range(1,32):
                satisfy = True
                for w1 in res_dicts:
                    s0= res_dicts[w1][t][0]
                    s1= res_dicts[w1][t][1]
                    if s0 < perf:
                        satisfy = False
                        break
                if satisfy:
                    #t is the min satisfy grow target
                    find_grow_target[w0] = t
                    break
            print(f'w0:{w0} find_grow_target:{find_grow_target[w0]}')
        for p in pd_dict_list:
            w0 = p['workload0']
            w0t = find_grow_target[w0]
            for c in range(ncore):
                p[f'realOneWithTarget{perf}_speedup{c}'] = p['fullgrow_dict'][w0t][c]
    for p in pd_dict_list:
        p.pop('fullgrow_dict')

    # pd_dict_list = sorted(pd_dict_list,key=lambda x:x['csv_speedup_cpu1'],reverse=True)

    mycolumns = ['workload','core0_setway']
    # atd_len = [10,15,20]
    # c_formats = [
    #     'nopart_cpu{}.ipc',
    #     'static_cpu{}.ipc',
    #     # 'csv_cpu{}.ipc',
    #     'csv_speedup_cpu{}',
    # ]
    # for l in atd_len:
    #     # c_formats.append(f'atd{l}M_cpu{{}}.ipc')
    #     c_formats.append(f'atd{l}M_speedup_cpu{{}}')
    # # for l in atd_len:
    #     # c_formats.append(f'fullatd{l}M_speedup_cpu{{}}')
    # # for l in atd_len:
    # #     c_formats.append(f'hpatd{l}M_speedup_cpu{{}}')
    # for l in atd_len:
    #     c_formats.append(f'fmatd{l}M_speedup_cpu{{}}')
    # for cf in c_formats:
    #     for c in range(ncore):
    #         mycolumns.append(cf.format(c))

    df = pd.DataFrame(pd_dict_list)
    df.to_csv('set_analyze/mix2_oneless_report.csv',index=False,float_format="%.5f")
