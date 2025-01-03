import os
import re
import numpy as np
import utils.common as c
import utils.target_stats as t
from utils.common import extract_newgem_raw_json
import numpy as np
import argparse
import math
import shutil

import json

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib import ticker
from matplotlib.patches import Patch
import sqlite3
from set_analyze.my_diff_color import *

parser = argparse.ArgumentParser(description="options to get set stats")
parser.add_argument('-j','--json', type=str,
    default=None)

opt = parser.parse_args()

confs=[
    "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_lvnapf_50M.json",
]

skip_res = [
    # re.compile(r'PrefetchSingleCorePolicy'),
    # re.compile(r'PageUCPPolicy'),
]
# ['DSSPRUAPLvLRURP','DSS256PRUAPLvLRURP',
# 							'DSSPRUAP2W1LvLRURP','DSS256PRUAP2W1LvLRURP',
# 							'DSSPRUAP3W1LvLRURP','DSS256PRUAP3W1LvLRURP',
# 							'DSSPRUAP3W2LvLRURP','DSS256PRUAP3W2LvLRURP']
interested_res = [
    re.compile(r'pf'),
    # re.compile(r'IpcSample'),
    # re.compile(r'NDSS256[AP]hPSerScRUAPLvLRURP'),
    # re.compile(r'NDSS256[AP]hPSerSc1RUAPLvLRURP'),
    # re.compile(r'DSS256PRUAP\dW\dLvLRURP'),
]
xlsx_drop_res = [
    # re.compile(r'nopart'),
]

pf_src_num = 13

# test_way = 4

def workname_postfix(workname):
    if workname in cs_works:
        return 'cs'
    elif workname in ps_works:
        return 'ps'
    elif workname in cs_ps_works:
        return 'cs-ps'
    return ''

def draw_ipc_access_issue_bar(ax,s_dicts,workload_name,full,pos:tuple):

    labels = [
        'pfIssued',
        'l3Access',
        'l3HitRate',
        'nipc',
    ]
    return labels

def draw_ipc_access_plot(ax,s_dicts,workload_name,full,pos:tuple):

    labels = [
        'l3Access',
        'nipc',
    ]
    return labels



def draw_db_by_func(base_dir,n_rows,worksname_waydict,draw_one_func,fig_name,n_cols=6,
                    force_update_json = False):

    work_stats_dict = {}
    dict_updated = False

    n_rows = math.ceil(len(worksname_waydict) / n_cols)


    force_update = False
    # force_update = True
    last_nsamples=1
    ncore = 1

    # test_way = 4
    bypass_name_list = [f'l3-bypassPfSrc{sr}' for sr in range(pf_src_num)]
    
    for i,work in enumerate(worksname_waydict):
        work = worksname[i]
        word_dir = os.path.join(base_dir,work)
        if not os.path.isdir(word_dir):
            continue

        for sr in range(pf_src_num):
            bypass_name = f'l3-bypassPfSrc{sr}'
            old_dir = os.path.join(word_dir,bypass_name)
            new_dir = os.path.join(word_dir,f'l3-16-bypassPfSrc{sr}')
            # print(f'{old_dir} -> {new_dir}')
            if os.path.isdir(old_dir):
                # print(f'mv {old_dir} {new_dir}')
                # os.system(f'mv {old_dir} {new_dir}')
                shutil.move(old_dir,new_dir)


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
    base_dir = base_dir_format.format(test_prefix)
    base_dir += '-new'
    
    pic_dir_path = f'set_analyze/{test_prefix}pics'
    os.makedirs(pic_dir_path, exist_ok=True)

    global cs_works
    cs_works = use_conf['cs_works']
    global ps_works
    ps_works = use_conf['ps_works']
    global cs_ps_works
    cs_ps_works = use_conf['cs-ps_works']


    global worksname
    worksname = os.listdir(base_dir)
    worksname.sort()
    # worksname = cs_works + cs_ps_works + ps_works
    # worksname.sort()

    interested_works = {}
    for w in worksname:
        interested_works[w] = max_assoc    

    n_works = len(worksname)
    n_cols = 4
    n_rows = math.ceil(n_works/n_cols)
    draw_db_by_func(base_dir,n_rows,interested_works,
                    n_cols=n_cols,
        draw_one_func=draw_ipc_access_issue_bar,fig_name=os.path.join(pic_dir_path,'none.png'))

if __name__ == '__main__':
    if opt.json:
        select_json = opt.json
        run_one_conf(select_json)
    else:
        for co in confs:
            select_json = co
            run_one_conf(select_json)