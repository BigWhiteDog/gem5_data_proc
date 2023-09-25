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
    # "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_goldencove48M_tailbm50M.json",
    "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_goldencoveLRU_tailbm250M.json",
    # "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_goldencoveLRU_tailbm50M.json",
]

# from cache_sensitive_names import *
from set_analyze.my_diff_color import *

def draw_one_workload_way_need(ax,s_dicts,workload_name,full_ass,pos:tuple):
    pass


def draw_db_by_func(base_dir,n_rows,worksname_paradict,draw_one_func,fig_name,n_cols=6,
                    force_update_json = False):

    work_stats_dict = {}
    array_base_dir = f'/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/{test_prefix}other/numpy-array'
    os.makedirs(array_base_dir, exist_ok=True)
    # npy_array_format = '{}_{}.npy'
    
    n_rows = math.ceil(len(worksname_paradict) / n_cols)

    parameters = {'axes.labelsize': 30,
          'axes.titlesize': 30,
          'xtick.labelsize': 24,
          'ytick.labelsize': 24,
          'legend.fontsize': 30,
          'font.size': 20,
          'axes.facecolor': 'white',
          'figure.facecolor': 'white',
          'savefig.facecolor': 'white',
          }
    plt.rcParams.update(parameters)



    fig,ax = plt.subplots(n_rows,n_cols)
    
    fig.set_size_inches(n_cols*8, 6*n_rows)

    s_2 = re.compile(r'(\w+)-([\w\.]+)')

    for i,work in enumerate(worksname_paradict):
        # work = worksname[i]
        para_dict = worksname_paradict[work]
        # full_ass = max_assoc
        word_dir = os.path.join(base_dir,work)
        if not os.path.isdir(word_dir):
            continue
        work_array_dir = os.path.join(array_base_dir, work, para_dict['length'])
        os.makedirs(work_array_dir, exist_ok=True)
        
        new_base = os.path.join(word_dir,f'l3-{para_dict["policy"]}-{para_dict["length"]}')
        db_path = os.path.join(new_base,'hm.db')
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        
        max_interval_query = 'SELECT MAX(INTERVAL) FROM UCPLookahead;'
        f_max = cur.execute(max_interval_query)
        interval_num = 0
        for mval in f_max:
            interval_num = int(mval) + 1
        cur.close()

        local_hitcnt_array = np.zeros((interval_num,all_set,ncpus,max_assoc),dtype=int)
        local_hitcnt_diff_array = np.zeros((interval_num,all_set,ncpus,max_assoc),dtype=int)
        global_ucp_decision_array = np.zeros((interval_num,ncpus),dtype=int)
        global_hitcnt_array = np.zeros((interval_num,ncpus,max_assoc),dtype=int)
        global_hitcnt_diff_array = np.zeros((interval_num,ncpus,max_assoc),dtype=int)
        
        for tti in range(interval_num):
            #record set local
            cur = con.cursor()
            local_hitcnt_query = f'SELECT SETIDX,HITCNTS FROM UCPLookahead WHERE INTERVAL = {tti} AND SETIDX != {all_set};'
            f = cur.execute(local_hitcnt_query)
            for setidx,idhitcnts in f:
                setidx = int(setidx)
                #record hitcnts of each cpu
                idhitcnts = idhitcnts.split(' ')
                for cpu,hitcnts in enumerate(idhitcnts):
                    hitcnts = hitcnts.split('-')
                    hitcnts = [int(x) for x in hitcnts]
                    local_hitcnt_array[tti,setidx,cpu,:] = hitcnts
                    if (tti == 0):
                        local_hitcnt_diff_array[tti,setidx,cpu,:] = hitcnts
                    else:
                        local_hitcnt_diff_array[tti,setidx,cpu,:] = hitcnts - np.right_shift(local_hitcnt_array[tti-1,setidx,cpu,:],1)
            cur.close()
            # record global
            cur = con.cursor()
            global_hitcnt_query = f'SELECT HITCNTS,ALLOCATIONS FROM UCPLookahead WHERE INTERVAL = {tti} AND SETIDX = {all_set};'
            f = cur.execute(global_hitcnt_query)
            for idhitcnts, allocs in f:
                #record hitcnts of each cpu
                idhitcnts = idhitcnts.split(' ')
                for cpu,hitcnts in enumerate(idhitcnts):
                    hitcnts = hitcnts.split('-')
                    hitcnts = [int(x) for x in hitcnts]
                    global_hitcnt_array[tti,cpu,:] = hitcnts
                    if (tti == 0):
                        global_hitcnt_diff_array[tti,cpu,:] = hitcnts
                    else:
                        global_hitcnt_diff_array[tti,cpu,:] = hitcnts - np.right_shift(global_hitcnt_array[tti-1,cpu,:],1)
                #record allocation decision
                allocs = allocs.split(' ')
                allocs = [int(x) for x in allocs]
                global_ucp_decision_array[tti,:] = allocs

            cur.close()

        con.close()

        #save to npy files
        np.save(os.path.join(work_array_dir,'local_hitcnt_array.npy'),local_hitcnt_array)
        np.save(os.path.join(work_array_dir,'local_hitcnt_diff_array.npy'),local_hitcnt_diff_array)
        np.save(os.path.join(work_array_dir,'global_hitcnt_array.npy'),global_hitcnt_array)
        np.save(os.path.join(work_array_dir,'global_hitcnt_diff_array.npy'),global_hitcnt_diff_array)
        np.save(os.path.join(work_array_dir,'global_ucp_decision_array.npy'),global_ucp_decision_array)

def run_one_conf(select_json:str):
    with open(select_json,'r') as f:
        global use_conf
        use_conf = json.load(f)
    if use_conf is None:
        exit(255)
    global test_prefix
    test_prefix = use_conf['test_prefix']
    cache_type = test_prefix.split('_')[1]
    wm_length = test_prefix.split('_')[2].strip('-')
    base_dir_format = use_conf['base_dir_format']
    logs_base_dir = base_dir_format.rsplit('/',2)[0]
    global ncpus
    ncpus = 4
    newucp_dirname = f'newucp-mix{ncpus}-short-{cache_type}-{wm_length}'
    # base_dir = base_dir_format.format(test_prefix)
    base_dir = os.path.join(logs_base_dir,newucp_dirname)

    pic_dir_path = f'set_analyze/{test_prefix}pics'
    os.makedirs(pic_dir_path, exist_ok=True)
    global worksname
    worksname = use_conf['cache_work_names'] #like mcf
    global all_set
    all_set = use_conf['all_set']
    global max_assoc
    max_assoc = use_conf['max_assoc']

    combs = os.listdir(base_dir)
    print(combs)
    param_dict = {}
    for comb in combs:
        param_dict[comb] = {
            'policy': 'BaseUCPPolicy',
            'length': '5M',
        }

    n_works = len(combs)
    n_cols = 5
    n_rows = math.ceil(n_works/n_cols)
    draw_db_by_func(base_dir,n_rows,param_dict,
                    n_cols=n_cols,
        draw_one_func=draw_one_workload_way_need,fig_name=os.path.join(pic_dir_path,'newucp_ucpsim.png'))


if __name__ == '__main__':
    if opt.json:
        select_json = opt.json
        run_one_conf(select_json)
    else:
        for co in confs:
            select_json = co
            run_one_conf(select_json)