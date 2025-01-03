import os
import re
import numpy as np
import utils.common as c
import utils.target_stats as t
from utils.common import extract_newgem_raw_json
import numpy as np
import argparse
import math

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
    # re.compile(r'PageUCPPolicy'),
]
interested_res = [
    re.compile(r'pf'),
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
    pass

def draw_ipc_access_plot(ax,s_dicts,workload_name,full,pos:tuple):
    pass

def draw_db_by_func(base_dir,n_rows,worksname_waydict,draw_one_func,fig_name,n_cols=6,
                    force_update_json = False):

    force_update = False
    # force_update = True
    last_nsamples=1
    ncore = 1

    use_conf['work_ways_pfsrc_property'] = {}
    for wo in worksname:
        use_conf['work_ways_pfsrc_property'][wo] = {}
        for tw in range(1,max_assoc+1):
            use_conf['work_ways_pfsrc_property'][wo][tw] = {}
            for ps in range(pf_src_num):
                use_conf['work_ways_pfsrc_property'][wo][tw][ps] = {}

    for i,work in enumerate(worksname_waydict):
        work = worksname[i]
        word_dir = os.path.join(base_dir,work)
        if not os.path.isdir(word_dir):
            continue

        ways_pfsrc_property = use_conf['work_ways_pfsrc_property'][work]
        for test_way in range(1, max_assoc+1):
            # s_dicts = {}
            l3_hitsPFSrc = np.zeros(pf_src_num)
            l3_missesPFSrc = np.zeros(pf_src_num)

            nomarl_base = os.path.join(word_dir,f'l3-{test_way}-pf')
            one_dict = extract_newgem_raw_json(nomarl_base,ncores=ncore,last_nsamples=last_nsamples)
            # if force_update:
            #     one_dict = extract_newgem_raw_json(nomarl_base,ncores=ncore,last_nsamples=last_nsamples)
            # elif not os.path.exists(os.path.join(nomarl_base,f'1period.json')):
            #     one_dict = extract_newgem_raw_json(nomarl_base,ncores=ncore,last_nsamples=last_nsamples)
            # with open(os.path.join(nomarl_base,f'1period.json'),'r') as f:
            #     one_dict = json.load(f)

            for ps in range(pf_src_num):
                l3_hitsPFSrc[ps] = one_dict[f'l3.hitsPFSrc_{ps}'][0]
                l3_missesPFSrc[ps] = one_dict[f'l3.missesPFSrc_{ps}'][0]
            l3_accessPFSrc = l3_hitsPFSrc + l3_missesPFSrc
            normal_ipc = one_dict['cpu.ipc'][0]
            l3Access_all = np.sum(l3_accessPFSrc)

            bypass_name_list = [f'l3-{test_way}-bypassPfSrc{sr}' for sr in range(pf_src_num)]
            bypassPfSrcIpc = np.zeros(pf_src_num)
            for ii,bypass_dir_name in enumerate(bypass_name_list):
                psrc = ii
                bypass_base = os.path.join(word_dir,bypass_dir_name)
                if force_update:
                    one_dict = extract_newgem_raw_json(bypass_base,ncores=ncore,last_nsamples=last_nsamples)
                elif not os.path.exists(os.path.join(bypass_base,f'1period.json')):
                    one_dict = extract_newgem_raw_json(bypass_base,ncores=ncore,last_nsamples=last_nsamples)
                else:
                    with open(os.path.join(bypass_base,f'1period.json'),'r') as f:
                        one_dict = json.load(f)
                bypassPfSrcIpc[psrc] = one_dict['cpu.ipc'][0]
            
            bypassPfSrcNipc = bypassPfSrcIpc/normal_ipc
            pfSrcAccessRate = l3_accessPFSrc/l3Access_all
            pfsrc_property = ways_pfsrc_property[test_way]
            for pfs in range(pf_src_num):
                pfsrc_property[pfs]['l3AccessRate'] = pfSrcAccessRate[pfs]
                pfsrc_property[pfs]['nipc'] = bypassPfSrcNipc[pfs]


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
    worksname = cs_works + cs_ps_works + ps_works
    # worksname.sort()

    interested_works = {}
    for w in worksname:
        interested_works[w] = max_assoc    

    n_works = len(worksname)
    n_cols = 4
    n_rows = math.ceil(n_works/n_cols)
    draw_db_by_func(base_dir,n_rows,interested_works,
                    n_cols=n_cols,
        draw_one_func=draw_ipc_access_plot,fig_name='none')

    with open(select_json,'w') as f:
        json.dump(use_conf,f,indent=4)


if __name__ == '__main__':
    if opt.json:
        select_json = opt.json
        run_one_conf(select_json)
    else:
        for co in confs:
            select_json = co
            run_one_conf(select_json)