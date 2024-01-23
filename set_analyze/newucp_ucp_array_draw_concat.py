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

from PIL import Image

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
    "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_oldincLRU_tailbm250M.json",
    # "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_goldencoveLRU_tailbm250M.json",
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
    fig_base_dir = f'/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/{test_prefix}other/workload-fig'
    os.makedirs(array_base_dir, exist_ok=True)
    os.makedirs(fig_base_dir, exist_ok=True)
    
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

    np.seterr(invalid='ignore')

    group_pngs_dict = {}
    group_sizes = [ 1<< g for g in range(8,13) ]
    for gs in group_sizes:
        group_pngs_dict[gs] = []

    for i,mix_names in enumerate(worksname_paradict):
        # work = worksname[i]
        workload_names = mix_names.split('-')
        para_dict = worksname_paradict[mix_names]
        # full_ass = max_assoc
        work_array_dir = os.path.join(array_base_dir, mix_names, para_dict['length'])
        work_fig_dir = os.path.join(fig_base_dir, mix_names, para_dict['length'])

        os.makedirs(work_array_dir, exist_ok=True)
        os.makedirs(work_fig_dir, exist_ok=True)

        
        for g in group_sizes:
            group_png_path = os.path.join(work_fig_dir,f'underglobal-group-{g}.png')
            group_png = Image.open(group_png_path)
            group_pngs_dict[g].append(group_png)
    
    s_width,s_height = group_pngs_dict[group_sizes[0]][0].size
    total_width = s_width * n_cols
    total_height = s_height * n_rows

    for g in group_sizes:
        new_img = Image.new('RGB', (total_width, total_height))
        current_row = 0
        current_col = 0
        for img in group_pngs_dict[g]:
            new_img.paste(img, (current_col* s_width, current_row * s_height))
            current_col += 1
            if current_col == n_cols:
                current_row += 1
                current_col = 0
        new_img.save(fig_name.format(g))



        


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
            'length': '10M',
        }

    n_works = len(combs)
    n_cols = 4
    n_rows = math.ceil(n_works/n_cols)
    draw_db_by_func(base_dir,n_rows,param_dict,
                    n_cols=n_cols,
        draw_one_func=draw_one_workload_way_need,fig_name=os.path.join(pic_dir_path,'newucp_array_underglobal_concat_{}.png'))


if __name__ == '__main__':
    if opt.json:
        select_json = opt.json
        run_one_conf(select_json)
    else:
        for co in confs:
            select_json = co
            run_one_conf(select_json)