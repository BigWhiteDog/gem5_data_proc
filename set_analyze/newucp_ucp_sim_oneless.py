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
    each_way_cnts = s_dicts['each_pos_data']
    x_val = np.arange(max_assoc)

    extra0_list_color = contrasting_orange[15]
    parts = ax.violinplot(each_way_cnts,x_val, widths = 1, points=256, showmeans=True,showextrema=True)

    
    if pos[1] == 0:
        ax.set_ylabel('hit cnts for each set')
    # ax.set_ylim(0, max_assoc)
    ax.set_xlim(-1, max_assoc)
    # ax.xaxis.grid(False)
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xticks(x_val)
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    # if pos[0] == 1:
    ax.set_xlabel('LRU hit pos')
    ax.set_title(f'{workload_name}')
    # if pos == (0,0):
    #     ax.legend(shadow=0, bbox_to_anchor=(-0.01,1.6), loc = 'upper left',  \
    #         borderaxespad=0.2, ncol = 1, columnspacing=0.5, labelspacing=0.1)
    #     # ax.legend(shadow=0, fontsize = 12, bbox_to_anchor=(-0.01,1.3,0,0), loc = 'upper left',  \
    #     #     borderaxespad=0.2, ncol = 10, columnspacing=0.5, labelspacing=0.1)


def draw_db_by_func(base_dir,n_rows,worksname_waydict,draw_one_func,fig_name,n_cols=6,
                    force_update_json = False):

    array_base_dir = f'/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/{test_prefix}other/numpy-array'
    os.makedirs(array_base_dir, exist_ok=True)
    
    n_rows = math.ceil(len(worksname_waydict) / n_cols)

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

    for i,work in enumerate(worksname_waydict):
        work_array_dir = os.path.join(array_base_dir, work, para_dict['length'])
        if not os.path.isdir(work_array_dir):
            exit(255)
        local_hitcnt_array = np.load(os.path.join(work_array_dir,'local_hitcnt_array.npy'))
        local_hitcnt_diff_array = np.load(os.path.join(work_array_dir,'local_hitcnt_diff_array.npy'))
        global_ucp_decision_array = np.load(os.path.join(work_array_dir,'global_ucp_decision_array.npy'))
        global_hitcnt_array = np.load(os.path.join(work_array_dir,'global_hitcnt_array.npy'))
        global_hitcnt_diff_array = np.load(os.path.join(work_array_dir,'global_hitcnt_diff_array.npy'))

        num_interval = local_hitcnt_array.shape[0]

        fy = i % n_cols
        fx = i // n_cols
        ax_bar = ax[fx,fy]
        s_dicts = {}

        s_dicts['each_tti_totalhit'] = list()
        s_dicts['each_tti_oneless_totalhit'] = list()

        all_tti_total_hit = 0
        all_tti_oneless_total_hit = 0

        for tti in range(num_interval):
            decisions = global_ucp_decision_array[tti]
            in_tti_hit_each = global_hitcnt_diff_array[tti]
            total_hit = 0
            minimal_last_alloc = 9999999999999999
            for core, deci in enumerate(decisions):
                total_hit += np.sum(in_tti_hit_each[core][:deci])
                if in_tti_hit_each[core][deci-1] < minimal_last_alloc:
                    minimal_last_alloc = in_tti_hit_each[core][deci-1]
            s_dicts['each_tti_totalhit'].append(total_hit)
            all_tti_total_hit += total_hit
            s_dicts['each_tti_oneless_totalhit'].append(total_hit-minimal_last_alloc)
            all_tti_oneless_total_hit += total_hit-minimal_last_alloc
        

                


                

        # draw_one_func(ax_bar,s_dicts,work,max_assoc,(fx,fy))

    # for i in range(len(worksname_waydict),n_rows*n_cols):
    #     fx = i // n_cols
    #     fy = i % n_cols
    #     ax[fx,fy].remove()

    
    # # legends = [Patch(color=contrasting_orange[4],label=f'95% perf CAT ways'),
    # #             Patch(color=contrasting_orange[15],label=f'Wasted cache space'),
    # #             ]
    # # fig.legend(handles = legends, loc = 'upper left', ncol = 2 )

    # # plt.tight_layout(rect=(0, 0, 1, 1))
    # plt.tight_layout()
    # plt.savefig(fig_name,dpi=300)
    # plt.clf()


def run_one_conf(select_json:str):
    with open(select_json,'r') as f:
        global use_conf
        use_conf = json.load(f)
    if use_conf is None:
        exit(255)
    global test_prefix
    test_prefix = use_conf['test_prefix']
    cache_type = test_prefix.split('-')[1]
    wm_length = test_prefix.split('-')[2].strip('-')
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