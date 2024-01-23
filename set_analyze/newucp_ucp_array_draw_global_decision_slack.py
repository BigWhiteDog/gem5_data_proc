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

    for i,mix_names in enumerate(worksname_paradict):
        # work = worksname[i]
        workload_names = mix_names.split('-')
        para_dict = worksname_paradict[mix_names]
        # full_ass = max_assoc
        work_array_dir = os.path.join(array_base_dir, mix_names, para_dict['length'])
        work_fig_dir = os.path.join(fig_base_dir, mix_names, para_dict['length'])

        os.makedirs(work_array_dir, exist_ok=True)
        os.makedirs(work_fig_dir, exist_ok=True)

        local_hitcnt_array = np.load(os.path.join(work_array_dir,'local_hitcnt_array.npy'))
        local_hitcnt_diff_array = np.load(os.path.join(work_array_dir,'local_hitcnt_diff_array.npy'))
        global_ucp_decision_array = np.load(os.path.join(work_array_dir,'global_ucp_decision_array.npy'))
        # global_hitcnt_array = np.load(os.path.join(work_array_dir,'global_hitcnt_array.npy'))
        # global_hitcnt_diff_array = np.load(os.path.join(work_array_dir,'global_hitcnt_diff_array.npy'))

        interval_num = local_hitcnt_diff_array.shape[0]
        
        # local_data_select = local_hitcnt_diff_array
        local_data_select = local_hitcnt_array
        local_hitsum = np.zeros((interval_num,all_set,ncpus),dtype=int)
        local_1less_delta_hit = np.zeros((interval_num,all_set,ncpus),dtype=int)
        local_1more_delta_hit = np.zeros((interval_num,all_set,ncpus),dtype=int)
        for inter in range(interval_num):
            cpu_decisions = global_ucp_decision_array[inter]
            for s in range(all_set):
                for c in range(ncpus):
                    local_hitsum[inter,s,c] = local_data_select[inter,s,c,0:cpu_decisions[c]].sum()
                    local_1less_delta_hit[inter,s,c] = local_data_select[inter,s,c,cpu_decisions[c]-1]
                    local_1more_delta_hit[inter,s,c] = local_data_select[inter,s,c,cpu_decisions[c]]


        # local_full_hitcnt_array = local_hitcnt_array.sum(axis=3)
        # local_full_hitcnt_array = local_hitcnt_diff_array.sum(axis=3)

        group_sizes = [ 1<< g for g in range(8,13) ]
        for g in group_sizes:
            tmp_group_array = local_hitsum.reshape(interval_num,-1,g,ncpus)
            hit_group_array = tmp_group_array.sum(axis=2)
            cpu_hitsum_array = local_hitsum.sum(axis=1)
            # hitsum_group_array = hit_group_array.sum(axis=2)
            # hitsum_mean_group_array = hitsum_group_array.mean(axis=1)

            #calculate 1less group and sort
            tmp_group_array = local_1less_delta_hit.reshape(interval_num,-1,g,ncpus)
            hit_group_array_1less = tmp_group_array.sum(axis=2)
            sorted_hit_group_array_1less = np.sort(hit_group_array_1less,axis=1)
            # sorted_hit_group_array_1less = np.flip(sorted_hit_group_array_1less,axis=1)

            #calculate 1more group and sort
            tmp_group_array = local_1more_delta_hit.reshape(interval_num,-1,g,ncpus)
            hit_group_array_1more = tmp_group_array.sum(axis=2)
            sorted_hit_group_array_1more = np.sort(hit_group_array_1more,axis=1)
            # sorted_hit_group_array_1more = np.flip(sorted_hit_group_array_1more,axis=1)


            ngroups = all_set // g
            fig = plt.figure()
            nrow = 5
            fig.set_size_inches(8*ncpus, 6*nrow)
            ax = []
            for c in range(ncpus):
                cpu_hit_array = hit_group_array[:,:,c]
                ax = fig.add_subplot(nrow,ncpus,c+1, projection='3d')

                #draw
                _x = np.arange(ngroups)
                _y = np.arange(interval_num)
                _xx, _yy = np.meshgrid(_x, _y)
                x, y = _xx.ravel(), _yy.ravel()
                bottom = np.zeros_like(x)
                top = cpu_hit_array.ravel()
                width = depth = 1
                ax.bar3d(x, y, bottom, width, depth, top, shade=True)
                
                ax.set_xlabel('group', labelpad=20)
                ax.set_ylabel('interval', labelpad=20)
                ax.set_zlabel('hitcnt', labelpad=20)
                ax.set_title(f'{workload_names[c]} hitcnt')

                #draw 1less hit group
                irow = 1
                cpu_1less_hit_array = sorted_hit_group_array_1less[:,:,c]
                ax = fig.add_subplot(nrow,ncpus,c+1+ncpus*irow, projection='3d')
                top = cpu_1less_hit_array.ravel()
                ax.bar3d(x, y, bottom, width, depth, top, color = contrasting_orange[5+irow], shade=True)
                ax.set_xlabel('group (sorted)', labelpad=20)
                ax.set_ylabel('interval', labelpad=20)
                ax.set_zlabel('1less way hitcnt', labelpad=20)
                ax.set_title(f'{workload_names[c]} 1less hitcnt')

                #draw 1less hit group relative to cpu hitsum
                irow = 2
                this_cpu_hitsum_array = cpu_hitsum_array[:,c]
                #tile it for each group in each interval
                this_cpu_hitsum_array = np.tile(this_cpu_hitsum_array,(ngroups,1)).T
                relative_cpu_1less_hit_array = cpu_1less_hit_array / this_cpu_hitsum_array
                ax = fig.add_subplot(nrow,ncpus,c+1+ncpus*irow, projection='3d')
                top = relative_cpu_1less_hit_array.ravel()
                ax.bar3d(x, y, bottom, width, depth, top, color = contrasting_orange[5+irow], shade=True)
                ax.set_xlabel('group (sorted)', labelpad=20)
                ax.set_ylabel('interval', labelpad=20)
                ax.set_zlabel('1less way hitcnt rel', labelpad=20)
                ax.set_title(f'{workload_names[c]} 1less hitcnt relative to cpu hitsum')

                #draw 1less hit group relative to cpu hitsum (accumulated)
                irow = 3
                acc_relative_cpu_1less_hit_array = np.cumsum(relative_cpu_1less_hit_array,axis=1)
                ax = fig.add_subplot(nrow,ncpus,c+1+ncpus*irow, projection='3d')
                top = acc_relative_cpu_1less_hit_array.ravel()
                ax.bar3d(x, y, bottom, width, depth, top, color = contrasting_orange[5+irow], shade=True)
                ax.set_xlabel('group (sorted)', labelpad=20)
                ax.set_ylabel('interval', labelpad=20)
                ax.set_zlabel('1less way hitcnt', labelpad=20)
                ax.set_title(f'{workload_names[c]} 1less hitcnt relative to cpu hitsum (accumulated)')

                #draw 1more hit group
                irow = 4
                cpu_1more_hit_array = sorted_hit_group_array_1more[:,:,c]
                ax = fig.add_subplot(nrow,ncpus,c+1+ncpus*irow, projection='3d')
                top = cpu_1more_hit_array.ravel()
                ax.bar3d(x, y, bottom, width, depth, top, color = contrasting_orange[5+irow], shade=True)
                ax.set_xlabel('group (sorted)', labelpad=20)
                ax.set_ylabel('interval', labelpad=20)
                ax.set_zlabel('1more way hitcnt', labelpad=20)
                ax.set_title(f'{workload_names[c]} 1more hitcnt')

                #draw 1more hit group relative to cpu hitsum
                irow = 5
                relative_cpu_1more_hit_array = cpu_1more_hit_array / this_cpu_hitsum_array
                ax = fig.add_subplot(nrow,ncpus,c+1+ncpus*irow, projection='3d')
                top = relative_cpu_1more_hit_array.ravel()
                ax.bar3d(x, y, bottom, width, depth, top, color = contrasting_orange[5+irow], shade=True)
                ax.set_xlabel('group (sorted)', labelpad=20)
                ax.set_ylabel('interval', labelpad=20)
                ax.set_zlabel('1more way hitcnt', labelpad=20)
                ax.set_title(f'{workload_names[c]} 1more hitcnt relative to cpu hitsum')         
              


            fig.tight_layout()
            fig.set_edgecolor('black')
            fig.set_linewidth(5)
            fig.savefig(os.path.join(work_fig_dir,f'underglobal-group-1way-slack-{g}.png'),dpi=300)
            plt.close(fig)



        


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