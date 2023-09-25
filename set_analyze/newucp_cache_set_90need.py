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
    "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_goldencoveLRU_tailbm50M.json",
    "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_goldencoveLRU_tailbm250M.json",
]

# from cache_sensitive_names import *
from set_analyze.my_diff_color import *

def draw_one_workload_way_need(ax,s_dicts,workload_name,full_ass,pos:tuple):
    label_s = ['min_ways_no_extra_miss','min_ways_1_extra_miss', 'min_ways_2_extra_miss']
    extra0_list = s_dicts[label_s[0]]
    s_extra0_list = sorted(extra0_list)
    x_val = np.arange(all_set)
    full_ass_vals = np.full(all_set,full_ass)

    extra0_list_color = contrasting_orange[15]
    alpha_set = 1.0
    ax.plot(s_extra0_list, label='wasted cache space', color = extra0_list_color,linewidth=2)
    ax.fill_between(x_val, full_ass_vals, s_extra0_list, color = extra0_list_color, alpha=alpha_set)
    ax.plot(full_ass_vals,label='95% perf CAT set ways', color = contrasting_orange[4],linewidth=5)
    if pos[1] == 0:
        ax.set_ylabel('Needed ways')
    ax.set_ylim(0, max_assoc)
    ax.set_xlim(0,all_set)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    if pos[0] == 1:
        ax.set_xlabel('Sorted set index')
    ax.set_title(f'{workload_name}')
    # if pos == (0,0):
    #     ax.legend(shadow=0, bbox_to_anchor=(-0.01,1.6), loc = 'upper left',  \
    #         borderaxespad=0.2, ncol = 1, columnspacing=0.5, labelspacing=0.1)
    #     # ax.legend(shadow=0, fontsize = 12, bbox_to_anchor=(-0.01,1.3,0,0), loc = 'upper left',  \
    #     #     borderaxespad=0.2, ncol = 10, columnspacing=0.5, labelspacing=0.1)


def draw_db_by_func(base_dir,n_rows,worksname_waydict,draw_one_func,fig_name,n_cols=6,
                    force_update_json = False):

    work_stats_dict = {}
    json_format = '/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/{}other/json/asplos23_cache_set_95need.json'
    json_path = json_format.format(test_prefix)
    
    #options of dict and json
    # force_update_json = True

    dict_updated = False

    #try load from json
    if json_path is not None and os.path.isfile(json_path) and not force_update_json:
        with open(json_path,'r') as f:
            json_dict = json.load(f)
            if len(json_dict) > 0:
                #it has data
                work_stats_dict.update(json_dict)
                dict_updated = True


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



    fig,ax = plt.subplots(n_rows,n_cols, sharex=True, sharey=True)
    
    fig.set_size_inches(n_cols*6, 5*n_rows)

    s_2 = re.compile(r'(\w+)-([\w\.]+)')

    for i,work in enumerate(worksname_waydict):
        work = worksname[i]
        full_ass = worksname_waydict[work]
        word_dir = os.path.join(base_dir,work)
        if not os.path.isdir(word_dir):
            continue
        fy = i % n_cols
        fx = i // n_cols
        ax_bar = ax[fx,fy]
        if dict_updated:
            s_dicts = work_stats_dict[work]
        else:
            s_dicts = {}
            s_dicts['unique_blocks_number'] = [0 for _ in range(all_set)]
            s_dicts['unique_reused_blocks_number'] = [0 for _ in range(all_set)]
            s_dicts['ways_miss_cnt'] = {}
            s_dicts['ways_miss_rate'] = {}

            partsname = os.listdir(word_dir) #like l3-1
            for part in partsname:
                if not os.path.isdir(os.path.join(word_dir,part)):
                    continue
                res = s_2.search(part)
                if not res:
                    continue
                if res.group(1) != 'l3':
                    continue
                ways = int(res.group(2))
                if ways > full_ass:
                    continue

                new_base = os.path.join(word_dir,part)
                db_path = os.path.join(new_base,'hm.db')
                all_access_query = 'SELECT SETIDX,sum(ISMISS),count(*) FROM HitMissTrace group by SETIDX;'
                con = sqlite3.connect(db_path)
                cur = con.cursor()
                f = cur.execute(all_access_query)

                s_dicts['ways_miss_cnt'][ways] = [0 for _ in range(all_set)]
                for setidx,msc,allaccess in f:
                    idx = int(setidx)
                    msc = int(msc)
                    allaccess = int(allaccess)
                    s_dicts['ways_miss_cnt'][ways][idx] = msc

                cur.close()
            s_dicts['min_ways_no_extra_miss'] = [full_ass for _ in range(all_set)]
            fullass_miss_cnt = s_dicts['ways_miss_cnt'][full_ass]

            for ways in s_dicts['ways_miss_cnt']:
                my_way_miss_cnt = s_dicts['ways_miss_cnt'][ways]
                for idx in range(all_set):
                    delta_miss = my_way_miss_cnt[idx] - fullass_miss_cnt[idx]
                    if delta_miss <= 0:
                        s_dicts['min_ways_no_extra_miss'][idx] = min(s_dicts['min_ways_no_extra_miss'][idx],ways)
            #update workdict
            work_stats_dict[work] = s_dicts

        draw_one_func(ax_bar,s_dicts,work,full_ass,(fx,fy))

    for i in range(len(worksname_waydict),n_rows*n_cols):
        fx = i // n_cols
        fy = i % n_cols
        ax[fx,fy].remove()

    if not dict_updated or force_update_json:
        #save to json
        if json_path is not None:
            jdpath = os.path.dirname(json_path)
            os.makedirs(jdpath,exist_ok=True)
            with open(json_path,'w') as f:
                json.dump(work_stats_dict,f,indent=2)
    
    legends = [Patch(color=contrasting_orange[4],label=f'95% perf CAT ways'),
                Patch(color=contrasting_orange[15],label=f'Wasted cache space'),
                ]
    fig.legend(handles = legends, loc = 'upper left', ncol = 2 )

    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(fig_name,dpi=300)
    plt.clf()


def run_one_conf(select_json:str):
    with open(select_json,'r') as f:
        global use_conf
        use_conf = json.load(f)
    if use_conf is None:
        exit(255)
    global test_prefix
    test_prefix = use_conf['test_prefix']
    base_dir_format = use_conf['base_dir_format']
    base_dir = base_dir_format.format(test_prefix)
    pic_dir_path = f'set_analyze/{test_prefix}pics'
    os.makedirs(pic_dir_path, exist_ok=True)
    global worksname
    worksname = use_conf['cache_work_names'] #like mcf
    cache_work_90perfways = use_conf['cache_work_90perfways']
    cache_work_95perfways = use_conf['cache_work_95perfways']
    cache_work_95perf_maxfull_ways = use_conf['cache_work_full95perfways']
    # cache_work_fullways = use_conf['cache_work_fullways']
    global all_set
    all_set = use_conf['all_set']
    global max_assoc
    max_assoc = use_conf['max_assoc']

    print(worksname)
    # worksname.remove('lbm')
    # cache_work_95perf_maxfull_ways.pop('lbm')

    n_works = len(worksname)
    n_cols = 6
    n_rows = math.ceil(n_works/n_cols)
    # draw_db_by_func(base_dir,n_rows,cache_work_90perfways,
    #     draw_one_func=draw_one_workload_way_need,fig_name= os.path.join(pic_dir_path,'way_need_90perf_dis.png'))
    # draw_db_by_func(base_dir,n_rows,cache_work_95perfways,
    #     draw_one_func=draw_one_workload_way_need,fig_name=os.path.join(pic_dir_path,'way_need_95perf_dis.png'))
    # draw_db_by_func(base_dir,n_rows,cache_work_fullways,
    #     draw_one_func=draw_one_workload_way_need,fig_name=os.path.join(pic_dir_path,'way_need_dis.png'))
    draw_db_by_func(base_dir,n_rows,cache_work_95perf_maxfull_ways,
                    n_cols=n_cols,
        draw_one_func=draw_one_workload_way_need,fig_name=os.path.join(pic_dir_path,'newucp_way_need_95_dis.png'))


if __name__ == '__main__':
    # base_dir = '/nfs/home/zhangchuanqi/lvna/for_xs/catlog/single-profiling/'
    if opt.json:
        select_json = opt.json
        run_one_conf(select_json)
    else:
        for co in confs:
            select_json = co
            run_one_conf(select_json)