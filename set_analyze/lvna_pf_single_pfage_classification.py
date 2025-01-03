import os
import re
import numpy as np
import utils.common as c
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
from set_analyze.my_diff_color import *

parser = argparse.ArgumentParser(description="options to get set stats")
parser.add_argument('-j','--json', type=str,
    default=None)

opt = parser.parse_args()

confs=[
    "/nfs/home/zhangchuanqi/lvna/5g/lazycat-data_proc/set_analyze/conf-json/conf_lvnapf_50M.json",
]

def draw_one_pfage_classify(ax,s_dicts,workload_name,full,pos:tuple):
    age_min_map = s_dicts['addr_age_min_map']
    age_max_map = s_dicts['addr_age_max_map']
    pfhit_cnt_map = s_dicts['addr_pfhit_cnt_map']
    total_pfhit = s_dicts['total_pfhit']

    #calculate correct rate of classification result
    part_start = 50
    part_end = 2000
    part_interval = 50
    correct_part_res = []
    for part in range(part_start,part_end,part_interval):
        right_part_cnt = 0
        for addr in age_min_map:
            if age_min_map[addr] > part or age_max_map[addr] < part:
                right_part_cnt += pfhit_cnt_map[addr]
        correct_part_res.append(right_part_cnt/total_pfhit)
    
    x = np.arange(part_start,part_end,part_interval)
    ax.plot(x,correct_part_res,label=f'{workload_name}')
    ax.set_title(f'{workload_name}')
    ax.set_xlabel('threshold age cycle')
    ax.set_ylabel('correct rate')
    # formatter = ticker.ScalarFormatter(useMathText=True)
    # formatter.set_scientific(True)
    # formatter.set_powerlimits((-1,1))
    # ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

    # ax.axvline(x=200, color='b', linestyle='--')
    # ax.axvline(x=500, color='b', linestyle='--')
    # ax.axvline(x=1000, color='b', linestyle='--')
    # # ax.axvline(x=2000, color='b', linestyle='--')

    # ax.set_xlim(0,2000)


def draw_db_by_func(base_dir,n_rows,worksname_waydict,draw_one_func,fig_name,n_cols=6,
                    force_update_json = False):

    work_stats_dict = {}
    dict_updated = False

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

    for i,work in enumerate(worksname_waydict):
        work = worksname[i]
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
            new_base = os.path.join(word_dir,f'l3-nopart')
            db_path = os.path.join(new_base,'hm.db')
            con = sqlite3.connect(db_path)
            cur = con.cursor()

            addr_age_map = {}
            addr_age_min_map = {}
            addr_age_max_map = {}
            addr_pfhit_cnt_map = {}

            all_access_query = 'SELECT BLKADDR,AGECYCLE FROM L3BlkFirstHitAge;'
            f = cur.execute(all_access_query)

            for blkaddr,agecycle in f:
                blkaddr = int(blkaddr)
                agecycle = int(agecycle)
                if blkaddr not in addr_age_map:
                    addr_age_map[blkaddr] = []
                    addr_age_min_map[blkaddr] = agecycle
                    addr_age_max_map[blkaddr] = agecycle
                    addr_pfhit_cnt_map[blkaddr] = 0
                addr_age_map[blkaddr].append(agecycle)
                if addr_age_min_map[blkaddr] > agecycle:
                    addr_age_min_map[blkaddr] = agecycle
                if addr_age_max_map[blkaddr] < agecycle:
                    addr_age_max_map[blkaddr] = agecycle
                addr_pfhit_cnt_map[blkaddr] += 1

            cur.close()
            s_dicts['addr_age_map'] = addr_age_map
            s_dicts['addr_age_min_map'] = addr_age_min_map
            s_dicts['addr_age_max_map'] = addr_age_max_map
            s_dicts['addr_pfhit_cnt_map'] = addr_pfhit_cnt_map
            total_pfhit = 0
            for k,v in addr_pfhit_cnt_map.items():
                total_pfhit += v
            s_dicts['total_pfhit'] = total_pfhit
            # work_stats_dict[work] = s_dicts

        draw_one_func(ax_bar,s_dicts,work,max_assoc,(fx,fy))

    for i in range(len(worksname_waydict),n_rows*n_cols):
        fx = i // n_cols
        fy = i % n_cols
        ax[fx,fy].remove()

    # if not dict_updated or force_update_json:
    #     #save to json
    #     if json_path is not None:
    #         jdpath = os.path.dirname(json_path)
    #         os.makedirs(jdpath,exist_ok=True)
    #         with open(json_path,'w') as f:
    #             json.dump(work_stats_dict,f,indent=2)
    
    # legends = [Patch(color=contrasting_orange[4],label=f'95% perf CAT ways'),
    #             Patch(color=contrasting_orange[15],label=f'Wasted cache space'),
    #             ]
    # fig.legend(handles = legends, loc = 'upper left', ncol = 2 )
    # plt.tight_layout(rect=(0, 0, 1, 1))


    plt.tight_layout()
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
    global max_assoc
    max_assoc = use_conf['max_assoc']
    global all_set
    all_set = use_conf['all_set']

    base_dir_format = use_conf['base_dir_format']
    base_dir = base_dir_format.format(test_prefix)
    pic_dir_path = f'set_analyze/{test_prefix}pics'
    os.makedirs(pic_dir_path, exist_ok=True)

    global worksname
    worksname = os.listdir(base_dir)
    worksname.sort()
    pf_works = [
		'GemsFDTD.06',
		'astar.06',
		'bwaves.06',
		'bwaves.17',
		'bzip2.06',
		'cactuBSSN.17',
		'cactusADM.06',
		'cam4.17',
		'cc_sv',
		'dealII.06',
		'fotonik3d.17',
		'gcc.06',
		'gromacs.06',
		'lbm.06',
		'lbm.17',
		'leslie3d.06',
		'libquantum.06',
		'mcf.06',
		'mcf.17',
		'milc.06',
		'moses',
		'nab.17',
		'namd.17',
		'omnetpp.06',
		'omnetpp.17',
		'parest.17',
		'perlbench.17',
		'pr_spmv',
		'roms.17',
		'soplex.06',
		'sphinx',
		'sphinx3.06',
		'sssp',
		'tc',
		'xalancbmk.06',
		'xalancbmk.17',
		'xapian',
		'xz.17',
		'zeusmp.06',
	]
    worksname = pf_works

    interested_works = {}
    for w in worksname:
        interested_works[w] = max_assoc    

    n_works = len(worksname)
    n_cols = 4
    n_rows = math.ceil(n_works/n_cols)
    draw_db_by_func(base_dir,n_rows,interested_works,
                    n_cols=n_cols,
        draw_one_func=draw_one_pfage_classify,fig_name=os.path.join(pic_dir_path,'pfage_class.png'))


if __name__ == '__main__':
    if opt.json:
        select_json = opt.json
        run_one_conf(select_json)
    else:
        for co in confs:
            select_json = co
            run_one_conf(select_json)