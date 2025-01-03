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
    # re.compile(r'UCP'),
    # re.compile(r'FairPrefUCPPolicy-2M'),
    # re.compile(r'PFAge(\d+)Pass(\d+)Sign(\d+)ScRP'),
    # re.compile(r'PAPLvLRURP'),
    # re.compile(r'DSS(\d*)PAPLvLRURP'),
    # re.compile(r'DSS(.*)UsefulPfPAPLvLRURP'),
    # re.compile(r'DSS256(.*)UsefulPfWbPAPLvLRURP'),
    # re.compile(r'DSS256(.*)StrictShctUpPAPLvLRURP'),
    # re.compile(r'DSS256(.*)PUPLvLRURP'),
    # re.compile(r'DSS256(.*)StrictShctUpPAPLvLRURP'),
    # re.compile(r'DSS256StrictShctUpPAPLvLRURP'),
    re.compile(r'DSS256UsefulPfWbPAPLvLRURP'),
    # re.compile(r'DSS256UnstrictPUPLvLRURP'),
    # re.compile(r'DSS256PUAPLvLRURP'),
    # re.compile(r'DSSPUAPLvLRURP'),
    # re.compile(r'NDSSPSerRUAPLvLRURP'),
    # re.compile(r'NDSS256PSerRUAPLvLRURP'),
    # re.compile(r'NDSSPSerScRUAPLvLRURP'),
    # re.compile(r'NDSS256PSerScRUAPLvLRURP'),
    # re.compile(r'NDSS256[AP]hPSerScRUAPLvLRURP'),
    # re.compile(r'NDSS256[AP]hPSerSc1RUAPLvLRURP'),
    re.compile(r'NDSS256AhP.*RUAPLvLRURP'),
    # re.compile(r'NDSSPSerSc1RUAPLvLRURP'),
    # re.compile(r'NDSS256PSerSc1RUAPLvLRURP'),
    # re.compile(r'DSS256PRUAP\dW\dLvLRURP'),
    # re.compile(r'DSS256PUPLvLRURP'),
    # re.compile(r'DSS([a-zA-Z]*)MayUsefulPfPAPLvLRURP'),
    # re.compile(r'DSS256PAPLvLRURP'),
]
xlsx_drop_res = [
    # re.compile(r'nopart'),
]

fig_name = f'pfagepass_multiway_ndss_AhP*RUAP.png'

def draw_one_pfage_classify(ax,s_dicts,workload_name,full,pos:tuple):
    policies = list(s_dicts.keys())
    policies.sort()
    x_labels = list(range(1,max_assoc+1))
    base_ipc = s_dicts['pf']
    for c,policy in enumerate(policies):
        # print(f'{policy}')
        if policy == 'pf':
            continue
        speedup =  s_dicts[policy] / base_ipc
        ax.plot(x_labels,speedup,label=f'{policy[:-2]}',color=contrasting_orange[(c+3)%len(contrasting_orange)], marker='o')

    ax.set_title(f'{workload_name}')
    ax.set_ylabel('Norm IPC')
    # ax.xaxis.set_major_formatter(formatter)
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylim(0.9,1.1)
    ax.set_xlabel('number of L3 ways (1MB/way)')
    ax.set_xlim(0,max_assoc+1)
    ax.set_xticks(x_labels)
    # ax.legend()


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

    policy_re = re.compile(r'l3-(\d+)-(\w+)')
    force_update = False
    last_nsamples=1
    ncore = 1

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
            partsname = os.listdir(word_dir) #like l3-1
            for part in partsname:
                if not os.path.isdir(os.path.join(word_dir,part)):
                    continue
                m = policy_re.match(part)
                if m is None:
                    continue
                ways = int(m.group(1))
                policy = m.group(2)
                if not any([r.match(policy) for r in interested_res]):
                    continue
                new_base = os.path.join(word_dir,part)
                if force_update:
                    one_dict = extract_newgem_raw_json(new_base,ncores=ncore,last_nsamples=last_nsamples)
                elif not os.path.exists(os.path.join(new_base,f'{last_nsamples}period.json')):
                    one_dict = extract_newgem_raw_json(new_base,ncores=ncore,last_nsamples=last_nsamples)
                else:
                    with open(os.path.join(new_base,f'{last_nsamples}period.json'),'r') as f:
                        one_dict = json.load(f)
                if policy not in s_dicts:
                    s_dicts[policy] = np.zeros(max_assoc)
                s_dicts[policy][ways-1] = one_dict['cpu.ipc'][0]
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
    legends = []
    policies = list(s_dicts.keys())
    policies.sort()
    for c,policy in enumerate(policies):
        # print(f'{policy}')
        if policy == 'pf':
            continue
        legends.append(Patch(color=contrasting_orange[(c+3)%len(contrasting_orange)],label=f'{policy[:-2]}'))
    fig.legend(handles = legends, loc = 'lower right', ncol = 1,borderaxespad=0, labelspacing=0, handlelength=0.5)

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
    base_dir += '-new'
    
    pic_dir_path = f'set_analyze/{test_prefix}pics'
    os.makedirs(pic_dir_path, exist_ok=True)

    global worksname
    worksname = os.listdir(base_dir)
    worksname.sort()
    # pf_works = [
	# 	'GemsFDTD.06',
	# 	'astar.06',
	# 	'bwaves.06',
	# 	'bwaves.17',
	# 	'bzip2.06',
	# 	'cactuBSSN.17',
	# 	'cactusADM.06',
	# 	'cam4.17',
	# 	'cc_sv',
	# 	'dealII.06',
	# 	'fotonik3d.17',
	# 	'gcc.06',
	# 	'gromacs.06',
	# 	'lbm.06',
	# 	'lbm.17',
	# 	'leslie3d.06',
	# 	'libquantum.06',
	# 	'mcf.06',
	# 	'mcf.17',
	# 	'milc.06',
	# 	'moses',
	# 	'nab.17',
	# 	'namd.17',
	# 	'omnetpp.06',
	# 	'omnetpp.17',
	# 	'parest.17',
	# 	'perlbench.17',
	# 	'pr_spmv',
	# 	'roms.17',
	# 	'soplex.06',
	# 	'sphinx',
	# 	'sphinx3.06',
	# 	'sssp',
	# 	'tc',
	# 	'xalancbmk.06',
	# 	'xalancbmk.17',
	# 	'xapian',
	# 	'xz.17',
	# 	'zeusmp.06',
	# ]
    # worksname = pf_works

    interested_works = {}
    for w in worksname:
        interested_works[w] = max_assoc    

    n_works = len(worksname)
    n_cols = 4
    n_rows = math.ceil(n_works/n_cols)
    draw_db_by_func(base_dir,n_rows,interested_works,
                    n_cols=n_cols,
        draw_one_func=draw_one_pfage_classify,fig_name=os.path.join(pic_dir_path,fig_name))


if __name__ == '__main__':
    if opt.json:
        select_json = opt.json
        run_one_conf(select_json)
    else:
        for co in confs:
            select_json = co
            run_one_conf(select_json)