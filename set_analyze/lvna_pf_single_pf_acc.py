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
    # re.compile(r'NDSS256[AP]hPSerScRUAPLvLRURP'),
    # re.compile(r'NDSS256[AP]hPSerSc1RUAPLvLRURP'),
    # re.compile(r'DSS256PRUAP\dW\dLvLRURP'),
]
xlsx_drop_res = [
    # re.compile(r'nopart'),
]

pf_src_num = 13

def workname_postfix(workname):
    if workname in cs_works:
        return 'cs'
    elif workname in ps_works:
        return 'ps'
    elif workname in cs_ps_works:
        return 'cs-ps'
    return ''

def draw_access_issue_bar(ax,s_dicts,workload_name,full,pos:tuple):
    dcache_pfAcc = s_dicts['dcache.pfUsefulSrc'] / s_dicts['dcache.pfIssuedSrc']
    #calculate sum for all pf src
    pfIssued_all = np.sum(s_dicts['dcache.pfIssuedSrc'])
    pfUseful_all = np.sum(s_dicts['dcache.pfUsefulSrc'])
    pfAcc_all = pfUseful_all / pfIssued_all
    l3Access_all = np.sum(s_dicts['l3.accessPFSrc'])
    #draw four bars
    #issued bar first
    x = np.arange(pf_src_num)
    width = 0.25
    ax.bar(x - 1.5*width, s_dicts['dcache.pfIssuedSrc'], width, label='pfIssued', color=contrasting_orange[4])
    ax.bar(x - 0.5*width, s_dicts['dcache.pfUsefulSrc'], width, label='pfUseful', color=contrasting_orange[5])
    ax.bar(x + 0.5*width, s_dicts['l3.accessPFSrc'], width, label='l3Access', color=contrasting_orange[6])
    # percentage bar for pfAcc
    ax2 = ax.twinx()
    ax2.bar(x + 1.5*width, dcache_pfAcc, width, label='l1dPfAcc', color=contrasting_orange[7])

    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}' for i in range(pf_src_num)])
    ax.set_xlabel('pf src')
    ax.set_ylabel('count')
    ax2.set_ylabel('pfAcc')
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax2.set_ylim(0,1)
    # draw total bar next to the last pf src
    ax.bar(pf_src_num + 0.5*width, pfIssued_all, width, label='pfIssued', color=contrasting_orange[4])
    ax.bar(pf_src_num + 1.5*width, pfUseful_all, width, label='pfUseful', color=contrasting_orange[5])
    ax.bar(pf_src_num + 2.5*width, l3Access_all, width, label='l3Access', color=contrasting_orange[6])
    ax2.bar(pf_src_num + 3.5*width, pfAcc_all, width, label='pfAcc', color=contrasting_orange[7])

    ax.set_title(f'{workload_name}({workname_postfix(workload_name)})')
    # ax.legend()
    # ax2.legend()
    # ax.grid(axis='y')
    # ax2.grid(False)

    labels = [
        'pfIssued',
        'pfUseful',
        'l3Access',
        'l1dPfAcc',
    ]
    return labels


def draw_hitrate_acc_bar(ax,s_dicts,workload_name,full,pos:tuple):
    dcache_pfAcc = s_dicts['dcache.pfUsefulSrc'] / s_dicts['dcache.pfIssuedSrc']
    #calculate sum for all pf src
    pfIssued_all = np.sum(s_dicts['dcache.pfIssuedSrc'])
    pfUseful_all = np.sum(s_dicts['dcache.pfUsefulSrc'])
    pfAcc_all = pfUseful_all / pfIssued_all
    l3Access_all = np.sum(s_dicts['l3.accessPFSrc'])
    l3Hits_all = np.sum(s_dicts['l3.hitsPFSrc'])
    l3HitRate_all = l3Hits_all / l3Access_all
    #draw four bars
    #Access bar first
    x = np.arange(pf_src_num)
    width = 0.25
    ax.bar(x - 1.5*width, s_dicts['l3.accessPFSrc'], width, label='l3Access', color=contrasting_orange[4])
    ax.bar(x - 0.5*width, s_dicts['l3.hitsPFSrc'], width, label='l3Hits', color=contrasting_orange[5])
    # percentage bar for l3HitRate and pfAcc
    ax2 = ax.twinx()
    ax2.bar(x + 0.5*width, s_dicts['l3.pfHitRate'], width, label='l3HitRate', color=contrasting_orange[6])
    ax2.bar(x + 1.5*width, dcache_pfAcc, width, label='pfAcc', color=contrasting_orange[7])

    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}' for i in range(pf_src_num)])
    ax.set_xlabel('pf src')
    ax.set_ylabel('count')
    ax2.set_ylabel('pfAcc/l3HitRate')
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax2.set_ylim(0,1)
    # draw total bar next to the last pf src
    ax.bar(pf_src_num + 0.5*width, l3Access_all, width, label='l3Access', color=contrasting_orange[4])
    ax.bar(pf_src_num + 1.5*width, l3Hits_all, width, label='l3Hits', color=contrasting_orange[5])
    ax2.bar(pf_src_num + 2.5*width, l3HitRate_all, width, label='l3HitRate', color=contrasting_orange[6])
    ax2.bar(pf_src_num + 3.5*width, pfAcc_all, width, label='pfAcc', color=contrasting_orange[7])

    ax.set_title(f'{workload_name}({workname_postfix(workload_name)})')
    labels = [
        'l3Access',
        'l3Hits',
        'l3HitRate',
        'l1dPfAcc',
    ]
    return labels

def draw_db_by_func(base_dir,n_rows,worksname_waydict,draw_one_func,fig_name,n_cols=6,
                    force_update_json = False):

    work_stats_dict = {}
    dict_updated = False

    n_rows = math.ceil(len(worksname_waydict) / n_cols)

    parameters = {'axes.labelsize': 30,
          'axes.titlesize': 30,
          'xtick.labelsize': 22,
          'ytick.labelsize': 22,
          'legend.fontsize': 30,
          'font.size': 20,
          'axes.facecolor': 'white',
          'figure.facecolor': 'white',
          'savefig.facecolor': 'white',
          }
    plt.rcParams.update(parameters)

    fig,ax = plt.subplots(n_rows,n_cols)
    
    fig.set_size_inches(n_cols*10, 6*n_rows)

    policy_re = re.compile(r'l3-1-pf')
    force_update = False
    # force_update = True
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
            # s_dicts['ipc'] = np.zeros(max_assoc)
            s_dicts['l3.hitsPFSrc'] = np.zeros(pf_src_num)
            s_dicts['l3.missesPFSrc'] = np.zeros(pf_src_num)
            s_dicts['dcache.pfIssuedSrc'] = np.zeros(pf_src_num)
            s_dicts['dcache.pfUsefulSrc'] = np.zeros(pf_src_num)
            s_dicts['l2.pfIssuedSrc'] = np.zeros(pf_src_num)
            s_dicts['l2.pfUsefulSrc'] = np.zeros(pf_src_num)
            s_dicts['l3.pfIssuedSrc'] = np.zeros(pf_src_num)
            s_dicts['l3.pfUsefulSrc'] = np.zeros(pf_src_num)
            partsname = os.listdir(word_dir) #like l3-1
            for part in partsname:
                if not os.path.isdir(os.path.join(word_dir,part)):
                    continue
                m = policy_re.match(part)
                if m is None:
                    continue
                # ways = int(m.group(1))
                # policy = m.group(2)
                # # if not any([r.match(policy) for r in interested_res]):
                # #     continue
                # if policy != 'pf':
                #     continue
                new_base = os.path.join(word_dir,part)
                if force_update:
                    one_dict = extract_newgem_raw_json(new_base,ncores=ncore,last_nsamples=last_nsamples)
                elif not os.path.exists(os.path.join(new_base,f'{last_nsamples}period.json')):
                    one_dict = extract_newgem_raw_json(new_base,ncores=ncore,last_nsamples=last_nsamples)
                else:
                    with open(os.path.join(new_base,f'{last_nsamples}period.json'),'r') as f:
                        one_dict = json.load(f)
                for ps in range(pf_src_num):
                    s_dicts['l3.hitsPFSrc'][ps] = one_dict[f'l3.hitsPFSrc_{ps}'][0]
                    s_dicts['l3.missesPFSrc'][ps] = one_dict[f'l3.missesPFSrc_{ps}'][0]
                    s_dicts['dcache.pfIssuedSrc'][ps] = one_dict[f'dcache.pfIssuedSrc_{ps}'][0]
                    s_dicts['dcache.pfUsefulSrc'][ps] = one_dict[f'dcache.pfUsefulSrc_{ps}'][0]
                    s_dicts['l2.pfIssuedSrc'][ps] = one_dict[f'l2.pfIssuedSrc_{ps}'][0]
                    s_dicts['l2.pfUsefulSrc'][ps] = one_dict[f'l2.pfUsefulSrc_{ps}'][0]
                    s_dicts['l3.pfIssuedSrc'][ps] = one_dict[f'l3.pfIssuedSrc_{ps}'][0]
                    s_dicts['l3.pfUsefulSrc'][ps] = one_dict[f'l3.pfUsefulSrc_{ps}'][0]
                s_dicts['total.pfIssuedSrc'] = s_dicts['dcache.pfIssuedSrc'] + s_dicts['l2.pfIssuedSrc'] + s_dicts['l3.pfIssuedSrc']
                s_dicts['total.pfUsefulSrc'] = s_dicts['dcache.pfUsefulSrc'] + s_dicts['l2.pfUsefulSrc'] + s_dicts['l3.pfUsefulSrc']
                s_dicts['l3.accessPFSrc'] = s_dicts['l3.hitsPFSrc'] + s_dicts['l3.missesPFSrc']
                s_dicts['pfAcc'] = s_dicts['total.pfUsefulSrc'] / s_dicts['total.pfIssuedSrc']
                s_dicts['l3.pfHitRate'] = s_dicts['l3.hitsPFSrc'] / (s_dicts['l3.hitsPFSrc'] + s_dicts['l3.missesPFSrc'])
            # work_stats_dict[work] = s_dicts

        mylabels = draw_one_func(ax_bar,s_dicts,work,max_assoc,(fx,fy))

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
    # mylabels = [
    #     'pfIssued',
    #     'pfUseful',
    #     'l3Access',
    #     'pfAcc',
    # ]
    for c,policy in enumerate(mylabels):
        legends.append(Patch(color=contrasting_orange[(c+4)%len(contrasting_orange)],label=f'{policy}'))
    fig.legend(handles = legends, loc = 'lower right', ncol = 1,borderaxespad=0, labelspacing=0, handlelength=0.5)

    plt.tight_layout()
    plt.savefig(fig_name,dpi=150)
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

    global cs_works
    cs_works = use_conf['cs_works']
    global ps_works
    ps_works = use_conf['ps_works']
    global cs_ps_works
    cs_ps_works = use_conf['cs-ps_works']


    global worksname
    worksname = os.listdir(base_dir)
    worksname.sort()
    global pf_works
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
    # worksname = pf_works
    # worksname = cs_ps_works + ps_works
    # worksname.sort()

    interested_works = {}
    for w in worksname:
        interested_works[w] = max_assoc    

    n_works = len(worksname)
    n_cols = 4
    n_rows = math.ceil(n_works/n_cols)
    draw_db_by_func(base_dir,n_rows,interested_works,
                    n_cols=n_cols,
        draw_one_func=draw_access_issue_bar,fig_name=os.path.join(pic_dir_path,'pf-1way-l1accu-l3access.png'))
    draw_db_by_func(base_dir,n_rows,interested_works,
                    n_cols=n_cols,
        draw_one_func=draw_hitrate_acc_bar,fig_name=os.path.join(pic_dir_path,'pf-1way-l1accu-l3hitrate-l3access.png'))


if __name__ == '__main__':
    if opt.json:
        select_json = opt.json
        run_one_conf(select_json)
    else:
        for co in confs:
            select_json = co
            run_one_conf(select_json)