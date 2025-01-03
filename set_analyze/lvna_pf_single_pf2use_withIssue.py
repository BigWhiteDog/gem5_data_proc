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

pf_src_num = 13
def workname_postfix(workname):
    if workname in cs_works:
        return 'cs'
    elif workname in ps_works:
        return 'ps'
    elif workname in cs_ps_works:
        return 'cs-ps'
    return ''

def draw_one_workload_l1_hit_age(ax,s_dicts,workload_name,full,pos:tuple):
    l1_hit_age = s_dicts['l1_hit_ages']
    # pf_src_len_pair_list = [(len(l1_hit_age[i]),i) for i in range(pf_src_num)]
    # pf_src_len_pair_list.sort(reverse=True)
    issue_num_pair_list = [(s_dicts['dcache.pfIssuedSrc'][i],i) for i in range(pf_src_num)]
    issue_num_pair_list.sort(reverse=True)

    most_num = 5
    for i in range(most_num):
        src_sel = issue_num_pair_list[i][1]
        src_issued_num = s_dicts['dcache.pfIssuedSrc'][src_sel] 
        hitage_hist, hitage_bins = np.histogram(l1_hit_age[src_sel],bins='auto')
        hitage_hist = hitage_hist / src_issued_num
        bin_centers = (hitage_bins[:-1] + hitage_bins[1:]) / 2
        hitage_cumsum = np.cumsum(hitage_hist)
        ax.plot(bin_centers,hitage_cumsum,label=f'most hit id {src_sel}',color=contrasting_orange[i+4])
        # ax.bar(bin_centers,hitage_hist,label=f'most hit id {src_sel}',color=contrasting_orange[i+4])
    
    ax.set_title(f'{workload_name}({workname_postfix(workload_name)})')
    
    ax.set_xlabel('age cycle')
    ax.set_ylabel('issue ratio')
    # formatter = ticker.ScalarFormatter(useMathText=True)
    # formatter.set_scientific(True)
    # formatter.set_powerlimits((-1,1))
    # ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

    # ax.axvline(x=200, color='b', linestyle='--')
    ax.axvline(x=512, color='b', linestyle='--')
    ax.axvline(x=2048, color='b', linestyle='--')
    ax.axvline(x=4096, color='b', linestyle='--')
    # ax.axvline(x=2000, color='b', linestyle='--')

    # ax.set_xlim(0,2000)
    ax.set_ylim(0,1)
    ax.set_xlim(0,5000)

    return [f'most hit id {s}' for s in range(most_num)]

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
            s_dicts['l3.hitsPFSrc'] = np.zeros(pf_src_num)
            s_dicts['l3.missesPFSrc'] = np.zeros(pf_src_num)
            s_dicts['dcache.pfIssuedSrc'] = np.zeros(pf_src_num)
            s_dicts['dcache.pfUsefulSrc'] = np.zeros(pf_src_num)
            s_dicts['l2.pfIssuedSrc'] = np.zeros(pf_src_num)
            s_dicts['l2.pfUsefulSrc'] = np.zeros(pf_src_num)
            s_dicts['l3.pfIssuedSrc'] = np.zeros(pf_src_num)
            s_dicts['l3.pfUsefulSrc'] = np.zeros(pf_src_num)
            
            new_base = os.path.join(word_dir,f'l3-16-pf')
            with open(os.path.join(new_base,f'1period.json'),'r') as f:
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
            
            db_path = os.path.join(new_base,'hm.db')
            con = sqlite3.connect(db_path)
            cur = con.cursor()

            n_hid = pf_src_num
            l1_hit_cnt = np.zeros(n_hid,dtype=int)
            l1_hit_ages = [list() for _ in range(n_hid)]
            l2_hit_cnt = np.zeros(n_hid,dtype=int)
            l2_hit_ages = [list() for _ in range(n_hid)]
            l3_hit_cnt = np.zeros(n_hid,dtype=int)
            l3_hit_ages = [list() for _ in range(n_hid)]
            total_hit_cnt = np.zeros(n_hid,dtype=int)
            total_hit_ages = [list() for _ in range(n_hid)]

            all_access_query = 'SELECT BLKID,HITLEVEL,AGECYCLE FROM L3BlkFirstHitAge;'
            f = cur.execute(all_access_query)

            for blkid,hitlevel,agecycle in f:
                blkid = int(blkid)
                hitlevel = int(hitlevel)
                agecycle = int(agecycle)
                if hitlevel == 1:
                    l1_hit_ages[blkid].append(agecycle)
                elif hitlevel == 2:
                    l2_hit_ages[blkid].append(agecycle)
                elif hitlevel == 3:
                    l3_hit_ages[blkid].append(agecycle)
                total_hit_ages[blkid].append(agecycle)

            cur.close()
            #sort by hit cnts for each level
            s_dicts['l1_hit_ages'] = l1_hit_ages
            s_dicts['l2_hit_ages'] = l2_hit_ages
            s_dicts['l3_hit_ages'] = l3_hit_ages
            s_dicts['total_hit_ages'] = total_hit_ages
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

    interested_works = {}
    for w in worksname:
        interested_works[w] = max_assoc    

    n_works = len(worksname)
    n_cols = 4
    n_rows = math.ceil(n_works/n_cols)
    draw_db_by_func(base_dir,n_rows,interested_works,
                    n_cols=n_cols,
        draw_one_func=draw_one_workload_l1_hit_age,fig_name=os.path.join(pic_dir_path,'pf2use_issue_dist.png'))


if __name__ == '__main__':
    if opt.json:
        select_json = opt.json
        run_one_conf(select_json)
    else:
        for co in confs:
            select_json = co
            run_one_conf(select_json)