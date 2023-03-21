from collections import OrderedDict
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
from sortedcontainers import SortedDict,SortedList,SortedKeyList
from matplotlib.ticker import MaxNLocator

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

opt = parser.parse_args()

from cache_sensitive_names import *
from set_analyze.my_diff_color import *

all_set = 16384

def getTruePositive(standard_set,est_set):
    tmp = np.logical_and(standard_set,est_set)
    return tmp.sum()/all_set
def getFalsePositive(standard_set,est_set):
    tmp = np.logical_and(standard_set,np.logical_not(est_set))
    return tmp.sum()/all_set
def getFalseNegative(standard_set,est_set):
    tmp = np.logical_and(np.logical_not(standard_set),est_set)
    return tmp.sum()/all_set
def getTrueNegative(standard_set,est_set):
    tmp = np.logical_and(np.logical_not(standard_set),np.logical_not(est_set))
    return tmp.sum()/all_set

def draw_one_workload_pnlast_f1(ax,s_dicts,workload_name,full_ass,pos:tuple):

    factor = s_dicts['factor']
    stride_size = s_dicts['stride_size']
    pn_est_bits = s_dicts['pn_est_bits']
    stride_est_bits = s_dicts['stride_est_bits']
    stride_est_bits = np.tile(stride_est_bits, factor)
    cont_est_bits = s_dicts['cont_est_bits']
    cont_est_bits = np.repeat(cont_est_bits, factor)
    sandc_est_bits = np.logical_and(stride_est_bits,cont_est_bits)
    real_ways = s_dicts['min_ways_no_extra_miss']
    real_full_bits = np.array([ i == full_ass for i in real_ways])
    labels = ['pn','stride','cont', 's_and_c']
    est_bits = [pn_est_bits,stride_est_bits,cont_est_bits,sandc_est_bits]
    
    f1_func = [getTruePositive,getFalsePositive,getFalseNegative,getTrueNegative]
    f1_labels = ['TP','FP','FN','TN']

    xvals = np.arange(len(labels))
    width = 0.2
    bots = np.zeros(len(labels))
    color = 0
    for f1_f,f1_label in zip(f1_func,f1_labels):
        vals = []
        for est_bit in est_bits:
            vals.append(f1_f(real_full_bits,est_bit))
        ax.bar(xvals,vals,width,bottom=bots,label=f1_label,color = contrasting_orange[color])
        bots += vals
        color += 1
    ax.set_xticks(xvals)
    ax.set_xticklabels(labels)
    ax.set_title(f'{workload_name}')
    ax.set_ylabel('rate')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))

    if pos == (0,0):
        bar_patch = [Patch(color=contrasting_orange[i],label=f'{f1_labels[i]}')
                    for i in range(len(f1_labels))]
        ax.legend(handles = bar_patch, shadow=0, fontsize = 13, bbox_to_anchor=(-0.01,1.4), loc = 'upper left',  \
            borderaxespad=0.1, ncol = 2, columnspacing=0.5, labelspacing=0.2)

def draw_one_workload_last_way_trace_scatter(ax,s_dicts,workload_name,full_ass,pos:tuple):
    grow_trace = s_dicts['grow_trace']
    time_t_list = [g[0] for g in grow_trace]
    tag_t_list = [g[1] for g in grow_trace]
    ax.scatter(time_t_list,tag_t_list, s=0.1, label='setidx trace', color = contrasting_orange[4])
    ax.set_xlabel('cycles')
    ax.set_ylabel('setidx')
    ax.set_xlim(0,s_dicts['cpu_num_cycles'])
    ax.set_title(f'{workload_name}')            

class SetPositiveState:
    def __init__(self, set_id, full_ass, meta_bits=2, start_postive=1, decrease_f=0.5):
        self.set_id = set_id
        self.full_ass = full_ass
        self.meta_bits = meta_bits
        self.positive_bits = np.full(full_ass, False)
        self.positive_cyclelen_record = {start_postive:0}
        self.positive_total_blocklen = {start_postive:0}
        self.positive_num = start_postive
        for i in range(start_postive):
            self.positive_bits[i] = True
        self.decrease_f = decrease_f
        self.hitlen = 0
        self.blocklen = 0

        self.meta_mask = (1 << meta_bits) - 1

    def newHit(self, way_id, delta_stamp, total_blocklen, total_hitlen):
        self.blocklen += 1
        self.hitlen += 1
        if not self.positive_bits[way_id]:
            #when hit a negative block, there will be new infeciton
            total_less_ways = self.full_ass - self.positive_num
            need_grow_ways = math.ceil(total_less_ways * self.decrease_f)
            self.positive_bits[way_id] = True
            self.positive_num += 1
            need_grow_ways -= 1
            for i in range(self.full_ass):
                if need_grow_ways <= 0:
                    break
                if not self.positive_bits[i]:
                    self.positive_bits[i] = True
                    self.positive_num += 1
                    need_grow_ways -= 1
            self.positive_cyclelen_record[self.positive_num] = delta_stamp
            self.positive_total_blocklen[self.positive_num] = total_blocklen


    def newBlock(self, way_id, meta_datas):
        self.blocklen += 1
        max_positive_idx = -1
        max_positive_value = -1
        if not self.positive_bits[way_id]:
            for i in range(self.full_ass):
                #find max metadata positive block
                if self.positive_bits[i]:
                    tmp_meta = meta_datas & self.meta_mask
                    if tmp_meta > max_positive_value:
                        max_positive_value = tmp_meta
                        max_positive_idx = i
                meta_datas >>= self.meta_bits
            if max_positive_idx >= 0:
                self.positive_bits[max_positive_idx] = False
            self.positive_bits[way_id] = True


def analyze_pn_lencycle_est(work_stats_dict,work,work_dir,full_ass):
    if work in work_stats_dict:
        return
    s_2 = re.compile(r'(\w+)-([\w\.]+)')
    s_dicts = {}

    partsname = os.listdir(work_dir) #like l3-1
    for part in partsname:
        if not os.path.isdir(os.path.join(work_dir,part)):
            continue
        res = s_2.search(part)
        if not res:
            continue
        if res.group(1) != 'l3':
            continue
        ways = int(res.group(2))
        if ways != full_ass:
            continue

        new_base = os.path.join(work_dir,part)
        with open(os.path.join(new_base,f'1period.json'),'r') as f:
            one_dict = json.load(f)
        
        another_part = os.path.join(work_dir,f'l3-{ways-1}')
        with open(os.path.join(another_part,f'1period.json'),'r') as f:
            another_dict = json.load(f)

        s_dicts['oneless_demand_hits'] = another_dict['l3.demandHits'][0]
        s_dicts['oneless_demand_misses'] = another_dict['l3.demandMisses'][0]
        s_dicts['oneless_ipc'] = another_dict['cpu.ipc'][0]
        
        s_dicts['ipc'] = one_dict['cpu.ipc'][0]
        s_dicts['cpu_num_cycles'] = one_dict['cpu.numCycles'][0]
        s_dicts['l3_total_demand'] = one_dict['l3.demandHits'][0] + one_dict['l3.demandMisses'][0]

        # print(f'work {work} ways {ways} totaldemand {s_dicts["l3_total_demand"]} totalhits {one_dict["l3.demandHits"][0]}'\
        #     f' totalmisses {one_dict["l3.demandMisses"][0]}')

        db_path = os.path.join(new_base,'hm.db')
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        stamp0_query = 'SELECT min(STAMP),max(STAMP)-min(STAMP) from HitMissTrace;'
        f = cur.execute(stamp0_query)
        stamp0,delta_stamp_last = f.fetchone()
        stamp0 = int(stamp0)

        all_access_query = 'SELECT SETIDX,WAYIDX,ISINS,METAS,VALIDMASK,STAMP FROM HitPosTrace ORDER BY ID;'
        f = cur.execute(all_access_query)

        pn_start_positive = full_ass -1
        pn_est_bits = np.full(all_set,False)
        factor = 32
        stride_size = all_set // factor
        stride_est_bits = np.full(all_set//factor, False)
        cont_est_bits = np.full(all_set//factor, False)
        find_sets = set()
        grow_trace = []
        hitlast_cnt = 0
        for setidx,wayidx,isins,metas,stamp in f:
            setidx = int(setidx)
            wayidx = int(wayidx)
            isins = bool(int(isins))
            stamp = int(stamp)
            delta_stamp = stamp - stamp0
            if isins:
                pass
            else:
                #hit block
                if wayidx >= pn_start_positive:
                    if setidx not in find_sets:
                    # if True:
                        find_sets.add(setidx)
                        grow_trace.append((delta_stamp,setidx))
                    hitlast_cnt += 1
                    pn_est_bits[setidx] = True
                    stride_est_bits[setidx % stride_size] = True
                    cont_est_bits[setidx // factor] = True
        print(f'work {work} ways {ways} totaldemand {s_dicts["l3_total_demand"]} totalhits {one_dict["l3.demandHits"][0]}'\
            f' totalmisses {one_dict["l3.demandMisses"][0]} ipc {s_dicts["ipc"]}'\
            f' hitlast_cnt {hitlast_cnt} grow_cnt {len(grow_trace)}'\
            f' oneless_demand_hits {s_dicts["oneless_demand_hits"]} oneless_demand_misses {s_dicts["oneless_demand_misses"]}'\
                f' oneless_ipc {s_dicts["oneless_ipc"]}')
        cur.close()

    # stride_tmp = pn_est_bits.reshape(stride_size,factor)
    # stride_est_bits = np.any(stride_tmp,axis=1)

    s_dicts['factor'] = factor
    s_dicts['stride_size'] = stride_size
    s_dicts['pn_est_bits'] = pn_est_bits
    s_dicts['stride_est_bits'] = stride_est_bits
    s_dicts['cont_est_bits'] = cont_est_bits
    s_dicts['grow_trace'] = grow_trace

    work_stats_dict[work] = s_dicts

def draw_db_by_func(base_dir,n_rows,worksname_waydict,analyze_func,draw_one_func,fig_name,
    csv_top_dir=None,
    input_stats_dict=None):
    fig,ax = plt.subplots(n_rows,4)
    fig.set_size_inches(24, 4.5*n_rows+3)

    # work_stats_dict = {}
    # if input_stats_dict is not None:
    #     work_stats_dict = input_stats_dict
    work_stats_dict = input_stats_dict

    for i,work in enumerate(worksname_waydict):
        full_ass = worksname_waydict[work]
        work_dir = os.path.join(base_dir,work)
        if not os.path.isdir(work_dir):
            continue
        fy = i % 4
        fx = i // 4
        ax_bar = ax[fx,fy]
        analyze_func(work_stats_dict,work,work_dir,full_ass)
        s_dicts = work_stats_dict[work]
        s_dicts['min_ways_no_extra_miss'] = []
        if csv_top_dir is not None:
            csv_file = os.path.join(csv_top_dir,f'{work}.csv')
            with open(csv_file,'r') as f:
                for i in range(all_set):
                    s_dicts['min_ways_no_extra_miss'].append(int(f.readline().strip()))
        draw_one_func(ax_bar,s_dicts,work,full_ass,(fx,fy))     

    for i in range(len(worksname_waydict),n_rows*4):
        fx = i // 4
        fy = i % 4
        ax[fx,fy].remove()

    plt.tight_layout()
    plt.savefig(fig_name,dpi=300)
    plt.clf()

    return work_stats_dict


if __name__ == '__main__':
    use_conf = conf_50M
    test_prefix = use_conf['test_prefix']
    base_dir = base_dir_format.format(test_prefix)
    csv_dir_path = f'set_analyze/{test_prefix}other/csv'
    pic_dir_path = f'set_analyze/{test_prefix}pics'
    os.makedirs(pic_dir_path, exist_ok=True)
    worksname = use_conf['cache_work_names'] #like mcf

    n_works = len(worksname)
    n_rows = math.ceil(n_works/4)

    waydict_format = 'cache_work_{}ways'
    # perf_prefixs = ['90perf','95perf','full']
    perf_prefixs = ['95perf']
    draw_picformat = [
        # (draw_one_workload_pnlast_cyclelen_hist,'pn_last_cyclelen_hist_{}.png',None),
        # (draw_one_workload_pnlast_blocklen_hist,'pn_last_blocklen_hist_{}.png',None),
        (draw_one_workload_pnlast_f1,'pn_last_compress_{}.png',os.path.join(csv_dir_path,'min0way_{}')),
        (draw_one_workload_last_way_trace_scatter,'pn_last_way_trace_scatter_{}.png',None),
    ]

    for perf_prefix in perf_prefixs:
        waydict_name = waydict_format.format(perf_prefix)
        waydict = use_conf[waydict_name]
        ret_dict = {}
        print(f'perf_prefix:{perf_prefix}')
        for draw_func,pic_name_format,csv_dir_format in draw_picformat:
            if csv_dir_format is not None:
                csv_dir = csv_dir_format.format(perf_prefix)
            else:
                csv_dir = None
            draw_db_by_func(base_dir,n_rows,waydict,
                analyze_func=analyze_pn_lencycle_est,
                draw_one_func=draw_func,
                fig_name=os.path.join(pic_dir_path,pic_name_format.format(perf_prefix)),
                csv_top_dir=csv_dir,
                input_stats_dict=ret_dict)