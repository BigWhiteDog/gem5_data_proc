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

def draw_one_workload_pnlast_cyclelen_hist(ax,s_dicts,workload_name,full_ass,pos:tuple):
    cycle_lens = s_dicts['last_one_need_cycle']

    x_val = np.arange(all_set)

    ax.hist(cycle_lens , bins = 'auto', label='last_one_need_cycle',histtype = 'bar', 
            density=True, cumulative=True,color = contrasting_orange[0],  linewidth=2)

    ax.set_ylabel('portion of sets')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.set_xlim(0,s_dicts['cpu_num_cycles'])
    ax.set_xlabel('needed cycles')
    ax.set_title(f'{workload_name}')
    if pos == (0,0):
        ax.legend(shadow=0, fontsize = 13, bbox_to_anchor=(-0.01,1.4), loc = 'upper left',  \
            borderaxespad=0.2, ncol = 2, columnspacing=0.5, labelspacing=0.1)
def draw_one_workload_pnlast_blocklen_hist(ax,s_dicts,workload_name,full_ass,pos:tuple):
    blocklens = s_dicts['last_one_need_block']

    x_val = np.arange(all_set)

    ax.hist(blocklens , bins = 'auto', label='last_one_need_block',histtype = 'bar', 
            density=True, cumulative=True,color = contrasting_orange[1],  linewidth=2)

    ax.set_ylabel('portion of sets')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.set_xlim(0,s_dicts['l3_total_demand'])
    ax.set_xlabel('needed blocks')
    ax.set_title(f'{workload_name}')
    if pos == (0,0):
        ax.legend(shadow=0, fontsize = 13, bbox_to_anchor=(-0.01,1.4), loc = 'upper left',  \
            borderaxespad=0.2, ncol = 2, columnspacing=0.5, labelspacing=0.1)
def draw_one_workload_pnlast_hitlen_hist(ax,s_dicts,workload_name,full_ass,pos:tuple):
    blocklens = s_dicts['last_one_need_hit']

    x_val = np.arange(all_set)

    ax.hist(blocklens , bins = 'auto', label='last_one_need_hit',histtype = 'bar', 
            density=True, cumulative=True,color = contrasting_orange[12],  linewidth=2)

    ax.set_ylabel('portion of sets')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    # ax.set_xlim(0,s_dicts['l3_total_demand'])
    ax.set_xlabel('needed hits')
    ax.set_title(f'{workload_name}')
    if pos == (0,0):
        ax.legend(shadow=0, fontsize = 13, bbox_to_anchor=(-0.01,1.4), loc = 'upper left',  \
            borderaxespad=0.2, ncol = 2, columnspacing=0.5, labelspacing=0.1)

def draw_one_workload_pnlast_tti_change(ax,s_dicts,workload_name,full_ass,pos:tuple):
    tti_record = s_dicts['tti_record']
    #tti record is accescnt, hitcnt, growcnt
    seperate_access = np.array([a for a,_,_ in tti_record])
    seperate_hit = np.array([h for _,h,_ in tti_record])
    seperate_grow = np.array([g for _,_,g in tti_record])

    seperate_hitrate = seperate_hit / seperate_access
    seperate_growrate = seperate_grow / seperate_access

    seperate_growthhitrate = seperate_grow / seperate_hit

    cumsum_access = seperate_access.cumsum()
    cumsum_hit = seperate_hit.cumsum()
    cumsum_grow = seperate_grow.cumsum()

    cum_hitrate = cumsum_hit / cumsum_access
    cum_growrate = cumsum_grow / cumsum_access

    cum_growthhitrate = cumsum_grow / cumsum_hit

    grow_setrate = seperate_grow / all_set
    cumsum_grow_setrate = cumsum_grow / all_set

    # ax.plot(seperate_hitrate, label='hitrate',color = contrasting_orange[0],  linewidth=2)
    # ax.plot(seperate_growrate, label='growrate',color = contrasting_orange[1],  linewidth=2)

    # ax.plot(cum_hitrate, label='cum_hitrate',color = contrasting_orange[2],  linewidth=2)
    # ax.plot(cum_growrate, label='cum_growrate',color = contrasting_orange[3],  linewidth=2)

    # ax.plot(seperate_growthhitrate, label='growthhitrate',color = contrasting_orange[0],  linewidth=2)
    # ax.plot(cum_growthhitrate, label='cum_growthhitrate',color = contrasting_orange[1],  linewidth=2)

    ax.plot(grow_setrate, label='grow_setrate',color = contrasting_orange[4],  linewidth=2)
    ax.plot(cumsum_grow_setrate, label='cum_grow_setrate',color = contrasting_orange[5],  linewidth=2)

    # ax.plot(normalize_grow_setrate, label='normalize_grow_setrate',color = contrasting_orange[6],  linewidth=2)

    ax.set_ylabel('rate')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.set_xlabel(f'number of interval (each is {s_dicts["tti_interval"]})')
    ax.set_title(f'{workload_name}')
    if pos == (0,0):
        ax.legend(shadow=0, fontsize = 13, bbox_to_anchor=(-0.01,1.4), loc = 'upper left',  \
            borderaxespad=0.2, ncol = 2, columnspacing=0.5, labelspacing=0.1)

def draw_one_workload_pnlast_tti_growthchange(ax,s_dicts,workload_name,full_ass,pos:tuple):
    tti_record = s_dicts['tti_record']
    #tti record is accescnt, hitcnt, growcnt
    seperate_access = np.array([a for a,_,_ in tti_record])
    seperate_hit = np.array([h for _,h,_ in tti_record])
    seperate_grow = np.array([g for _,_,g in tti_record])

    seperate_hitrate = seperate_hit / seperate_access
    seperate_growrate = seperate_grow / seperate_access

    seperate_growthhitrate = seperate_grow / seperate_hit

    cumsum_access = seperate_access.cumsum()
    cumsum_hit = seperate_hit.cumsum()
    cumsum_grow = seperate_grow.cumsum()

    cum_hitrate = cumsum_hit / cumsum_access
    cum_growrate = cumsum_grow / cumsum_access

    cum_growthhitrate = cumsum_grow / cumsum_hit

    grow_setrate = seperate_grow / all_set
    cumsum_grow_setrate = cumsum_grow / all_set

    # ax.plot(seperate_hitrate, label='hitrate',color = contrasting_orange[0],  linewidth=2)
    # ax.plot(seperate_growrate, label='growrate',color = contrasting_orange[1],  linewidth=2)

    # ax.plot(cum_hitrate, label='cum_hitrate',color = contrasting_orange[2],  linewidth=2)
    # ax.plot(cum_growrate, label='cum_growrate',color = contrasting_orange[3],  linewidth=2)

    ax.plot(seperate_growthhitrate, label='growthhitrate',color = contrasting_orange[0],  linewidth=2)
    ax.plot(cum_growthhitrate, label='cum_growthhitrate',color = contrasting_orange[1],  linewidth=2)

    # ax.plot(grow_setrate, label='grow_setrate',color = contrasting_orange[4],  linewidth=2)
    # ax.plot(cumsum_grow_setrate, label='cum_grow_setrate',color = contrasting_orange[5],  linewidth=2)

    # ax.plot(normalize_grow_setrate, label='normalize_grow_setrate',color = contrasting_orange[6],  linewidth=2)

    ax.set_ylabel('rate')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
    ax.set_xlabel(f'number of interval (each is {s_dicts["tti_interval"]})')
    ax.set_title(f'{workload_name}')
    if pos == (0,0):
        ax.legend(shadow=0, fontsize = 13, bbox_to_anchor=(-0.01,1.4), loc = 'upper left',  \
            borderaxespad=0.2, ncol = 2, columnspacing=0.5, labelspacing=0.1)
def draw_one_workload_pnlast_growbucket(ax,s_dicts,workload_name,full_ass,pos:tuple):
    total_hit_len = s_dicts['total_hit_len']
    record_bucket_number = s_dicts['record_bucket_number'] 
    grow_hitlen_record = s_dicts['grow_hitlen_record']

    r_record = np.flip(grow_hitlen_record)
    r_c_record = r_record.cumsum()
    r_crate_record = r_c_record / total_hit_len
    ax.plot(r_crate_record, label='cum_growrate',color = contrasting_orange[0],  linewidth=2)

    ax.set_ylabel('rate')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
    ax.set_xlabel(f'number of buckets (each is {all_set // record_bucket_number}grow set)')
    ax.set_title(f'{workload_name}')
    if pos == (0,0):
        ax.legend(shadow=0, fontsize = 13, bbox_to_anchor=(-0.01,1.4), loc = 'upper left',  \
            borderaxespad=0.2, ncol = 2, columnspacing=0.5, labelspacing=0.1)



class SetPositiveState:
    def __init__(self, set_id, full_ass, meta_bits=2, start_postive=1, decrease_f=0.5):
        self.set_id = set_id
        self.full_ass = full_ass
        self.meta_bits = meta_bits
        self.positive_bits = np.full(full_ass, False)
        self.positive_cyclelen_record = {start_postive:0}
        self.positive_total_blocklen = {start_postive:0}
        self.positive_total_hitlen = {start_postive:0}
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
            self.positive_total_hitlen[self.positive_num] = total_hitlen
            return True
        else:
            return False


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
        s_dicts['cpu_num_cycles'] = one_dict['cpu.numCycles'][0]
        s_dicts['l3_total_demand'] = one_dict['l3.demandHits'][0] + one_dict['l3.demandMisses'][0]

        db_path = os.path.join(new_base,'hm.db')
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        stamp0_query = 'SELECT min(STAMP),max(STAMP)-min(STAMP) from HitMissTrace;'
        f = cur.execute(stamp0_query)
        stamp0,delta_stamp_last = f.fetchone()

        tti_hitblock_l = []
        tti_len = 5_000_000

        all_hitmiss_query = 'SELECT ISMISS,STAMP FROM HitMissTrace ORDER BY ID;'
        f = cur.execute(all_hitmiss_query)
        def insertIntoTTI(t,ishit,isgrow):
            # total access, ishit, isgrow
            if t >= len(tti_hitblock_l):
                for _ in range( t - len(tti_hitblock_l) + 1 ):
                    tti_hitblock_l.append([0,0,0])
            if isgrow:
                #grow
                tti_hitblock_l[t][2] += 1
            elif ishit:
                #hit access cnt and add hitcnt
                tti_hitblock_l[t][0] += 1
                tti_hitblock_l[t][1] += 1
            else:
                #miss, add access cnt
                tti_hitblock_l[t][0] += 1
        
        for ismiss,stamp in f:
            ismiss = bool(int(ismiss))
            stamp = int(stamp)
            delta_stamp = stamp - stamp0
            tti = delta_stamp // tti_len
            insertIntoTTI(tti,not ismiss,False)

        all_access_query = 'SELECT SETIDX,WAYIDX,ISINS,METAS,STAMP FROM HitPosTrace ORDER BY ID;'
        f = cur.execute(all_access_query)

        # pn_start_positive = math.ceil(full_ass/2)
        pn_start_positive = full_ass -1
        pn_states = [SetPositiveState(i,full_ass,
            start_postive=pn_start_positive,
            decrease_f=1,
            ) for i in range(all_set)]
        total_hit_len = 0
        total_block_len = 0
        
        total_grow_cnt = 0
        record_bucket_number = 32
        one_grow_bucket = all_set//record_bucket_number
        grow_hitlen_record = np.zeros(record_bucket_number)
        for setidx,wayidx,isins,metas,stamp in f:
            setidx = int(setidx)
            wayidx = int(wayidx)
            isins = bool(int(isins))
            metas = int(metas)
            stamp = int(stamp)
            delta_stamp = stamp - stamp0
            tti = delta_stamp // tti_len
            total_block_len += 1
            if isins:
                #insert block
                pn_states[setidx].newBlock(wayidx,metas)
            else:
                #hit block
                total_hit_len += 1
                grow_happend = pn_states[setidx].newHit(wayidx,delta_stamp,
                                total_block_len,total_hit_len)
                insertIntoTTI(tti,True,grow_happend)
                if grow_happend:
                    bucket_idx = total_grow_cnt // one_grow_bucket
                    total_grow_cnt += 1
                    grow_hitlen_record[bucket_idx] += 1

        cur.close()
    s_dicts['est_used_ways'] = [0 for _ in range(all_set)]
    s_dicts['last_one_need_cycle'] = [0 for _ in range(all_set)]
    s_dicts['last_one_need_block'] = [0 for _ in range(all_set)]
    s_dicts['last_one_need_hit'] = [0 for _ in range(all_set)]
    s_dicts['tti_record'] = tti_hitblock_l
    s_dicts['tti_interval'] = f'{tti_len//1_000_000}M'
    
    s_dicts['total_hit_len'] = total_hit_len
    s_dicts['record_bucket_number'] = record_bucket_number
    s_dicts['grow_hitlen_record'] = grow_hitlen_record

    for idx in range(all_set):
        set_pn_state = pn_states[idx]
        s_dicts['est_used_ways'][idx] = max(set_pn_state.positive_num,1)
        pcycle_dict = set_pn_state.positive_cyclelen_record
        pb_dict = set_pn_state.positive_total_blocklen
        ph_dict = set_pn_state.positive_total_hitlen
        if set_pn_state.positive_num == pn_start_positive:
            s_dicts['last_one_need_cycle'][idx] = s_dicts['cpu_num_cycles']
            s_dicts['last_one_need_block'][idx] = s_dicts['l3_total_demand']
            # s_dicts['last_one_need_hit'][idx] = s_dicts['l3_total_demand']
        else:
            s_dicts['last_one_need_cycle'][idx] = pcycle_dict[set_pn_state.positive_num]
            s_dicts['last_one_need_block'][idx] = pb_dict[set_pn_state.positive_num]
            s_dicts['last_one_need_hit'][idx] = ph_dict[set_pn_state.positive_num]

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
    perf_prefixs = ['90perf','95perf','full']
    draw_picformat = [
        (draw_one_workload_pnlast_cyclelen_hist,'pn_last_cyclelen_hist_{}.png',None),
        (draw_one_workload_pnlast_blocklen_hist,'pn_last_blocklen_hist_{}.png',None),
        (draw_one_workload_pnlast_hitlen_hist,'pn_last_hitlen_hist_{}.png',None),
        (draw_one_workload_pnlast_tti_change,'pn_last_tti_change_{}.png',None),
        (draw_one_workload_pnlast_tti_growthchange,'pn_last_tti_growchange_{}.png',None),
        (draw_one_workload_pnlast_growbucket,'pn_last_growbucket_{}.png',None),
    ]

    for perf_prefix in perf_prefixs:
        waydict_name = waydict_format.format(perf_prefix)
        waydict = use_conf[waydict_name]
        ret_dict = {}
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