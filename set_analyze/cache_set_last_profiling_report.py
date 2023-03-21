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

opt = parser.parse_args()

from cache_sensitive_names import *
from set_analyze.my_diff_color import *

all_set = 16384
# full_ass = 8
tail_set = int(0.001*all_set)

def outputcsv_minway_fromdb(csv_top_dir,work,s_dicts):
    csv_file = os.path.join(csv_top_dir,f'{work}.csv')
    with open(csv_file,'w') as f:
        for setneed in s_dicts['min_ways_no_extra_miss']:
            f.write(f'{setneed}\n')


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
        tti_len = 1_000_000

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

        cur.close()
    s_dicts['est_used_ways'] = [0 for _ in range(all_set)]
    s_dicts['last_one_need_cycle'] = [0 for _ in range(all_set)]
    s_dicts['last_one_need_block'] = [0 for _ in range(all_set)]
    s_dicts['tti_record'] = tti_hitblock_l

    for idx in range(all_set):
        set_pn_state = pn_states[idx]
        s_dicts['est_used_ways'][idx] = max(set_pn_state.positive_num,1)
        pcycle_dict = set_pn_state.positive_cyclelen_record
        pb_dict = set_pn_state.positive_total_blocklen
        if set_pn_state.positive_num == pn_start_positive:
            s_dicts['last_one_need_cycle'][idx] = s_dicts['cpu_num_cycles']
            s_dicts['last_one_need_block'][idx] = s_dicts['l3_total_demand']
        else:
            s_dicts['last_one_need_cycle'][idx] = pcycle_dict[set_pn_state.positive_num]
            s_dicts['last_one_need_block'][idx] = pb_dict[set_pn_state.positive_num]

    work_stats_dict[work] = s_dicts


def outputcsv_by_func(base_dir,worksname_waydict,analyze_func,output_csv_func,csv_top_dir,input_stats_dict):

    work_stats_dict = input_stats_dict

    for i,work in enumerate(worksname_waydict):
        full_ass = worksname_waydict[work]
        work_dir = os.path.join(base_dir,work)
        if not os.path.isdir(work_dir):
            continue
        analyze_func(work_stats_dict,work,work_dir,full_ass)
        s_dicts = work_stats_dict[work]
        output_csv_func(csv_top_dir,work,s_dicts)

    return work_stats_dict

if __name__ == '__main__':
    use_conf = conf_50M
    test_prefix = use_conf['test_prefix']
    base_dir = base_dir_format.format(test_prefix)
    csv_dir_path = f'set_analyze/{test_prefix}other/csv'
    os.makedirs(csv_dir_path, exist_ok=True)
    worksname = use_conf['cache_work_names'] #like mcf

    n_works = len(worksname)
    n_rows = math.ceil(n_works/4)

    waydict_format = 'cache_work_{}ways'
    perf_prefixs = ['90perf','95perf','full']
    # perf_prefixs = ['95perf']
    csvfunc_dir_pair = [
        (outputcsv_minway_fromdb,os.path.join(csv_dir_path,'lastway_hashed_{}')),
        # (draw_one_workload_pn_blocklen,'pn_est_blocklen_contour_{}.png'),
        # (draw_one_workload_pn_cyclelen,'pn_est_cyclelen_contour_{}.png'),
    ]

    for perf_prefix in perf_prefixs:
        waydict_name = waydict_format.format(perf_prefix)
        waydict = use_conf[waydict_name]
        ret_dict = {}
        for csvfunc,csv_dir_format in csvfunc_dir_pair:
            csv_dir = csv_dir_format.format(perf_prefix)
            os.makedirs(csv_dir, exist_ok=True)
            outputcsv_by_func(base_dir,waydict,
            analyze_func=analyze_minway_fromdb,
            output_csv_func=csvfunc,
            csv_top_dir=csv_dir,
            input_stats_dict=ret_dict)