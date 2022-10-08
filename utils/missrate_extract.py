from collections import OrderedDict
import os
import numpy as np
import utils.common as c
import utils.target_stats as t
import numpy as np
import argparse
import random

from utils.common import extract_samples_raw_json

import json

json_path = "/nfs/home/zhangchuanqi/lvna/5g/DirtyStuff/resources/simpoint_cpt_desc/hwfinal.json"

parser = argparse.ArgumentParser(description="options to get set stats")
# parser.add_argument('-d','--stats_dir', type=str,
#     help='stats dir to analyze',required=True)
# parser.add_argument('--ids',default=16,type=int)
# parser.add_argument('--nsamples',default=2,type=int)
# parser.add_argument('--l3_sets',default=4096,type=int)

opt = parser.parse_args()

def report_least_miss(worknames,base_dir_format,last_nsamples = 4):
    s_dicts = {}
    out_dict = {}
    for w in worknames:
        base_dir = base_dir_format.format(w)
        st_filter_file = os.path.join(base_dir,f"{last_nsamples}period.json")
        s_dicts[w] = extract_samples_raw_json(base_dir)
        total_hits = np.sum(s_dicts[w]['l3.demand_hits'])
        total_misses = np.sum(s_dicts[w]['l3.demand_misses'])
        total_miss_rate = total_misses / (total_hits + total_misses)
        out_dict[w] = OrderedDict()
        out_dict[w]['total_miss_rate'] = total_miss_rate
        out_dict[w].update(s_dicts[w])
    json.dump(out_dict,open('/nfs/home/zhangchuanqi/lvna/5g/gem5_data_proc/hw_stats_out_dir/4M_period_hmmer_miss.json',"w"),indent=4)
    out_sort_w = sorted(out_dict.items(),key=lambda x:x[1]['total_miss_rate'])
    return out_sort_w
        
if __name__ == '__main__':
    # worknames = ['hmmer_o30','hmmer_o31','hmmer_o2_retro0','hmmer_o2_retro1','hmmer0','hmmer1','hmmer2']
    # worknames = [f'spa{i}' for i in range(2)] + \
    #             [f'uaa{i}' for i in range(2)] + \
    #             [f'epa{i}' for i in range(3)] + \
    #             [f'cga{i}' for i in range(1)]
    # worknames = [f'isa{i}' for i in range(4)] + \
    #             [f'mga{i}' for i in range(1)] + \
    #             [f'bta{i}' for i in range(3)]
    # worknames = [f'lua{i}' for i in range(1)] + \
    #             [f'fta{i}' for i in range(3)]
    # ['spa0','spa1','uaa0','uaa1','epa0','epa1','bta0','lua0','fta0']
    # with open(json_path) as json_file:
    #     workload_dict = json.load(json_file)
    # task_loads = [f'hmmer_o2_retro{i}' for i in range(2)] + [f'hmmer_o3{i}' for i in range(2)] + [f'hmmer{i}' for i in range(3)]
    # task_loads = [f'hmmer_o3{i}' for i in range(2)] + [f'hmmer{i}' for i in range(3)]
    # full_worknames = [ '-'.join(e) for e in itertools.permutations(task_loads,4)]
    base_dir_top = '/nfs/home/zhangchuanqi/lvna/5g/ff-reshape/log/new_hw_test/period4/4MLLC_native'
    works = filter(lambda x: '-' in x ,os.listdir(base_dir_top))
    out_sort_w = report_least_miss(works,'{}/{{}}'.format(base_dir_top),last_nsamples=4)
    # with open('/nfs/home/zhangchuanqi/lvna/5g/gem5_data_proc/hw_stats_out_dir/period_hmmer_miss.json',"r") as f:
        # out_dict = json.load(f)
    # out_sort_w = sorted(out_dict.items(),key=lambda x:x[1]['total_miss_rate'])
    new_dict = {}
    for i in range(16):
        k,v = out_sort_w[i]
        print(k,v['total_miss_rate'])

    # t_work_combine = ['hmmer_o31-hmmer0-hmmer_o30-hmmer1']
    # report_least_miss(t_work_combine,'/nfs/home/zhangchuanqi/lvna/5g/ff-reshape/log/16M/{0}/100000')