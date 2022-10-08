from math import ceil
import os
import os.path as osp
import numpy as np
import utils.common as c
from utils.common import multi_stats_lastn_factory
import utils.target_stats as t

import numpy as np
import argparse

import json

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="options to get set stats")
parser.add_argument('-d','--stats_dir', type=str,
    help='stats dir to analyze',required=True)
parser.add_argument('--ids',default=4,type=int)
parser.add_argument('--nsamples',default=4,type=int)
parser.add_argument('--l3_sets',default=4096,type=int)

opt = parser.parse_args()

def extract_set_trace(stats_dir,n_id,last_nsamples,n_l3_sets):
    set_access_dict = {}
    st_file = os.path.join(stats_dir,"stats.txt")
    out_dir_top = os.path.join(stats_dir,"set_est")
    if not os.path.exists(out_dir_top):
        os.makedirs(out_dir_top, exist_ok=True)
    target_keys = []
    for i in range(n_id):
        target_keys.extend([f"l3.tags.slice_set_accesses_{i}::{x}" for x in range(n_l3_sets)])
    sets_get_func = multi_stats_lastn_factory(t.cache_set_targets, target_keys,last_n=last_nsamples)
    set_access_dict = sets_get_func(st_file)
    set_id_dict = {}
    for i in range(n_id):
        tmp_2dlist = [set_access_dict[f"l3.tags.slice_set_accesses_{i}::{x}"] for x in range(n_l3_sets)]
        tmp_2darr = np.array(tmp_2dlist).transpose(1,0)
        res_out = []
        out_file = os.path.join(out_dir_top,f"{i}.json")
        for c in range(last_nsamples):
            res_out.append(tmp_2darr[c].tolist())
        set_id_dict[i] = res_out
        with open(out_file,"w") as out_f:
            json.dump(res_out,out_f,indent=4)
    return set_id_dict

def draw_llc_access(set_id_dict,n_id,last_nsamples,n_l3_sets):
    fig, ax = plt.subplots(2,ceil(n_id/2))
    fig.suptitle("LLC access for different id")
    for i in range(n_id):
        fy = i % 2
        fx = i // 2
        sax = ax[fx,fy]
        sax.set_title(f"set accesses PDF for ID {i}")
        sax.set_ylim(0,0.5)
        sets_accesnums = set_id_dict[i][0]
        sax.hist(sets_accesnums,bins=np.arange(0,32,1),density=True)
    plt.show()
    pass

if __name__ == '__main__':
    res_dict = extract_set_trace(opt.stats_dir,n_id=opt.ids,last_nsamples=opt.nsamples,n_l3_sets=opt.l3_sets)
    draw_llc_access(res_dict,n_id=opt.ids,last_nsamples=opt.nsamples,n_l3_sets=opt.l3_sets)