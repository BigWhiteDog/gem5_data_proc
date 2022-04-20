import os
import sys
import os.path as osp
import numpy as np
import utils.common as c
from utils.common import multi_stats_lastn_factory
import utils.target_stats as t

import numpy as np

def extract_set_trace(stats_dir,n_id = 2,last_nsamples = 40,n_l3_sets = 4096):
    set_access_dict = {}
    st_file = os.path.join(stats_dir,"stats.txt")
    out_dir_top = os.path.join(stats_dir,"set_est")
    if not os.path.exists(out_dir_top):
        os.makedirs(out_dir_top, exist_ok=True)
    out_dirs=[]
    target_keys = []
    for i in range(n_id):
        tmp_path = os.path.join(out_dir_top,str(i))
        os.makedirs(tmp_path, exist_ok=True)
        out_dirs.append(tmp_path)
        target_keys.extend([f"l3.tags.slice_set_accesses_{i}::{x}" for x in range(n_l3_sets)])
    sets_get_func = multi_stats_lastn_factory(t.cache_set_targets, target_keys,last_n=last_nsamples)
    set_access_dict = sets_get_func(st_file)
    for i in range(n_id):
        tmp_2dlist = [set_access_dict[f"l3.tags.slice_set_accesses_{i}::{x}"] for x in range(n_l3_sets)]
        tmp_2darr = np.array(tmp_2dlist).transpose(1,0)
        for c in range(last_nsamples):
            out_file = os.path.join(out_dirs[i],f"{c}.csv")
            with open(out_file,"w") as out_f:
                for num in tmp_2darr[c]:
                    out_f.write(f"{num}\n")

if __name__ == '__main__':
    extract_set_trace(sys.argv[1],n_id=16,last_nsamples=int(sys.argv[2]))