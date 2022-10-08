import os
import os.path as osp
import numpy as np
import utils.common as c
from utils.mspecs_set_extract import extract_set_trace
import utils.target_stats as t

import numpy as np
import argparse

import json

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="options to get set stats")
parser.add_argument('-d','--stats_dir', type=str,
    help='top of stats dir to analyze',required=True)
parser.add_argument('--ids',default=1,type=int)
parser.add_argument('--nsamples',default=40,type=int)
parser.add_argument('--l3_sets',default=4096,type=int)

opt = parser.parse_args()

def draw_self_llc_access(myax,set_id_dict,last_nsamples,n_l3_sets):
        sax.set_ylim(0,0.5)
        sets_accesnums = set_id_dict[i][0]
        np.histogram(sets_accesnums,bins=np.arange(0,256,1),density=True)
        sax.hist()

if __name__ == '__main__':
    works = ['mcf','omnetpp','sphinx3','xalancbmk']
    fig,ax = plt.subplots(2,2)
    fig.suptitle("LLC set access PDF for different workload")
    for i,w in enumerate(works):
        fy = i % 2
        fx = i // 2
        sax = ax[fx,fy]
        sax.set_title(f"set accesses PDF for {w}")
        stats_dir = os.path.join(opt.stats_dir,w)
        set_id_dict = extract_set_trace(stats_dir,1,opt.nsamples,opt.l3_sets)
        draw_self_llc_access(ax[i],set_id_dict,opt.nsamples,opt.l3_sets)
    plt.show()