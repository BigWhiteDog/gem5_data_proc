import numpy as np
import utils.common as c
from utils.common import multi_stats_lastn_factory
import utils.target_stats as t
import csv
import numpy as np
import argparse

import json

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib import ticker


def insert_avg_dict(s_dicts:dict,ncore=4,last_nsamples=8,stage_len=4):
    nstage = last_nsamples // stage_len
    for i in range(ncore):
        for j in range(nstage):
            indexs = list(range(j*stage_len+1,(j+1)*stage_len))
            s_dicts[f'avg_ipc{i}_stage{j}'] = np.mean(
                [s_dicts[f'cpu{i}.ipc'][x] for x in indexs])
            s_dicts[f'avg_misses{i}_stage{j}'] = np.mean(
                [s_dicts[f'l3.demand_misses::.cpu{i}'][x] for x in indexs])
            s_dicts[f'avg_mpki{i}_stage{j}'] = np.mean(
                [s_dicts[f'l3.demand_misses::.cpu{i}'][x]/ 
                s_dicts[f'cpu{i}.committedInsts'][x]*1000 for x in indexs])



    