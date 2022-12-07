import os
import numpy as np
import argparse
import re
import sqlite3
from fitter import Fitter
import fitter

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib import ticker

db_path = '/nfs/home/zhangchuanqi/lvna/for_xs/catlog/single-profiling/xalancbmk/l3-8/hm.db'
con = sqlite3.connect(db_path)
cur = con.cursor()

all_set = 16384

all_access_query = 'SELECT SETIDX,STAMP FROM HitMissTrace'
f = cur.execute(all_access_query)
out_res = [x for x in f]

inter_array = []
set_last_time = {}
for idx,stamp in out_res:
    # if idx in max_setidx_cnt or idx in min_setidx_cnt:
    #     continue
    if idx not in set_last_time:
        set_last_time[idx] = stamp
    else:
        # if stamp - set_last_time[idx] < interval:
        #     inter_array.append(stamp - set_last_time[idx])
        inter_array.append(stamp - set_last_time[idx])
        set_last_time[idx] = stamp
f = Fitter(inter_array, distributions=fitter.get_common_distributions())
f.fit()

f.summary()
f.get_best()

