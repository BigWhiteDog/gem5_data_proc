# %%
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

drop_cnt = 50

max_cnts_query = f"SELECT SETIDX,count(*) FROM HitMissTrace "\
    f"group by SETIDX ORDER BY count(*) DESC LIMIT {drop_cnt}"

f = cur.execute(max_cnts_query)
max_setidx_cnt = [x[0] for x in f]
print(max_setidx_cnt)

min_cnts_query = "SELECT SETIDX,count(*) FROM HitMissTrace "\
    f"group by SETIDX ORDER BY count(*) ASC LIMIT {drop_cnt}"

f = cur.execute(min_cnts_query)
min_setidx_cnt = [x[0] for x in f]
print(min_setidx_cnt)

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
# %%
f = Fitter(inter_array,xmax=100000, distributions=fitter.get_common_distributions())
# f = Fitter(access_interval)
f.fit()
# %%
f.summary()
f.get_best()

# %%
