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

db_path = '/nfs/home/zhangchuanqi/lvna/for_xs/catlog/single-profiling/sphinx3/l3-8/hm.db'
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

cnt_array = []
set_cnt = {}
for idx,stamp in out_res:
    # if idx in max_setidx_cnt or idx in min_setidx_cnt:
    #     continue
    if idx not in set_cnt:
        set_cnt[idx] = 1
    else:
        set_cnt[idx] += 1
cnt_array = [x for x in set_cnt.values()]
# %%
f = Fitter(cnt_array, distributions=fitter.get_common_distributions())
# f = Fitter(cnt_array)
# f = Fitter(access_interval)
f.fit()
# %%
f.summary()
# f.get_best()

# %%
