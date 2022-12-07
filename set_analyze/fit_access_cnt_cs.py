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
from cache_sensitive_names import *

works = cache_work_names
db_path_format = '/nfs/home/zhangchuanqi/lvna/for_xs/catlog/single-profiling/{}/l3-8/hm.db'
work = 'gcc_g23'
con = sqlite3.connect(db_path_format.format(work))
cur = con.cursor()
all_set = 16384
tail_set = int(0.001*all_set)

all_access_query = 'SELECT count(*) FROM HitMissTrace group by SETIDX'
f = cur.execute(all_access_query)
out_res = [x[0] for x in f]
out_res.sort()
out_res = out_res[tail_set:-tail_set]

# norm_list = list(filter(lambda x: 'norm' in x, fitter.get_distributions()))
# fit_list = fitter.get_common_distributions() + norm_list
fit_list = fitter.get_common_distributions()
# fit_list = fitter.get_distributions()

# f = Fitter(out_res,xmax=1000, distributions=fitter.get_common_distributions()+norm_list)
f = Fitter(out_res, distributions=fit_list)
# f = Fitter(out_res,xmax=1000)
# f = Fitter(out_res)
# f = Fitter(access_interval)
f.fit()
print(work)
# print(f.summary(plot=True))
f.summary()
# f.get_best()

# %%