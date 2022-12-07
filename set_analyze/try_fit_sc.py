# %%
import os
import numpy as np
import numpy as np
import argparse
import re
import sqlite3

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib import ticker

db_path = '/nfs/home/zhangchuanqi/lvna/for_xs/catlog/single-profiling/xalancbmk/l3-8/hm.db'
con = sqlite3.connect(db_path)
cur = con.cursor()

# %%
max_cnt10_query = "SELECT SETIDX,count(*) FROM HitMissTrace group by SETIDX ORDER BY count(*) DESC LIMIT 10"

f = cur.execute(max_cnt10_query)
max_setidx_cnt = [x for x in f]
print(max_setidx_cnt)

# %%
max_seti = max_setidx_cnt[0][0]
max_seti_cnt = max_setidx_cnt[0][1]


# %%
def get_seti_hitmiss(cur,seti):
    query = f"SELECT ISMISS FROM HitMissTrace WHERE SETIDX={seti}"
    f = cur.execute(query)
    cnt_list = [x[0] for x in f]
    return cnt_list


# %%
all_set = 16384
class SaturatedInteger:
    def __init__(self, val, lo, hi):
        self.real, self.lo, self.hi = val, lo, hi

    def __add__(self, other):
        return min(self.real + other.real, self.hi)

    def __sub__(self, other):
        return max(self.real - other.real, self.lo)

    def isSaturated(self):
        return self.real == self.hi
    def isLowSaturated(self):
        return self.real == self.lo

    def isHighHalf(self):
        return self.real > (self.hi + self.lo)/2
    def idLowHalf(self):
        return self.real <= (self.hi + self.lo)/2

    def isHighPortion(self,portion):
        return self.real > self.hi*portion + self.lo*(1-portion)
    def isLowPortion(self,portion):
        return self.real <= self.hi*portion + self.lo*(1-portion)

    def __get__(self):
        return self.real
    def __set__(self, val):
        if val > self.hi:
            self.real = self.hi
        elif val < self.lo:
            self.real = self.lo
        else:
            self.real = val


