#!/usr/bin/env python3

import pandas as pd
import os

from utils import common as c

def handle_one_path(target_dir:str,
    ncore=4,last_nsamples=8,
    stage_len=4):

    print('=====================')
    config = target_dir.rsplit('/',3)[-2:]
    print(f'conf: {config}')
    print('=====================')
    s = c.extract_samples_raw_json(target_dir, ncore, last_nsamples)
    # iad.insert_avg_dict(s,last_nsamples=last_nsamples)
    
    nstage = last_nsamples // stage_len


    for j in range(nstage):
        print('=====================')
        print(f'stage {j}')
        print('=====================')
        print('')
        indexs = list(range(j*stage_len+1,(j+1)*stage_len))
        ipc_dict = {}
        cycle_dict = {}
        mpki_dict = {}
        misses_dict = {}
        all_dicts = [ipc_dict,cycle_dict,mpki_dict,misses_dict]
        # all_dicts = [ipc_dict,cycle_dict,mpki_dict]
        # all_dicts = [ipc_dict,cycle_dict,misses_dict]
        big_dict = {}
        for i in range(ncore):
            ipc_dict[f'cpu{i}.ipc'] = [ s[f'cpu{i}.ipc'][x] for x in indexs]
            cycle_dict[f'cpu{i}.cycle'] = [s[f'cpu{i}.numCycles'][x] for x in indexs]
            misses_dict[f'l3_misses{i}'] = [s[f'l3.demand_misses::.cpu{i}'][x] for x in indexs]
            mpki_dict[f'l3_mpki{i}'] = [s[f'l3_mpki_cpu{i}'][x] for x in indexs]
        for i,d in enumerate(all_dicts):
            df = pd.DataFrame(d,index=indexs)
            # df.loc['avg'] = df.mean()
            if d == cycle_dict or d == misses_dict:
                df.style.format('{:d}')
                pass
            else:
                df.style.format(precision=5)
            # df.to_csv(f'df{i}.csv')
            # print(df.to_string(index=False))
        for d in all_dicts:
            big_dict.update(d)
        df = pd.DataFrame(big_dict,index=indexs)
        df.loc['avg'] = df.mean()
        df.to_csv(f'c-{"-".join(config)}-stage{j}.csv',index=True,sep='\t')
        # print(df.to_string(index=False))
        # mean_df = pd.DataFrame(columns=df.columns)
        # mean_df.loc['avg'] = df.mean()
        # print(mean_df.to_string(index=False,header=False))
        print('')
    with open(f'c-{"-".join(config)}.csv','w') as outfile:
        print('=====================',file=outfile)
        print(f'conf: {config}',file=outfile)
        print('=====================',file=outfile)
        csv_names = [f'c-{"-".join(config)}-stage{j}.csv' for j in range(nstage)]
        for fname in csv_names:
            print('\n\n',file=outfile)
            with open(fname) as infile:
                outfile.write(infile.read())
            os.remove(fname)

if __name__ == '__main__':
    t_paths = []
    t_paths.append('/nfs/home/zhangchuanqi/lvna/5g/ff-reshape/log/new_hw_test/period_hmmer_o3_0-period_hmmer_o3_3-period_hmmer_o2_0-period_hmmer_o2_2/2560kBLLC/9tti/try-waymask/l3-nopart/l2-nopart')
    t_paths.append('/nfs/home/zhangchuanqi/lvna/5g/ff-reshape/log/new_hw_test/period_hmmer_o3_0-period_hmmer_o3_3-period_hmmer_o2_0-period_hmmer_o2_2/2560kBLLC/9tti/try-waymask/l3-5/l2-nopart')
    t_paths.append('/nfs/home/zhangchuanqi/lvna/5g/ff-reshape/log/new_hw_test/period_hmmer_o3_0-period_hmmer_o3_3-period_hmmer_o2_0-period_hmmer_o2_2/2560kBLLC/9tti/try-tb/l3-nopart/l2-nopart/l3-tb-1024-1')
    t_paths.append('/nfs/home/zhangchuanqi/lvna/5g/ff-reshape/log/new_hw_test/period_hmmer_o3_0-period_hmmer_o3_3-period_hmmer_o2_0-period_hmmer_o2_2/2560kBLLC/9tti/try-waymask/l3-9/hot-0.95')

    for p in t_paths:
        handle_one_path(p)