from audioop import mul
from ctypes import sizeof
import os
import os.path as osp
import pandas as pd
import numpy as np
import utils.common as c
import utils.target_stats as t
import json
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# matplotlib.rcParams['font.size'] = 8.0
plt.style.use('fivethirtyeight')

def get_ipc(stat_path: str):
    targets = t.ipc_target
    stats = c.get_stats(stat_path, targets, insts=100*10**6, re_targets=True)
    return stats['ipc']

def single_stat_factory(targets, key, prefix=''):
    def get_single_stat(stat_path: str):
        # print(stat_path)
        if prefix == '':
            stats = c.get_stats(stat_path, targets, re_targets=True)
        else:
            assert prefix == 'xs_'
            stats = c.xs_get_stats(stat_path, targets, re_targets=True)
        if stats is not None:
            if key in stats:
                return stats[key]
            else:
                return 0
        else:
            return None
    return get_single_stat

def multi_stats_factory(targets, keys, insts: int=100*(10**6),prefix=''):
    def get_multi_stats(stat_path: str):
        # print(stat_path)
        if prefix == '':
            stats = c.get_stats(stat_path, targets, insts=insts, re_targets=True)
        else:
            assert prefix == 'xs_'
            stats = c.xs_get_stats(stat_path, targets, re_targets=True)
        if stats is not None:
            res = []
            for key in keys:
                if key in stats:
                    res.append(stats[key])
                else:
                    res.append(0)
            return res
        else:
            return None
    return get_multi_stats

def glob_stats_l2(path: str, fname = 'm5out/stats.txt'):
    stats_files = []
    for x in os.listdir(path):
        x_path = osp.join(path, x)
        if osp.isdir(x_path):
            for codename, f in glob_stats(x_path, fname):
                stats_files.append((f'{x}/{codename}', f))
    return stats_files

def glob_stats(path: str, fname = 'm5out/stats.txt'):
    stat_files = []
    for x in os.listdir(path):
        stat_path = osp.join(path, x, fname)
        if osp.isfile(stat_path) and osp.getsize(stat_path) > 10 * 1024:  # > 10K
            stat_files.append((x, stat_path))
    return stat_files

def glob_weighted_stats(path: str, get_func, filtered=True, stat_names=['ipc'],
        get_funcs=None,
        simpoints='/home51/zyy/expri_results/simpoints17.json',
        stat_file='m5out/stats.txt'):
    stat_tree = {}
    with open(simpoints) as jf:
        points = json.load(jf)
    print(points.keys())
    for workload in os.listdir(path):
        print('Extracting', workload)
        weight_pattern = re.compile(f'.*/{workload}_\d+_(\d+\.\d+([eE][-+]?\d+)?)/.*\.gz')
        workload_dir = osp.join(path, workload)
        bmk = workload.split('_')[0]
        if bmk not in stat_tree:
            stat_tree[bmk] = {}
        stat_tree[bmk][workload] = {}
        for point in os.listdir(workload_dir):
            if point not in points[workload]:
                continue
            point_dir = osp.join(workload_dir, point)
            stats_file = osp.join(point_dir, stat_file)
            weight = float(points[workload][str(point)])
            if get_funcs is not None:
                stat_list = [ f(stats_file) for f in get_funcs]
                if None not in stat_list:
                    stat_tree[bmk][workload][int(point)] = [weight] + stat_list
            else:
                stat = get_func(stats_file)
                if stat is not None:
                    stat_tree[bmk][workload][int(point)] = (weight, stat)
        if len(stat_tree[bmk][workload]):
            df = pd.DataFrame.from_dict(stat_tree[bmk][workload], orient='index')
            df.columns = ['weight'] + stat_names
            stat_tree[bmk][workload] = df
            print(df)
        else:
            stat_tree[bmk].pop(workload)
            if not len(stat_tree[bmk]):
                stat_tree.pop(bmk)

    return stat_tree


def glob_weighted_cpts(path: str):
    stat_tree = {}
    for dir_name in os.listdir(path):
        weight_pattern = re.compile(f'(\w+)_(\d+)_(\d+\.\d+([eE][-+]?\d+)?)')
        workload_dir = osp.join(path, dir_name)
        print(dir_name)
        m = weight_pattern.match(dir_name)
        if m is not None:
            workload = m.group(1)
            point = m.group(2)
            weight = float(m.group(3))
            bmk = workload.split('_')[0]
            if bmk not in stat_tree:
                stat_tree[bmk] = {}
            if workload not in stat_tree[bmk]:
                stat_tree[bmk][workload] = {}
            stat_tree[bmk][workload][int(point)] = weight
    for bmk in stat_tree:
        for workload in stat_tree[bmk]:
            item = stat_tree[bmk][workload]
            df = pd.DataFrame.from_dict(item, orient='index')
            df.columns = ['weight']
            stat_tree[bmk][workload] = df
    return stat_tree



def coveraged(coverage: float, df: pd.DataFrame):
    df = df.sort_values('weight', ascending=False)
    i = 0
    cummulated = 0.0
    for row in df.iterrows():
        cummulated += row[1]['weight']
        i += 1
        if cummulated > coverage:
            break
    df = df.iloc[:i]
    return(cummulated, df)


def weighted_cpi(df: pd.DataFrame):
    if 'cpi' in df.columns:
        return np.dot(df['weight'], df['cpi']) / np.sum(df['weight']), np.sum(df['weight'])
    else:
        assert 'ipc' in df.columns
        return np.dot(df['weight'], 1.0/df['ipc']) / np.sum(df['weight']), np.sum(df['weight'])

def weighted_one_stat(df: pd.DataFrame, stat_name):
    return np.dot(df['weight'], df[stat_name]) / np.sum(df['weight']), np.sum(df['weight'])        


def gen_json(path, output):
    tree = glob_weighted_cpts(path)
    d = {}
    count = 0
    expected_coverage = 0.5
    for bmk in tree:
        for workload in tree[bmk]:
            d[workload] = {}
            coverage, selected = coveraged(expected_coverage, tree[bmk][workload])
            weights = selected['weight']
            for row in weights.index:
                count += 1
                d[workload][int(row)] = weights[row]
            print(d[workload])
    print(f'{count} points in total')
    with open(f'{output}_cover{expected_coverage}.json', 'w') as f:
        json.dump(d, f, indent=4)


def draw_llc_access(stat_file='m5out/stats.txt', get_func=None, get_funcs=None, inst_step = 2*(10**6)):
    start_inst=50*(10**6)
    unique_list_dict = {}
    access_list_dict = {}
    nsamples = 2
    for i in range(nsamples):
        unique_get_func = multi_stats_factory(t.cache_set_targets,["l3.tags.slice_set_accesses_unique::"+str(i) for i in range(4096)],insts=start_inst+(i+1)*inst_step)
        unique_list_dict[i]= unique_get_func(stat_file)
        access_get_func = multi_stats_factory(t.cache_set_targets,["l3.tags.slice_set_accesses::"+str(i) for i in range(4096)],insts=start_inst+(i+1)*inst_step)
        access_list_dict[i]= access_get_func(stat_file)

    # print(sum(access_list_dict[1]))
    # return
    # print(unique_list)
    # print(len(unique_list))
    # print(access_list)
    # print(len(access_list))
    x=np.arange(4096)
    bar_width = 0.1

    fig=plt.figure(figsize=(24, 30))
    plt.title("LLC step")

    plt.xlim(0,4096)
    for i in range(nsamples):
        plt.subplot(2*nsamples, 1, i+1)
        plt.xlim((0,4096))
        plt.ylim((0,20))
        plt.yticks(np.arange(0,20,2))
        plt.title(f"access per set {i}")
        plt.bar(x=x,height=access_list_dict[i])
        plt.subplot(2*nsamples, 1, i+1+nsamples)
        plt.xlim((0,4096))
        plt.ylim((0,16))
        plt.title(f"unique access per set {i}")
        plt.bar(x=x,height=unique_list_dict[i])
    plt.tight_layout(pad=2,h_pad=2)
    plt.show()
    pass

def draw_l2_access(stat_file='m5out/stats.txt', get_func=None, get_funcs=None, inst_step = 2*(10**6)):
    start_inst=50*(10**6)
    unique_list_dict = {}
    access_list_dict = {}
    nsamples = 2
    for i in range(nsamples):
        unique_get_func = multi_stats_factory(t.cache_set_targets,["l2.tags.slice_set_accesses_unique::"+str(i) for i in range(1024)],insts=start_inst+(i+1)*inst_step)
        unique_list_dict[i]= unique_get_func(stat_file)
        access_get_func = multi_stats_factory(t.cache_set_targets,["l2.tags.slice_set_accesses::"+str(i) for i in range(1024)],insts=start_inst+(i+1)*inst_step)
        access_list_dict[i]= access_get_func(stat_file)
    # print(unique_list)
    # print(len(unique_list))
    # print(access_list)
    # print(len(access_list))
    x=np.arange(1024)
    bar_width = 0.1

    fig=plt.figure(figsize=(24, 30))
    plt.title("L2 step")

    plt.xlim(0,1024)
    for i in range(nsamples):
        plt.subplot(2*nsamples, 1, i+1)
        plt.xlim((0,1024))
        plt.title(f"access per set {i}")
        plt.bar(x=x,height=access_list_dict[i])
        plt.subplot(2*nsamples, 1, i+1+nsamples)
        plt.xlim((0,1024))
        plt.title(f"unique access per set {i}")
        plt.bar(x=x,height=unique_list_dict[i])
    plt.tight_layout(pad=2,h_pad=2)
    plt.show()
    pass


def draw_llc_cdf_access(stat_file='m5out/stats.txt', get_func=None, get_funcs=None, inst_step = 2*(10**6)):
    start_inst=50*(10**6)
    unique_list_dict = {}
    access_list_dict = {}
    nsamples = 1
    for i in range(nsamples):
        unique_get_func = multi_stats_factory(t.cache_set_targets,["l3.tags.slice_set_accesses_unique::"+str(i) for i in range(4096)],insts=start_inst+(i+1)*inst_step)
        unique_list_dict[i]= unique_get_func(stat_file)
        access_get_func = multi_stats_factory(t.cache_set_targets,["l3.tags.slice_set_accesses::"+str(i) for i in range(4096)],insts=start_inst+(i+1)*inst_step)
        access_list_dict[i]= access_get_func(stat_file)
    # print(unique_list)
    # print(len(unique_list))
    # print(access_list)
    # print(len(access_list))
    x=np.arange(4096)
    bar_width = 0.1

    fig=plt.figure(figsize=(10, 18))

    for i in range(nsamples):


        max_access = np.max(access_list_dict[i])
        count, bins_count = np.histogram(access_list_dict[i], bins=np.arange(max_access+1))
        pdf = count / sum(count)
        cdf = np.cumsum(pdf)
        plt.subplot(3*nsamples, 1, nsamples*i+1)
        plt.title(f"access per set histogram {i}")
        plt.xlabel('sets whose access number == i')
        plt.plot(bins_count[:-1], count)

        plt.subplot(3*nsamples, 1, nsamples*i+2)
        plt.title(f"access per set cdf {i}")
        plt.xlabel('sets whose access number <= i')
        plt.ylim(0,1.1)
        plt.plot(bins_count[:-1], cdf)

        plt.subplot(3*nsamples, 1, nsamples*i+3)
        plt.title(f"access cdf for accesses in different type of set {i}")
        plt.xlabel('accesses in set whose access number <= i')
        access_count = count * bins_count[:-1]
        print(f"total access: {sum(access_count)}")
        pdf = access_count / sum(access_count)
        cdf = np.cumsum(pdf)
        plt.ylim(0,1.1)
        plt.plot(bins_count[:-1], cdf)
        # plt.title(f"unique access per set {i}")
        # max_access = np.max(unique_list_dict[i])
        # count, bins_count = np.histogram(unique_list_dict[i], bins=max_access)
        # pdf = count / sum(count)
        # cdf = np.cumsum(pdf)
        # plt.plot(bins_count[1:], cdf)
    plt.tight_layout(pad=2,h_pad=2)
    plt.show()
    pass

def draw_llc_tb_ipc(stat_dir_prefix='/home/zcq/lvna/5g/ff-reshape/log/xal2-xal1-gcc-gcc2_1000_'):
    ipc_list_dict = {}
    inck = [4,8,16,32,64,128,256,512]
    for inc in inck:
        stat_file=stat_dir_prefix + str(inc) +'/stats.txt'
        ipcs_get_func = multi_stats_factory(t.standard_targets,["cpu"+str(i)+".ipc" for i in range(4)])
        ipc_list_dict[inc]= ipcs_get_func(stat_file)
    print(ipc_list_dict)
    # print(len(unique_list))
    # print(access_list)
    # print(len(access_list))

    plt.figure(figsize=(8, 6))
    x_axis = list(ipc_list_dict.keys())
    plt.xscale("log")

    for i in range(4):
        plt.plot(x_axis,[ ipcs[i] for ipcs in ipc_list_dict.values()] , label = f"cpu {i} ipc")

    plt.legend()
    plt.tight_layout(pad=2,h_pad=2)
    plt.show()
    pass

if __name__ == '__main__':
    # tree = glob_weighted_stats(
    #         '/home51/zyy/expri_results/omegaflow_spec17/of_g1_perf/',
    #         single_stat_factory(t.ipc_target, 'cpi')
    #         )
    # gen_json('/home51/zyy/expri_results/nemu_take_simpoint_cpt_06',
    #         '/home51/zyy/expri_results/simpoints06')

    # draw_llc_access(stat_file='/home/zcq/lvna/5g/ff-reshape/gcc_22850000000_set_out/stats.txt')
    # draw_llc_cdf_access(stat_file='/home/zcq/lvna/5g/ff-reshape/gcc_22850000000_set_out/stats.txt')
    draw_llc_tb_ipc()
    # draw_l2_access(stat_file='/home/zcq/lvna/5g/ff-reshape/xalancbmk_144250000000_0.153516_set_out/stats.txt',inst_step = 350000)
    # draw_llc_access(stat_file='/home/zcq/lvna/5g/ff-reshape/xalancbmk_144250000000_0.153516_set_out/stats.txt',inst_step = 350000)
