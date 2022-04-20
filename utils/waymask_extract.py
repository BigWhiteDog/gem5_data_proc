from cProfile import label
from functools import cmp_to_key
import os
import os.path as osp
from typing import List
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
    access_list_dict = {}
    nsamples = 2
    for id in range(nsamples):
        access_get_func = multi_stats_factory(t.cache_set_targets,["l3.tags.slice_set_accesses_"+str(id)+"::"+str(i) for i in range(4096)],insts=start_inst+5000)
        access_list_dict[id]= access_get_func(stat_file)

    # print(sum(access_list_dict[1]))
    # return
    # print(unique_list)
    # print(len(unique_list))
    # print(access_list)
    # print(len(access_list))
    x=np.arange(4096)
    bar_width = 0.1

    fig=plt.figure(figsize=(8, 6))
    plt.title("LLC access for different id")

    plt.xlim(0,4096)
    for i in range(nsamples):
        plt.subplot(nsamples, 1, i+1)
        plt.xlim((0,4096))
        plt.ylim((0,20))
        plt.yticks(np.arange(0,20,1))
        plt.title(f"access per set id{i}")
        plt.bar(x=x,height=access_list_dict[i])
    # plt.tight_layout(pad=2,h_pad=2)
    plt.show()
    pass

def compare_waymasks(x,y):
    xs = [ int(a,base=16) for a in x.split('-')]
    ys = [ int(a,base=16) for a in y.split('-')]
    assert(len(xs) == len(ys))
    for a,b in zip(xs,ys):
        if a < b:
            return -1
        elif a > b:
            return 1
    return 0

def draw_ipc_speedup(ax,log_dir: str, workloads: List[str]):
    start_inst=50*(10**6)
    mix_workload_name = "-".join(workloads)
    stats_dir = os.path.join(log_dir,mix_workload_name) #like log/mcf-sphinx3
    parts_name = os.listdir(stats_dir) #like 0x1-0xfe and nopart
    nsamples = len(workloads)
    ipc_dict = {}
    speed_up_dict = {}
    for w in workloads:
        ipc_dict[w] = {}
        speed_up_dict[w] = {}
    speed_up_dict['mix'] = {}
    for p in parts_name:
        ipc_get_func = c.multi_stats_lastn_factory(t.cache_set_targets,[f"cpu{i}.ipc" for i in range(nsamples)],last_n=20)
        st_file = os.path.join(stats_dir , p + "/stats.txt")
        ipcs_dict = ipc_get_func(st_file)
        ipcs = [np.average(ipcs_dict[f"cpu{i}.ipc"]) for i in range(nsamples)]
        for w,ipc in zip(workloads,ipcs):
            ipc_dict[w][p] = ipc
    for w in workloads:
        for p in parts_name:
            speed_up_dict[w][p] = ipc_dict[w][p] / ipc_dict[w]['nopart']
    for p in parts_name:
        speed_up_dict['mix'][p] = np.average([speed_up_dict[w][p] for w in workloads])

    xlabels = workloads + ['mix'] #workloads+mix
    # other_labels = list(filter(lambda x: not x.startswith('0'), parts_name)) #all nopart , qos ...
    other_labels = ['nopart']
    color_labels = sorted(list(filter(lambda x: x.startswith('0'), parts_name)), key=cmp_to_key(compare_waymasks)) #all 0x1-0xfe ... sorted
    color_labels = other_labels + color_labels # nopart, qos, 0x1-0xfe, ...


    x=np.arange(len(xlabels))
    bar_width = 0.05

    # fig=plt.figure(figsize=(8, 6))
    # fig, ax = plt.subplots()
    rec_list = []
    bar_offset = -0.2
    for l in color_labels:
        tmp_yl = []
        for w in xlabels:
            tmp_yl.append(speed_up_dict[w][l])
        r = ax.bar(x+bar_offset, tmp_yl, bar_width, label = l)
        rec_list.append(r)
        bar_offset += bar_width
    ax.set_ylim(0.9,1.1)
    ax.set_xticks(x,xlabels)
    ax.set_title("speed up for different strategy")
    ax.set_ylabel('Performance over NoPart')

    # ax.legend()

    return ax

if __name__ == '__main__':
    # draw_llc_access(stat_file='/nfs/home/zhangchuanqi/lvna/5g/ff-reshape/log/mcf-sphinx3_/stats.txt')
    # draw_llc_access(stat_file='/nfs/home/zhangchuanqi/lvna/5g/ff-reshape/log/omnetpp-sphinx3_/stats.txt')
    # draw_ipc_speedup(stat_dir='/nfs/home/zhangchuanqi/lvna/5g/ff-reshape/log',workloads=['omnetpp','sphinx3'])
    # draw_ipc_speedup(log_dir='/nfs/home/zhangchuanqi/lvna/5g/ff-reshape/log/10M',workloads=['mcf','omnetpp'])
    # draw_ipc_speedup(log_dir='/nfs/home/zhangchuanqi/lvna/5g/ff-reshape/log/10M',workloads=['mcf','xalancbmk'])
    # draw_ipc_speedup(log_dir='/nfs/home/zhangchuanqi/lvna/5g/ff-reshape/log/10M',workloads=['mcf','sphinx3'])
    fig, ax = plt.subplots(3,2)
    draw_ipc_speedup(ax[0][0],log_dir='/nfs/home/zhangchuanqi/lvna/5g/ff-reshape/log/40x1M',workloads=['mcf','omnetpp'])
    draw_ipc_speedup(ax[1][0],log_dir='/nfs/home/zhangchuanqi/lvna/5g/ff-reshape/log/40x1M',workloads=['mcf','xalancbmk'])
    draw_ipc_speedup(ax[2][0],log_dir='/nfs/home/zhangchuanqi/lvna/5g/ff-reshape/log/40x1M',workloads=['mcf','sphinx3'])
    draw_ipc_speedup(ax[0][1],log_dir='/nfs/home/zhangchuanqi/lvna/5g/ff-reshape/log/40x1M',workloads=['omnetpp','xalancbmk'])
    draw_ipc_speedup(ax[1][1],log_dir='/nfs/home/zhangchuanqi/lvna/5g/ff-reshape/log/40x1M',workloads=['omnetpp','sphinx3'])
    draw_ipc_speedup(ax[2][1],log_dir='/nfs/home/zhangchuanqi/lvna/5g/ff-reshape/log/40x1M',workloads=['xalancbmk','sphinx3'])
    plt.show()
    # draw_llc_tb_ipc()
    # draw_l2_access(stat_file='/home/zcq/lvna/5g/ff-reshape/xalancbmk_144250000000_0.153516_set_out/stats.txt',inst_step = 350000)
    # draw_llc_access(stat_file='/home/zcq/lvna/5g/ff-reshape/xalancbmk_144250000000_0.153516_set_out/stats.txt',inst_step = 350000)
