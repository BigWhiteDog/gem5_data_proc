import pandas as pd
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 500)
import utils as u
import utils.common as c
import utils.target_stats as t
import json
import numpy as np
import sys
import os.path as osp
from scipy.stats import gmean,tstd,tmean
from statistics import geometric_mean
import argparse
import re

simpoints17 = '/home51/zyy/expri_results/simpoints.json'

simpoints_file = {
        'hw17': 'resources/simpoint_cpt_desc/hw17.json',
        '17': '/nfs-nvme/home/share/checkpoints_profiles/spec17_rv64gc_o2_50m/simpoint_summary.json',
        '06': '/nfs-nvme/home/share/checkpoints_profiles/spec06_rv64gc_o2_50m/simpoint_summary.json',
        }

def gen_coverage():
    tree = u.glob_weighted_stats(
            '/home51/zyy/expri_results/omegaflow_spec17/of_g1_perf/',
            u.single_stat_factory(t.ipc_target, 'cpi')
            )
    d = {}
    for bmk in tree:
        for workload in tree[bmk]:
            d[workload] = {}
            coverage, selected = u.coveraged(0.8, tree[bmk][workload])
            print(coverage)
            weights = selected['weight']
            for row in weights.index:
                d[workload][int(row)] = weights[row]
            print(d[workload])
    with open(simpoints, 'w') as f:
        json.dump(d, f, indent=4)


def get_insts(fname: str):
    assert osp.isfile(fname)
    p = re.compile('total guest instructions = (\d+)')
    with open(fname) as f:
        for line in f:
            m = p.search(line)
            if m is not None:
                return m.group(1)
    return None


def compute_weighted_cpi(ver, confs, base, simpoints, prefix, insts_file_fmt, stat_file,
        clock_rate, min_coverage=0.0, blacklist=[], whitelist=[], merge_benckmark=False):
    target = eval(f't.{prefix}ipc_target')
    workload_dict = {}
    bmk_stat = {}
    for conf, file_path in confs.items():
        workload_dict[conf] = {}
        bmk_stat[conf] = {}
        tree = u.glob_weighted_stats(
                file_path,None,
                get_funcs=[
                    lambda file_path: u.single_stat_factory(target, 'ipc', prefix)(file_path),
                    lambda file_path: u.single_stat_factory(target, 'Insts', prefix)(file_path),
                ],
                stat_names=["ipc","insts"],
                simpoints=simpoints,
                stat_file=stat_file,
                )
        with open(simpoints) as jf:
            js = json.load(jf)
            print(js.keys())
        times = {}
        for bmk in tree:
            if bmk in blacklist:
                continue
            if len(whitelist) and bmk not in whitelist:
                continue
            cpis = []
            weights = []
            time = 0
            coverage = 0
            count = 0
            for workload, df in tree[bmk].items():
                selected = dict(js[workload])
                keys = [int(x) for x in selected]
                keys = [x for x in keys if x in df.index]
                df = df.loc[keys]
                df = df.drop(df[df['insts'] < 10_000_000].index)
                cpi, weight = u.weighted_cpi(df)
                weights.append(weight)

                workload_dict[conf][workload] = {}
                workload_dict[conf][workload]['CPI'] = cpi
                workload_dict[conf][workload]['IPC'] = 1.0/cpi
                workload_dict[conf][workload]['Coverage'] = weight

                # merge multiple sub-items of a benchmark
                if merge_benckmark:
                    insts_file = insts_file_fmt.format(ver, workload)
                    insts = int(get_insts(insts_file))
                    workload_dict[conf][workload]['TotalInst'] = insts
                    workload_dict[conf][workload]['PredictedCycles'] = insts*cpi
                    seconds = insts*cpi / clock_rate
                    workload_dict[conf][workload]['PredictedSeconds'] = seconds
                    time += seconds
                    coverage += weight
                    count += 1

            if merge_benckmark:
                bmk_stat[conf][bmk] = {}
                bmk_stat[conf][bmk]['time'] = time
                ref_time = c.get_spec_ref_time(bmk, ver)
                assert ref_time is not None
                bmk_stat[conf][bmk]['ref_time'] = ref_time
                bmk_stat[conf][bmk]['score'] = ref_time / time
                bmk_stat[conf][bmk]['Coverage'] = coverage/count

    for conf in confs:
        print(conf, '='*60)
        df = pd.DataFrame.from_dict(workload_dict[conf], orient='index')
        workload_dict[conf] = df
        print(df)

        if merge_benckmark:
            df = pd.DataFrame.from_dict(bmk_stat[conf], orient='index')
            bmk_stat[conf] = df
            excluded = df[df['Coverage'] <= min_coverage]
            df = df[df['Coverage'] > min_coverage]
            print(df)
            df.to_csv(f'{conf}_bmk_ipc.csv',sep='\t')
            print('Estimated score @ 1.5GHz:', geometric_mean(df['score']))
            print('Estimated score per GHz:', geometric_mean(df['score'])/(clock_rate/(10**9)))
            print('Excluded because of low coverage:', list(excluded.index))


    tests = []
    conf_name = []
    for conf in confs.keys():
        if conf == base:
            continue
        rel = workload_dict[conf]['IPC']/workload_dict[base]['IPC']
        print(f'{conf}  Mean relative performance: {gmean(rel)}')
        tests.append(rel)
        conf_name.append(conf)
    if len(tests):
        dfx = pd.concat(tests, axis=1)
        dfx.columns = conf_name
        print('Relative performance:')
        print(dfx)
        dfx.to_csv(f'Nanhu_bmk_ipc_rel.csv',sep='\t')

def compute_best_weighted_cpi(ver, confs, base, simpoints, prefix, insts_file_fmt, stat_file,
        clock_rate, min_coverage=0.0, blacklist=[], whitelist=[], merge_benckmark=False):
    target = eval(f't.{prefix}ipc_target')
    workload_dict = {}
    bmk_stat = {}
    df_dict = {}
    for conf, file_path in confs.items():
        workload_dict[conf] = {}
        bmk_stat[conf] = {}
        df_dict[conf] = {}
        tree = u.glob_weighted_stats(
                file_path,
                u.single_stat_factory(target, 'ipc', prefix),
                simpoints=simpoints,
                stat_file=stat_file,
                )
        with open(simpoints) as jf:
            js = json.load(jf)
            print(js.keys())
        times = {}
        for bmk in tree:
            if bmk in blacklist:
                continue
            if len(whitelist) and bmk not in whitelist:
                continue
            cpis = []
            weights = []
            time = 0
            coverage = 0
            count = 0
            for workload, df in tree[bmk].items():
                selected = dict(js[workload])
                keys = [int(x) for x in selected]
                keys = [x for x in keys if x in df.index]
                df = df.loc[keys]
                cpi, weight = u.weighted_cpi(df)
                weights.append(weight)


                workload_dict[conf][workload] = {}
                workload_dict[conf][workload]['CPI'] = cpi
                workload_dict[conf][workload]['IPC'] = 1.0/cpi
                workload_dict[conf][workload]['Coverage'] = weight

                df_dict[conf][workload] = df

                # merge multiple sub-items of a benchmark
                if merge_benckmark:
                    insts_file = insts_file_fmt.format(ver, workload)
                    insts = int(get_insts(insts_file))
                    workload_dict[conf][workload]['TotalInst'] = insts
                    workload_dict[conf][workload]['PredictedCycles'] = insts*cpi
                    seconds = insts*cpi / clock_rate
                    workload_dict[conf][workload]['PredictedSeconds'] = seconds
                    time += seconds
                    coverage += weight
                    count += 1

            if merge_benckmark:
                bmk_stat[conf][bmk] = {}
                bmk_stat[conf][bmk]['time'] = time
                ref_time = c.get_spec_ref_time(bmk, ver)
                assert ref_time is not None
                bmk_stat[conf][bmk]['ref_time'] = ref_time
                bmk_stat[conf][bmk]['score'] = ref_time / time
                bmk_stat[conf][bmk]['Coverage'] = coverage/count

    #now try to find a best conf
    other_confs = set(confs.keys()) - set([base])
    best_other_conf = 'best_other'

    workload_dict[best_other_conf] = {}
    bmk_stat[best_other_conf] = {}
    for bmk in bmk_stat[base]:
        weights = []
        time = 0
        coverage = 0
        count = 0
        for workload in workload_dict[base]:
            workload_dict[best_other_conf][workload] = {}
            best_raw_df = df_dict[base][workload]
            for conf in other_confs:
                ipc_max = np.maximum(best_raw_df['ipc'],df_dict[conf][workload]['ipc'])
                best_raw_df['ipc'] = ipc_max
            cpi, weight = u.weighted_cpi(best_raw_df)
            weights.append(weight)

            workload_dict[best_other_conf][workload] = {}
            workload_dict[best_other_conf][workload]['CPI'] = cpi
            workload_dict[best_other_conf][workload]['IPC'] = 1.0/cpi
            workload_dict[best_other_conf][workload]['Coverage'] = weight
            # merge multiple sub-items of a benchmark
            if merge_benckmark:
                insts_file = insts_file_fmt.format(ver, workload)
                insts = int(get_insts(insts_file))
                workload_dict[best_other_conf][workload]['TotalInst'] = insts
                workload_dict[best_other_conf][workload]['PredictedCycles'] = insts*cpi
                seconds = insts*cpi / clock_rate
                workload_dict[best_other_conf][workload]['PredictedSeconds'] = seconds
                time += seconds
                coverage += weight
                count += 1
        if merge_benckmark:
            bmk_stat[best_other_conf][bmk] = {}
            bmk_stat[best_other_conf][bmk]['time'] = time
            ref_time = c.get_spec_ref_time(bmk, ver)
            assert ref_time is not None
            bmk_stat[best_other_conf][bmk]['ref_time'] = ref_time
            bmk_stat[best_other_conf][bmk]['score'] = ref_time / time
            bmk_stat[best_other_conf][bmk]['Coverage'] = coverage/count

    confs[best_other_conf] = ''

    for conf in confs:
        print(conf, '='*60)
        df = pd.DataFrame.from_dict(workload_dict[conf], orient='index')
        workload_dict[conf] = df
        print(df)

        if merge_benckmark:
            df = pd.DataFrame.from_dict(bmk_stat[conf], orient='index')
            bmk_stat[conf] = df
            excluded = df[df['Coverage'] <= min_coverage]
            df = df[df['Coverage'] > min_coverage]
            print(df)
            print('Estimated score @ 1.5GHz:', geometric_mean(df['score']))
            print('Estimated score per GHz:', geometric_mean(df['score'])/(clock_rate/(10**9)))
            print('Excluded because of low coverage:', list(excluded.index))


    tests = []
    for conf in confs.keys():
        if conf == base:
            continue
        rel = workload_dict[conf]['IPC']/workload_dict[base]['IPC']
        print(f'{conf}  Mean relative performance: {gmean(rel)}')
        tests.append(rel)
    if len(tests):
        dfx = pd.concat(tests, axis=1)
        print('Relative performance:')
        print(dfx)


def compute_weighted_llc_maki(ver, confs, base, simpoints, prefix, insts_file_fmt, stat_file,
        clock_rate, min_coverage=0.0, blacklist=[], whitelist=[], merge_benckmark=False):
    target = eval(f't.{prefix}llc_targets')
    workload_dict = {}
    bmk_stat = {}
    for conf, file_path in confs.items():
        workload_dict[conf] = {}
        bmk_stat[conf] = {}
        tree = u.glob_weighted_stats(
                file_path,None,
                get_funcs=[
                    lambda file_path: u.single_stat_factory(target, 'l3.demand_misses', prefix)(file_path)/
                        (u.single_stat_factory(target, 'Insts', prefix)(file_path)/1000),
                    lambda file_path: tmean([u.single_stat_factory(target, 'slice_set_accesses::'+str(i), prefix)(file_path) for i in range(4)]) /
                        (u.single_stat_factory(target, 'Insts', prefix)(file_path)/1000),
                    lambda file_path: tstd([u.single_stat_factory(target, 'slice_set_accesses::'+str(i), prefix)(file_path) for i in range(4)],ddof=0) /
                        (u.single_stat_factory(target, 'Insts', prefix)(file_path)/1000),
                ],
                stat_names=["llc_mpki","slice_access_pki","slice_access_pki_dev"],
                simpoints=simpoints,
                stat_file=stat_file,
                )
        with open(simpoints) as jf:
            js = json.load(jf)
            print(js.keys())
        times = {}
        for bmk in tree:
            if bmk in blacklist:
                continue
            if len(whitelist) and bmk not in whitelist:
                continue
            weights = []
            coverage = 0
            count = 0
            sum_misses = 0
            sum_access_avg = 0
            sum_access_std = 0
            sum_insts = 0
            for workload, df in tree[bmk].items():
                selected = dict(js[workload])
                keys = [int(x) for x in selected]
                keys = [x for x in keys if x in df.index]
                df = df.loc[keys]
                mpki, weight = u.weighted_one_stat(df,"llc_mpki")
                apki, _ = u.weighted_one_stat(df,"slice_access_pki")
                apki_dev, _ = u.weighted_one_stat(df,"slice_access_pki_dev")
                weights.append(weight)

                workload_dict[conf][workload] = {}
                workload_dict[conf][workload]['LLC_MPKI'] = mpki
                workload_dict[conf][workload]['slice_apki'] = apki
                workload_dict[conf][workload]['slice_apki_dev'] = apki_dev
                workload_dict[conf][workload]['Coverage'] = weight

                # merge multiple sub-items of a benchmark
                if merge_benckmark:
                    insts_file = insts_file_fmt.format(ver, workload)
                    insts = int(get_insts(insts_file))
                    workload_dict[conf][workload]['TotalInst'] = insts
                    workload_dict[conf][workload]['PredictedLLCMisses'] = (insts/1000)*mpki
                    # workload_dict[conf][workload]['Predicted STD of slice access'] = (insts/1000)*apki_dev
                    sum_misses += (insts/1000)*mpki
                    sum_access_avg += (insts/1000)*apki
                    sum_access_std += (insts/1000)*apki_dev
                    sum_insts += insts
                    coverage += weight
                    count += 1

            if merge_benckmark:
                bmk_stat[conf][bmk] = {}
                bmk_stat[conf][bmk]['LLC_MPKI'] = sum_misses/(sum_insts/1000)
                bmk_stat[conf][bmk]['LLC_slice_apki'] = sum_access_avg/(sum_insts/1000)
                bmk_stat[conf][bmk]['LLC_STD_slice_apki'] = sum_access_std/(sum_insts/1000)
                bmk_stat[conf][bmk]['Coverage'] = coverage/count

    for conf in confs:
        print(conf, '='*60)
        df = pd.DataFrame.from_dict(workload_dict[conf], orient='index')
        workload_dict[conf] = df
        print(df)

        if merge_benckmark:
            df = pd.DataFrame.from_dict(bmk_stat[conf], orient='index')
            excluded = df[df['Coverage'] <= min_coverage]
            df = df[df['Coverage'] > min_coverage]
            print(df)
            print('Excluded because of low coverage:', list(excluded.index))

    for conf in confs.keys():
        if conf == base:
            continue
        bmk_rel_mpki = {}
        for bmk in bmk_stat[conf].keys():
            bmk_rel_mpki[bmk] = bmk_stat[conf][bmk]['LLC_MPKI'] / bmk_stat[base][bmk]['LLC_MPKI']

        print(conf + ' Relative MPKI:')
        print(pd.DataFrame.from_dict(bmk_rel_mpki, orient='index'))


def compute_weighted_llc_mpki(ver, confs, base, simpoints, prefix, insts_file_fmt, stat_file,
        clock_rate, min_coverage=0.0, blacklist=[], whitelist=[], merge_benckmark=False):
    target = eval(f't.{prefix}llc_targets_newgem')
    workload_dict = {}
    bmk_stat = {}
    for conf, file_path in confs.items():
        workload_dict[conf] = {}
        bmk_stat[conf] = {}
        tree = u.glob_weighted_stats(
                file_path,None,
                get_funcs=[
                    lambda file_path: u.single_stat_factory(target, 'l3.demandMisses', prefix)(file_path),
                    lambda file_path: u.single_stat_factory(target, 'Insts', prefix)(file_path),
                ],
                stat_names=["llc_misses","insts"],
                simpoints=simpoints,
                stat_file=stat_file,
                )
        with open(simpoints) as jf:
            js = json.load(jf)
            print(js.keys())
        times = {}
        for bmk in tree:
            if bmk in blacklist:
                continue
            if len(whitelist) and bmk not in whitelist:
                continue
            weights = []
            coverage = 0
            count = 0
            sum_misses = 0
            sum_access_avg = 0
            sum_access_std = 0
            sum_insts = 0
            for workload, df in tree[bmk].items():
                selected = dict(js[workload])
                keys = [int(x) for x in selected]
                keys = [x for x in keys if x in df.index]
                df = df.loc[keys]
                df = df.drop(df[df['insts'] < 10_000_000].index)
                mpki_array = np.array(df['llc_misses']) / (np.array(df['insts']) / 1000)
                df['llc_mpki'] = mpki_array.tolist()
                mpki, weight = u.weighted_one_stat(df,"llc_mpki")
                weights.append(weight)

                workload_dict[conf][workload] = {}
                workload_dict[conf][workload]['LLC_MPKI'] = mpki
                workload_dict[conf][workload]['Coverage'] = weight

                # merge multiple sub-items of a benchmark
                if merge_benckmark:
                    insts_file = insts_file_fmt.format(ver, workload)
                    insts = int(get_insts(insts_file))
                    workload_dict[conf][workload]['TotalInst'] = insts
                    workload_dict[conf][workload]['PredictedLLCMisses'] = (insts/1000)*mpki
                    # workload_dict[conf][workload]['Predicted STD of slice access'] = (insts/1000)*apki_dev
                    sum_misses += (insts/1000)*mpki
                    sum_insts += insts
                    coverage += weight
                    count += 1

            if merge_benckmark:
                bmk_stat[conf][bmk] = {}
                bmk_stat[conf][bmk]['LLC_MPKI'] = sum_misses/(sum_insts/1000)
                bmk_stat[conf][bmk]['Coverage'] = coverage/count

    for conf in confs:
        print(conf, '='*60)
        df = pd.DataFrame.from_dict(workload_dict[conf], orient='index')
        workload_dict[conf] = df
        print(df)

        if merge_benckmark:
            df = pd.DataFrame.from_dict(bmk_stat[conf], orient='index')
            excluded = df[df['Coverage'] <= min_coverage]
            df = df[df['Coverage'] > min_coverage]
            print(df)
            df.to_csv(f'{conf}_bmk_mpki.csv',sep='\t')
            print('Excluded because of low coverage:', list(excluded.index))

    # tests = []
    # conf_name = []
    # for conf in confs.keys():
    #     if conf == base:
    #         continue
    #     bmk_rel_mpki = {}
    #     for bmk in bmk_stat[conf].keys():
    #         bmk_rel_mpki[bmk] = bmk_stat[conf][bmk]['LLC_MPKI'] / bmk_stat[base][bmk]['LLC_MPKI']

    #     print(conf + ' Relative MPKI:')
    #     rel_pd = pd.DataFrame.from_dict(bmk_rel_mpki, orient='index')
    #     print(rel_pd)
    tests = []
    conf_name = []
    for conf in confs.keys():
        rel = workload_dict[conf]['LLC_MPKI']
        tests.append(rel)
        conf_name.append(conf)
    if len(tests):
        dfx = pd.concat(tests, axis=1)
        dfx.columns = conf_name
        print('all llc mpki:')
        print(dfx)
        dfx.to_csv(f'Nanhu_bmk_mpki.csv',sep='\t')

def compute_llc_miss_access(ver, confs, base, simpoints, prefix, insts_file_fmt, stat_file,
        clock_rate, min_coverage=0.0, blacklist=[], whitelist=[], merge_benckmark=False):
    target = eval(f't.{prefix}llc_targets')
    workload_dict = {}
    bmk_stat = {}
    for conf, file_path in confs.items():
        workload_dict[conf] = {}
        bmk_stat[conf] = {}
        tree = u.glob_weighted_stats(
                file_path,None,
                get_funcs=[
                    lambda file_path: u.single_stat_factory(target, 'ipc', prefix)(file_path),
                    lambda file_path: u.single_stat_factory(target, 'l3.tags.slice_set_accesses_avg', prefix)(file_path),
                    lambda file_path: np.sqrt(u.single_stat_factory(target, 'l3.tags.slice_set_accesses_var', prefix)(file_path)),
                    lambda file_path: u.single_stat_factory(target, 'l3.tags.slice_set_accesses_80', prefix)(file_path),
                    lambda file_path: u.single_stat_factory(target, 'l3.tags.slice_set_accesses_80_lastT', prefix)(file_path),
                    lambda file_path: u.single_stat_factory(target, 'l3.tags.slice_set_accesses_80_intersect', prefix)(file_path),
                ],
                stat_names=["ipc","llc_access_avg","llc_accesses_std","llc_accesses_80","llc_accesses_80_lastT","llc_accesses_80_intersect"],
                simpoints=simpoints,
                stat_file=stat_file,
                )
        with open(simpoints) as jf:
            js = json.load(jf)
            print(js.keys())
        times = {}
        for bmk in tree:
            if bmk in blacklist:
                continue
            if len(whitelist) and bmk not in whitelist:
                continue
            weights = []
            for workload, df in tree[bmk].items():
                selected = dict(js[workload])
                keys = [int(x) for x in selected]
                keys = [x for x in keys if x in df.index]
                df = df.loc[keys]
                ipc, weight = u.weighted_one_stat(df,"ipc")
                llcaa, weight = u.weighted_one_stat(df,"llc_access_avg")
                llcas, _ = u.weighted_one_stat(df,"llc_accesses_std")
                llca8, _ = u.weighted_one_stat(df,"llc_accesses_80")
                llca8l, _ = u.weighted_one_stat(df,"llc_accesses_80_lastT")
                llca8i, _ = u.weighted_one_stat(df,"llc_accesses_80_intersect")
                weights.append(weight)

                workload_dict[conf][workload] = {}
                workload_dict[conf][workload]['IPC'] = ipc
                workload_dict[conf][workload]['llc_access_avg'] = llcaa
                workload_dict[conf][workload]['llc_accesses_std'] = llcas
                workload_dict[conf][workload]['llc_accesses_80'] = llca8
                workload_dict[conf][workload]['llc_accesses_80_lastT'] = llca8l
                workload_dict[conf][workload]['llc_accesses_80_intersect'] = llca8i
                workload_dict[conf][workload]['Coverage'] = weight

    for conf in confs:
        print(conf, '='*60)
        df = pd.DataFrame.from_dict(workload_dict[conf], orient='index')
        workload_dict[conf] = df
        print(df)

        if merge_benckmark:
            df = pd.DataFrame.from_dict(bmk_stat[conf], orient='index')
            excluded = df[df['Coverage'] <= min_coverage]
            df = df[df['Coverage'] > min_coverage]
            print(df)
            print('Excluded because of low coverage:', list(excluded.index))


def gem5_spec(ver='17'):
    confs = {
            'FullO3': f'/nfs/home/zhangchuanqi/lvna/5g/cdp_search/all_50m_wrapper{ver}/FullWindowO3Config',
            'O3_2_14': f'/nfs/home/zhangchuanqi/lvna/5g/cdp_search/all_50m_wrapper{ver}/O3_2_14',
            'O3_4_12': f'/nfs/home/zhangchuanqi/lvna/5g/cdp_search/all_50m_wrapper{ver}/O3_4_12',
            'O3_8_8': f'/nfs/home/zhangchuanqi/lvna/5g/cdp_search/all_50m_wrapper{ver}/O3_8_8',
            'O3_12_4': f'/nfs/home/zhangchuanqi/lvna/5g/cdp_search/all_50m_wrapper{ver}/O3_12_4',
            'O3_14_2': f'/nfs/home/zhangchuanqi/lvna/5g/cdp_search/all_50m_wrapper{ver}/O3_14_2',
            }

    compute_best_weighted_cpi(
            ver=ver,
            confs=confs,
            base='FullO3',
            simpoints=f'/nfs/home/zhangchuanqi/lvna/5g/DirtyStuff/resources/simpoint_cpt_desc/simpoints{ver}.json',
            prefix = '',
            stat_file='m5out/stats.txt',
            insts_file_fmt =
            '/nfs-nvme/home/share/checkpoints_profiles/spec{}_rv64gc_o2_50m/profiling/{}/nemu_out.txt',
            clock_rate = 2 * 10**9,
            min_coverage = 0.79,
            # blacklist = ['gamess'],
            merge_benckmark=True,
            )

    # compute_weighted_llc_mpki(
    #         ver=ver,
    #         confs=confs,
    #         base='FullO3',
    #         simpoints=f'/home51/zcq/playground/DirtyStuff/resources/simpoint_cpt_desc/simpoints{ver}_abort.json',
    #         prefix = '',
    #         stat_file='m5out/stats.txt',
    #         insts_file_fmt =
    #         '/bigdata/zyy/checkpoints_profiles/betapoint_profile_{}_fix_mem_addr/{}/nemu_out.txt',
    #         clock_rate = 2 * 10**9,
    #         min_coverage = 0.5,
    #         # blacklist = ['gamess'],
    #         merge_benckmark=True,
    #         )

def hw_gem5_spec(ver='17'):
    confs = {
            'FullO3': f'/nfs/home/zhangchuanqi/lvna/5g/bm_search/all_50m_wrapper17/FullWindowO3Config',
            }

    # compute_weighted_cpi(
    #         ver=ver,
    #         confs=confs,
    #         base='FullO3',
    #         simpoints=f'/nfs/home/zhangchuanqi/lvna/5g/DirtyStuff/resources/simpoint_cpt_desc/simpoints17.json',
    #         prefix = '',
    #         stat_file='m5out/stats.txt',
    #         insts_file_fmt =
    #         '/bigdata/zyy/checkpoints_profiles/betapoint_profile_{}_fix_mem_addr/{}/nemu_out.txt',
    #         clock_rate = 2 * 10**9,
    #         min_coverage = 0.8,
    #         # blacklist = ['gamess'],
    #         merge_benckmark=False,
    #         )
    compute_llc_miss_access(
            ver=ver,
            confs=confs,
            base='FullO3',
            simpoints=f'/nfs/home/zhangchuanqi/lvna/5g/DirtyStuff/resources/simpoint_cpt_desc/simpoints17.json',
            prefix = '',
            stat_file='m5out/stats.txt',
            insts_file_fmt =
            '/bigdata/zyy/checkpoints_profiles/betapoint_profile_{}_fix_mem_addr/{}/nemu_out.txt',
            clock_rate = 2 * 10**9,
            min_coverage = 0.8,
            # blacklist = ['gamess'],
            merge_benckmark=False,
            )

def gem5_spec2006():
    ver = '06'
    confs = {
            'FullO3': '/bigdata/zcq/gem5_playground/near_xs/test_new_wrapper06/FullWindowO3Config',
            'Complex': '/bigdata/zcq/gem5_playground/near_xs/test_new_wrapper06/ComplexO3Config',
            }

    compute_weighted_cpi(
            ver=ver,
            confs=confs,
            base='FullO3',
            simpoints=f'/home51/zcq/playground/DirtyStuff/resources/simpoint_cpt_desc/simpoints{ver}_abort.json',
            prefix = '',
            stat_file='m5out/stats.txt',
            insts_file_fmt =
            '/bigdata/zyy/checkpoints_profiles/betapoint_profile_{}_fix_mem_addr/{}/nemu_out.txt',
            clock_rate = 2 * 10**9,
            min_coverage = 0.5,
            # blacklist = ['gamess'],
            merge_benckmark=True,
            )

    compute_weighted_llc_mpki(
            ver=ver,
            confs=confs,
            base='FullO3',
            simpoints=f'/home51/zcq/playground/DirtyStuff/resources/simpoint_cpt_desc/simpoints{ver}_abort.json',
            prefix = '',
            stat_file='m5out/stats.txt',
            insts_file_fmt =
            '/bigdata/zyy/checkpoints_profiles/betapoint_profile_{}_fix_mem_addr/{}/nemu_out.txt',
            clock_rate = 2 * 10**9,
            min_coverage = 0.5,
            # blacklist = ['gamess'],
            merge_benckmark=True,
            )


def xiangshan_spec2006():
    ver = '06'
    confs = {
            'XiangShan1': '/home/zyy/expri_results/xs_simpoint_batch/SPEC06_EmuTasksConfig',
            'XiangShan2': '/home/zyy/expri_results/xs_simpoint_batch/SPEC06_EmuTasksConfig',
            }

    compute_weighted_cpi(
            ver=ver,
            confs=confs,
            base='XiangShan1',
            simpoints=f'/home51/zyy/expri_results/simpoints{ver}.json',
            prefix = 'xs_',
            stat_file='simulator_err.txt',
            insts_file_fmt =
            '/bigdata/zyy/checkpoints_profiles/betapoint_profile_{}_fix_mem_addr/{}/nemu_out.txt',
            clock_rate = 1.5 * 10**9,
            min_coverage = 0.75,
            # blacklist = ['gamess'],
            merge_benckmark=True,
            )

def nanhu_gem5_spec(ver='06'):
    confs = {
            'Nanhu': f'/nfs/home/zhangchuanqi/lvna/for_xs/cacheconflict_search/nanhu_test_{ver}/NanhuConfig',
            'Nanhu8M': f'/nfs/home/zhangchuanqi/lvna/for_xs/cacheconflict_search/nanhu_test_{ver}/Nanhu8ML3Config',
            'Nanhu8M16ass': f'/nfs/home/zhangchuanqi/lvna/for_xs/cacheconflict_search/nanhu_test_{ver}/Nanhu8M16assL3Config',
            }

    compute_weighted_cpi(
            ver=ver,
            confs=confs,
            base='Nanhu',
            simpoints=simpoints_file[ver],
            prefix = '',
            stat_file='m5out/stats.txt',
            insts_file_fmt =
            '/nfs-nvme/home/share/checkpoints_profiles/spec{}_rv64gc_o2_50m/profiling/{}/nemu_out.txt',
            clock_rate = 2 * 10**9,
            min_coverage = 0.79,
            # blacklist = ['gamess'],
            merge_benckmark=True,
            )

    compute_weighted_llc_mpki(
            ver=ver,
            confs=confs,
            base='Nanhu',
            simpoints=simpoints_file[ver],
            prefix = '',
            stat_file='m5out/stats.txt',
            insts_file_fmt =
            '/nfs-nvme/home/share/checkpoints_profiles/spec{}_rv64gc_o2_50m/profiling/{}/nemu_out.txt',
            clock_rate = 2 * 10**9,
            min_coverage = 0.79,
            # blacklist = ['gamess'],
            merge_benckmark=True,
            )


if __name__ == '__main__':
    # xiangshan_spec2006()
    # gem5_spec2017()
    # gem5_spec2006()
    # gem5_spec("06")
    # gem5_spec("17")
    # hw_gem5_spec()
    nanhu_gem5_spec()
