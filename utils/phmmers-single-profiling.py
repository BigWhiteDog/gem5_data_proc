from operator import index
import os
import numpy as np
import utils.common as c
from utils.common import extract_samples_raw_json
import utils.target_stats as t
import numpy as np
import argparse

import json

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib import ticker

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    ax.grid(False)
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    # ax.tick_params(top=True, bottom=False,
    #                labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
    #          rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts



json_path = "/nfs/home/zhangchuanqi/lvna/5g/DirtyStuff/resources/simpoint_cpt_desc/hwfinal.json"

parser = argparse.ArgumentParser(description="options to get set stats")
# parser.add_argument('-d','--stats_dir', type=str,
#     help='stats dir to analyze',required=True)
# parser.add_argument('--ids',default=16,type=int)
# parser.add_argument('--nsamples',default=2,type=int)
# parser.add_argument('--l3_sets',default=4096,type=int)

opt = parser.parse_args()

mycolor = ['#044E48','#06746B','#20876B','#6A9A48','#B5AC23','#E6B80B','#FACA3E','#FFDF80','#FFEBB0']
mycolor = ['#661900','#B22C00','#E6450F','#FF6500','#FF8C00','#FFB200','#FFCB33','#FFDF80','#FDEDBE']


def profile_hmmer(s_dicts:dict):
    fig,ax = plt.subplots(2,2)
    fig.set_size_inches(16,9)
    sorted_worknames = sorted(list(s_dicts.keys()))
    for i in range(4):
        fy = i % 2
        fx = i // 2
        now_ax = ax[fx,fy]
        now_work = sorted_worknames[i]
        now_work_dict = s_dicts[now_work]
        #ways are in ascending order
        sorted_ways = sorted(list(now_work_dict.keys()))
        nways = len(sorted_ways)
        #incs are in descending order
        sorted_incs = sorted(list(now_work_dict[sorted_ways[0]]),reverse=True)
        nincs = len(sorted_incs)
        max_ipc = now_work_dict[sorted_ways[-1]][sorted_incs[0]]['cpu.ipc'][0]
        slowdowns = np.empty((nincs,nways))
        for wi,way in enumerate(sorted_ways):
            for ii,inc in enumerate(sorted_incs):
                slowdowns[ii,wi] = now_work_dict[way][inc]['cpu.ipc'][0] / max_ipc
        row_labels = [f'{k*0.125}GB/s' for k in sorted_incs]
        row_labels[0] = 'unlimited'
        col_labels = [f'{k}' for k in sorted_ways]
        im, cbar = heatmap(slowdowns,row_labels,col_labels,ax=now_ax,cmap='YlGn_r',cbarlabel='slowdown',aspect='auto')
        texts = annotate_heatmap(im, valfmt='{x:.2f}',textcolors=("white","black"))
        now_ax.set_title(now_work)
    
    # fig.tight_layout()
    plt.savefig(f'profiling_hmmer.png',dpi=300)

if __name__ == '__main__':
    # t_work_combine = ['hmmer_o31-hmmer0-hmmer_o30-hmmer1']
    # report_hmmer('/nfs/home/zhangchuanqi/lvna/5g/ff-reshape/log/new_hw_test/16M/hmmer_o31-hmmer0-hmmer_o30-hmmer1')
    all_base  = '/nfs/home/zhangchuanqi/lvna/5g/ff-reshape/log/new_hw_test/single-profiling'
    works = os.listdir(all_base)
    st_dict = {}
    for w in works:
        st_dict[w] = {}
        work_base = os.path.join(all_base,w)
        ass_names = os.listdir(work_base)
        for ass in ass_names:
            wayn = int(ass.split('-')[-1])
            st_dict[w][wayn] = {}
            tb_base = os.path.join(work_base,ass)
            tb_names = os.listdir(tb_base)
            for tb in tb_names:
                inc = int(tb.split('-')[-1])
                tb_path = os.path.join(tb_base,tb)
                # st_dict[w][wayn][inc] = extract_samples_raw_json(tb_path,1,1)
                with open(os.path.join(tb_path,'1period.json')) as f:
                    st_json = json.load(f)
                st_dict[w][wayn][inc] = st_json

    profile_hmmer(st_dict)