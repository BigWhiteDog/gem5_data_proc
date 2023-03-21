from matplotlib.lines import Line2D
import pandas as pd
import matplotlib.pyplot as plt

# Read data from xlsx file
file = '全policy数据(1).xlsx'
all = pd.read_excel(file, sheet_name=None)
sheets = [  all['16bucket'].groupby("workload0"), 
            all['32bucket'].groupby("workload0"), 
            all['64bucket'].groupby("workload0")]

# collect data
exp = 'realOneWithTarget0.99_speedup1'
data = []
w0s = ["astar_biglakes", "cactusADM", "gcc_200", "gcc_cpdecl", "gcc_g23", "gcc_scilab", "lbm", "mcf", "omnetpp", "soplex_pds", "sphinx3", "xalancbmk"]

for w0 in w0s:
    for grouped in sheets:
        bar = grouped.get_group(w0)[exp].values
        data.append(bar)
        print(w0, bar)

# draw boxplots
fig, ax = plt.subplots()
bp = ax.boxplot(data, showfliers=True, patch_artist=True,
                medianprops=dict(color='red'),
                flierprops=dict(markerfacecolor='white', markersize=5, markeredgecolor='black'))

# set box colors alternatively
colors = ['orange','yellow','green']
for i, box in enumerate(bp['boxes']):
    box.set_facecolor(colors[i % 3])

# set vlines to separate different workloads
for i in range(1, len(w0s)):
    ax.axvline(x=3*i+0.5, color='gray')

# set xlabels
ax.set_xticks([3*x+2 for x in range(0, len(w0s))])
ax.set_xticklabels(w0s)

# set legend
legends = [ Line2D([0], [0], color=colors[0], lw=2, label='16 Buckets'),
            Line2D([0], [0], color=colors[1], lw=2, label='32 Buckets'),
            Line2D([0], [0], color=colors[2], lw=2, label='64 Buckets')]
plt.legend(handles = legends)
plt.title(exp)
fig.set_size_inches(23, 14)
plt.show()