import pandas as pd
import matplotlib.pyplot as plt

# Read data from xlsx file
file = '临时policy数据(1).xlsx'
df = pd.read_excel(file)

# df = df[df.columns[:4]]

# Group the dataframe by "workload0" column and store it in a new dataframe
grouped = df.groupby("workload0")

# compare cpu0 and cpu1
exp1 = 'csv_speedup_cpu0'
exp2 = 'csv_speedup_cpu1'

# collect data needed to draw boxplots
data = []
labels = []
for name, group in grouped:
    data.append(group[exp1].values)
    data.append(group[exp2].values)
    labels.append(name)
xlen = len(labels)

# draw boxplots
fig, ax = plt.subplots()
bp = ax.boxplot(data, showfliers=True, patch_artist=True)

# set xlabels
ax.set_xticks([2*i + 1.5 for i in range(0, xlen)])
ax.set_xticklabels(labels)

# set vlines to separate different workloads
for i in range(0, xlen):
    ax.axvline(x=2*i+0.5, color='gray')

# set box colors alternatively
for i, box in enumerate(bp['boxes']):
    color = 'red' if i % 2 == 0 else 'green'
    box.set_facecolor(color)

plt.show()