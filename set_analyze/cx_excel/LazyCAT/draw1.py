import pandas as pd
import matplotlib.pyplot as plt

# Read data from xlsx file
file = '临时policy数据(1).xlsx'
df = pd.read_excel(file)

df = df[df.columns[:3]]

# Group the dataframe by "workload0" column and store it in a new dataframe
grouped = df.groupby("workload0")

exp = 'csv_speedup_cpu0'
data = []
labels = []

for name, group in grouped:
    data.append(group[exp].values)
    labels.append(name)

plt.boxplot(data, labels=labels, showfliers=False)
plt.show()