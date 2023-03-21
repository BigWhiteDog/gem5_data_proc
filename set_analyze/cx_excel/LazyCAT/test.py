import pandas as pd
import matplotlib.pyplot as plt

a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
b = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

plt.boxplot(b, labels='b')
plt.show()