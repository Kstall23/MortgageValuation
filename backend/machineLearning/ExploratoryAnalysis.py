import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import git
from git import Repo
import os
pd.set_option('display.max_columns', 11)
pd.set_option('display.width', 170)


os.chdir('..')
repo_dir = os.path.join(os.getcwd(), "Brightvine_model_files")
input_file_name = "Mortgage Dataset.csv"
data = pd.read_csv(os.path.join(repo_dir, input_file_name), sep=',', names=['Year', 'MonthlyIncome', 'UPBatAcquisition', 'LTVRatio', 'BorrowerCount', 'InterestRate', 'OriginationValue', 'HousingExpenseToIncome', 'TotalDebtToIncome', 'B1CreditScore', 'B2CreditScore'], header=0)


# Summary Statistics
print(data.describe())
nines = data[data['HousingExpenseToIncome'] == 999]
non = data[data['HousingExpenseToIncome'] != 999]

print(nines)
print(non.describe())
print(len(nines) / len(data))

''' Stuff for the change between UPBatAcquisition and OriginationValue '''
# change  = data[data['UPBatAcquisition'] != data['OriginationValue']]
# diff = []
# for index, row in change.iterrows():
#     diff.append(row['OriginationValue'] - row['UPBatAcquisition'])
# Sdiff = pd.Series(diff)
# print(Sdiff)
# print(Sdiff.describe())
# print(len(Sdiff[Sdiff < 0]))
# print(change)
# print(change.describe())
# print(len(change) / len(data))

print()
print()

# Correlation Matrix
corr = data.corr()
print(corr)
print()
# Heatmap for Correlation Matrix
# mask = np.triu(np.ones_like(corr, dtype=bool)) # mask for the upper triangle
# fig, ax = plt.subplots()
# cmap = sns.diverging_palette(230,20, as_cmap=True)  # create color map
# sns.heatmap(corr, mask=mask, cmap=cmap, center=0, linewidths=.75, cbar_kws={"shrink":.8})   # draw heatmap
# plt.show()

sns.pairplot(data)
plt.savefig('pairplot.png')