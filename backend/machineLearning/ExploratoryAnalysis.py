import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import git
from git import Repo
import os
import sys
pd.set_option('display.max_columns', 11)
pd.set_option('display.width', 200)
np.set_printoptions(linewidth=150, threshold=sys.maxsize)

os.chdir('..')
repo_dir = os.path.join(os.getcwd(), "Brightvine_model_files")
input_file_name = "Mortgage Dataset.csv"
columns = ['Year', 'MonthlyIncome', 'UPBatAcquisition', 'LTVRatio', 'BorrowerCount', 'InterestRate', 'OriginationValue', 'HousingExpenseToIncome', 'TotalDebtToIncome', 'B1CreditScore', 'B2CreditScore']
data = pd.read_csv(os.path.join(repo_dir, input_file_name), sep=',', names=columns, header=0)


''' # Summary Statistics '''
# print(data.describe())
# nines = data[data['HousingExpenseToIncome'] == 999]
# non = data[data['HousingExpenseToIncome'] != 999]

# print(nines)
# print(non.describe())
# print(len(nines) / len(data))

''' # Stuff for the change between UPBatAcquisition and OriginationValue '''
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

''' # Correlation Matrix '''
# corr = data.corr()
# print(corr)
# print()
''' # Heatmap for Correlation Matrix, Pairplot '''
# mask = np.triu(np.ones_like(corr, dtype=bool)) # mask for the upper triangle
# fig, ax = plt.subplots()
# cmap = sns.diverging_palette(230,20, as_cmap=True)  # create color map
# sns.heatmap(corr, mask=mask, cmap=cmap, center=0, linewidths=.75, cbar_kws={"shrink":.8})   # draw heatmap
# plt.show()
# sns.pairplot(data)
# plt.savefig('pairplot.png')


''' # Clustering Attempt '''
from sklearn.cluster import KMeans
import PreProcessing

data = PreProcessing.removeMissing(data)
data = PreProcessing.removeExtremeOutliers(data, columns)
data, columns = PreProcessing.normalize(data, columns)
trainData, testData = PreProcessing.create_folds(data)

kmeans = KMeans(n_clusters = 6, n_init = 10)
clusters = kmeans.fit_predict(data)


# plt.scatter(np.asarray(trainData['MonthlyIncome']), np.asarray(trainData['OriginationValue']), c=clusters)
# plt.xlabel("Monthly Income")
# plt.ylabel("Loan Origination")
# plt.show()


''' # Does PCA provide any insight? '''
from sklearn.decomposition import PCA
pca = PCA()                                         # set up pca-er thing

pca_data = pca.fit_transform(data)                  # perform pca on dataset
exp_var_pca = pca.explained_variance_ratio_         # this shows variance explained by each component
cum_sum = np.cumsum(exp_var_pca)                    # this shows the cumulative variance explained by each component

print(exp_var_pca)

# graph it
# plt.step(range(0,len(cum_sum)), cum_sum, where='mid', label='Cumulative Explained Variance')
# plt.ylabel('Explained Variance Ratio')
# plt.xlabel('Number of Principal Components')
# plt.title('Cumulative Explained Variance')
# plt.show()


pca2 = PCA(n_components=2)
pca_data2 = pca2.fit_transform(data)

''' # graph the pca scatteplot '''
'''
# scale x and y
x = pca_data2[:,0]
y = pca_data2[:,1]
scalex = 1.0/(x.max() - x.min())
scaley = 1.0/(y.max() - y.min())

# variables
num_arrows = pca2.components_.shape[1]
xArrows = pca2.components_[0]
yArrows = pca2.components_[1]
cols = data.columns

plt.figure(figsize=(12,8))

# scatter plot
scatter = plt.scatter(x * scalex, y * scaley, c = clusters, cmap='Set1')

# plot arrows
for i in range(num_arrows):
    plt.arrow(0, 0, xArrows[i], yArrows[i], color='g', alpha=0.75, lw=2)
    plt.text(xArrows[i] * 1.15, yArrows[i] * 1.15, cols[i], color='r', size=10)

plt.xlabel("Pincipal Component 1")
plt.ylabel("Pricipal Component 2")
plt.title("Biplot of 2 Pricipal Components for Mortgage Dataset")
plt.grid(alpha=.4)
plt.show()
'''

''' # graph PCA scatterplot separated by cluster '''

fig, ax = plt.subplots(2,3, sharex=True, sharey=True)
axrow = 0
axcol = 0
colors = ['red', 'darkorange', 'yellow', 'green', 'deepskyblue', 'darkblue', 'violet', 'hotpink']

for i in range(6):

    # deal with indexing the graph position
    if i == 3:
        axrow += 1
        axcol = 0

    # grab the data for one cluster using a boolean mask
    mask = clusters == i
    masked_pca_data2 = pca_data2[mask]

    # scale x and y
    x = masked_pca_data2[:,0]
    y = masked_pca_data2[:,1]
    scalex = 1.0/(x.max() - x.min())
    scaley = 1.0/(y.max() - y.min())

    # variables
    num_arrows = pca2.components_.shape[1]
    xArrows = pca2.components_[0]
    yArrows = pca2.components_[1]
    cols = data.columns


    # scatter plot
    ax[axrow, axcol].scatter(x * scalex, y * scaley, c = colors[i])

    # plot arrows
    for j in range(num_arrows):
        ax[axrow, axcol].arrow(0, 0, xArrows[j], yArrows[j], color='lightseagreen', alpha=0.75, lw=2)
        ax[axrow, axcol].text(xArrows[j] * 1.15, yArrows[j] * 1.15, cols[j], color='black', size=10)

    ax[axrow, axcol].grid(alpha=.4)
    ax[axrow, axcol].set_title("Cluster " + str(i))

    axcol += 1

plt.show()





''' Exploratory Graphing for figuring out distributions and outliers '''
# fig, ax = plt.subplots(3, 4)

# x = 0
# y = 0
# for i in range(10):
#     if (i == 4) or (i == 8):
#         x += 1
#         y = 0
#     counts, edges, bars = ax[x, y].hist(data[columns[i]], bins=10)
#     ax[x, y].set_title(columns[i])
#     ax[x, y].bar_label(bars, fontsize='xx-small')
#     ax[x, y].axvline(data[columns[i]].mean(), linestyle='dashed', color='lime')
#     ax[x, y].axvline(data[columns[i]].std()*3 + data[columns[i]].quantile(.75), linestyle='solid', color='r')
#     ax[x, y].axvline(data[columns[i]].quantile(.25) - data[columns[i]].std()*3, linestyle='solid', color='r')

#     y += 1

# mean = data[columns[8]].mean()
# std = data[columns[8]].std()
# max = data[columns[8]].max()

# upper = data[columns[0]].quantile(.75)
# med = data[columns[0]].median()
# lower = data[columns[0]].quantile(.25)
# iqr = upper - lower

# counts, edges, bars = plt.hist(data[columns[0]])
# plt.bar_label(bars)

# # plt.scatter(data[columns[0]], np.zeros(len(data)))
# plt.axvline(med, linestyle='solid', color='lime')
# plt.axvline(upper, linestyle='solid', color='lime')
# plt.axvline(lower, linestyle='solid', color='lime')
# plt.axvline(lower - 3*iqr, linestyle='dashed', color='red')
# plt.axvline(upper + 3*iqr, linestyle='dashed', color='red')

# val = med
# while (val < (max + std)):
#     val += std
#     plt.axvline(val, linestyle='dashed', color='red')

# plt.title(columns[0])
# plt.show()
