''' # Import statements '''
import PreProcessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import git
from git import Repo

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import os
import sys
pd.set_option('display.max_columns', 11)
pd.set_option('display.width', 200)
np.set_printoptions(linewidth=150, threshold=sys.maxsize)


''' # Import data '''
os.chdir('..')
repo_dir = os.path.join(os.getcwd(), "Brightvine_model_files")
input_file_name = "Mortgage Dataset.csv"
columns = ['Year', 'MonthlyIncome', 'UPBatAcquisition', 'LTVRatio', 'BorrowerCount', 'InterestRate', 'OriginationValue', 'HousingExpenseToIncome', 'TotalDebtToIncome', 'B1CreditScore', 'B2CreditScore']
data = pd.read_csv(os.path.join(repo_dir, input_file_name), sep=',', names=columns, header=0)
print()
print()


''' # Preprocessing steps '''
data = PreProcessing.removeMissing(data)
data = PreProcessing.removeExtremeOutliers(data, columns)
data, columns, ss = PreProcessing.normalize(data, columns)
trainData, testData = PreProcessing.create_folds(data)


''' # Variables '''
NUM_CLUSTERS = 8
NUM_PCs = 6


''' # Reduce Dimensionality, Apply Clustering, Reduce Further in order to Visualize '''
# Use PCA to reduce dimensionality
pca = PCA(n_components=NUM_PCs)
pca_data = pca.fit_transform(data)

# Use KMeans to cluster the PCA-ed data
kmeans = KMeans(n_clusters = NUM_CLUSTERS, n_init = 10)
clusters = kmeans.fit_predict(pca_data)

# Reduce to 2 components this time to graph it
pca2 = PCA(n_components=2)
pca_data2 = pca2.fit_transform(data)


''' # Look at the cluster centroids, do they mean anything? '''
print(kmeans.cluster_centers_)
inversed_centroids = pd.DataFrame(ss.inverse_transform(pca.inverse_transform(kmeans.cluster_centers_)), columns=columns)
print(inversed_centroids)
for col in columns:
    print(inversed_centroids.sort_values(col, axis=0))


''' # Graph Biplot on 2 PCAs, with arrows, subplots separated by cluster '''
if NUM_CLUSTERS % 3 == 0:
    plus = 0
else:
    plus = 1
subplot_rows = NUM_CLUSTERS // 3 + plus

fig, ax = plt.subplots(subplot_rows,3, sharex=True, sharey=True)
axrow = 0
axcol = 0
colors = ['red', 'darkorange', 'yellow', 'green', 'deepskyblue', 'darkblue', 'violet', 'hotpink', 'lightcoral', 'sandybrown', 'khaki', 'lime', 'cyan', 'slateblue', 'fuchsia']

for i in range(NUM_CLUSTERS):

    # deal with indexing the graph position
    if (i % 3 == 0) and (i != 0):
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



''' # Summary Statistics '''
# print(data.describe())
# nines = data[data['HousingExpenseToIncome'] == 999]
# non = data[data['HousingExpenseToIncome'] != 999]

# print(nines)
# print(non.describe())
# print(len(nines) / len(data))


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


''' # Determining Best Cluster Count '''
# cluster_labels = []
# centroids = []
# inertias = []
# sils = []
# MAX_CLUSTERS = 21

# for k in range(1, MAX_CLUSTERS):
#     kmeans = KMeans(n_clusters=k, n_init=10, )
#     clusters = kmeans.fit_predict(data)
#     print("Assigned points to", k, "different clusters")

#     cluster_labels.append(clusters)                     # save cluster labels for each cluster count
#     centroids.append(kmeans.cluster_centers_)           # save cluster centroids for each cluster count
#     inertias.append(kmeans.inertia_)                    # save inertia value for each cluster count
#     print("other values stored")
#     # if k != 1:  # silhouette score doesn't work with one cluster
#         # sils.append(silhouette_score(data, clusters))   # save silhouette scores for each cluster count
#     # print("sil score calculated")
# print(inertias)

# # Graph it
# fig, (iner, sil) = plt.subplots(1, 2, figsize=(12,5))

# iner.plot(range(1,MAX_CLUSTERS), inertias, marker='*', color='blue')
# iner.set_title('Inertia Score by K value')
# iner.set_xlabel('Number of Clusters')
# iner.set_ylabel('Inertia Value')

# # sil.plot(range(2,MAX_CLUSTERS), sils, marker='x', color='green')
# # sil.set_title('Silhouette Score by K value')
# # sil.set_xlabel('Number of Clusters')
# # sil.set_ylabel('Silhouette Score')

# plt.show()

# # While the silhouette score really should be considered in tandem 
# # with the inertia when determining optimal cluster count... 
# # It was far too costly a computation to calculate on this massive dataset
# # Based on the Inertia graph, I'm going to have to go with 8 clusters for this problem


''' # Determining Best number of PC's for PCA '''
# pca = PCA() 

# pca_data = pca.fit_transform(data)                  # perform pca on dataset
# exp_var_pca = pca.explained_variance_ratio_         # this shows variance explained by each component
# cum_sum = np.cumsum(exp_var_pca)                    # this shows the cumulative variance explained by each component

# graph it
# plt.step(range(1,len(cum_sum)+1), cum_sum, where='mid', label='Cumulative Explained Variance')
# plt.ylabel('Explained Variance Ratio')
# plt.xlabel('Number of Principal Components')
# plt.title('Cumulative Explained Variance')
# plt.show()

# Based on this graph, I will reduce the data to 6 Princpal Components to preserve 90% of the variance
# (effectively removes the correlation between columns like UPBAtAcquisition/Origination, BorrowerCount/B2CreditScore)


''' # Graph Biplot on 2 PCAs, with arrows, colored by cluster '''
# # scale x and y
# x = pca_data2[:,0]
# y = pca_data2[:,1]
# scalex = 1.0/(x.max() - x.min())
# scaley = 1.0/(y.max() - y.min())

# # variables
# num_arrows = pca2.components_.shape[1]
# xArrows = pca2.components_[0]
# yArrows = pca2.components_[1]
# cols = data.columns

# plt.figure(figsize=(12,8))

# # scatter plot
# scatter = plt.scatter(x * scalex, y * scaley, c = clusters, cmap='Set1')

# # plot arrows
# for i in range(num_arrows):
#     plt.arrow(0, 0, xArrows[i], yArrows[i], color='g', alpha=0.75, lw=2)
#     plt.text(xArrows[i] * 1.15, yArrows[i] * 1.15, cols[i], color='r', size=10)

# plt.xlabel("Pincipal Component 1")
# plt.ylabel("Pricipal Component 2")
# plt.title("Biplot of 2 Pricipal Components for Mortgage Dataset")
# plt.grid(alpha=.4)
# plt.show()
