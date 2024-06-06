import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
import itertools
import math
import random
from numpy import linalg as LA
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import coint

#load dataframe
norm_ret_df = pd.read_csv('./norm_ret_df.csv', index_col = False)
#perform PCA with 10 components, whiten=True -> will return 10 principle components for each asset
pca = PCA(n_components=10, whiten=True)
beta=pca.fit_transform(norm_ret_df.T)
df_beta=pd.DataFrame(beta)
stock_pca = df_beta.values
print(stock_pca.shape)

#standardise principal component array with prepcoessing.StandardScaler() and apply this with fit_transform to the pca_stock array 
X = preprocessing.StandardScaler().fit_transform(stock_pca)

######### DBSCAN #########
# #perform DBSCAN clustering algorithm on preprocessed data eps=2, min_samples =3
dbscan = DBSCAN(eps=2, min_samples=3)
dbscan.fit_predict(X)
labels = dbscan.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print(labels)

#attach cluster labels to the last column norm_ret_df
df_beta['labels']=labels
df_beta['labels'] = df_beta['labels'].astype("category")
df_beta=df_beta.set_index(norm_ret_df.T.index)
df_beta.sort_values(by=['labels'])
print(df_beta)

#sort the rows with the same label into each of their own dataframes
k_list=np.arange(0,n_clusters_,1)
k_list
d = {}
for k in k_list:
    d[k] = pd.DataFrame()
    d[k] = df_beta[df_beta['labels'] ==k]

print(d[0])
print(d[1])
print(d[2])
print(d[3])
print(d[4])

#######KMEANS#######
#run kmeans on pre processed data (X) set clusters to 5, n_init = 10, and random_state=42 so alog is iniitialised the same every time
kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
kmeans.fit(X)
label_kmeans=kmeans.predict(X) 
center_kmeans=kmeans.cluster_centers_
print(label_kmeans)
print(center_kmeans)

####Hierarchical Clustering#####
# run agglomerative hierarchical clustering on preprocessed data X
clusters = 5
hc = AgglomerativeClustering(n_clusters= clusters, affinity='euclidean', linkage='ward')
labels_hc = hc.fit_predict(X)
print(labels_hc)

###optimal pairs via statistical analysis with DBSCAN clusters

#fetch normalised log returns for assets belonging to DBSCAN cluster 1 (label = 1)
cluster1_asset_list = d[1].index.values
clusters_norm_ret_df = norm_ret_df[cluster1_asset_list]
print(clusters_norm_ret_df)

#cumulative returns
cumulative_norm_ret=clusters_norm_ret_df.cumsum()

#optimal pairs via minimum sum of Euclidean squared distances btw cumulative log normalised returns
pair_order_list = itertools.combinations(cluster1_asset_list,2)
pairs=list(pair_order_list)
asset1_list=[]
asset2_list=[]
euclidean_distance_list=[]
for i in range(0,len(pairs)):
    asset1_list.append(pairs[i][0])
    asset2_list.append(pairs[i][1])

    dist = LA.norm(cumulative_norm_ret[asset1_list[i]]-cumulative_norm_ret[asset2_list[i]])
    euclidean_distance_list.append(dist)

# asset1_list,asset2_list

sdd_list=list(zip(pairs,euclidean_distance_list))
sdd_list.sort(key = lambda x: x[1])

#sort every pairwise combination based off of the euclidean squared distances. A unique optimal pair will occur with the minimum.
# example: if pair A and B have a distance of 2 and A and C have a distance of 3 and  C and D have a distance of 4, the optimal pairing would be (A,B) and (C,D)
# write the pairs in the tuple form (A,B) returns a list of these unique optimal pairs
# Each asset in a pair should not have previously been paired with another (no repeating assets per pairs)
sdd1=[]
sdd2=[]
for i in range(0,len(sdd_list)):
    sdd1.append(sdd_list[i][0][0])
    sdd2.append(sdd_list[i][0][1])

selected_stocks = []
selected_pairs_messd = []
opt_asset1=[]
opt_asset2=[]

for i in range(0,len(sdd_list)):
    s1=sdd1[i]
    s2=sdd2[i]

    if (s1 not in selected_stocks) and (s2 not in selected_stocks):
        selected_stocks.append(s1)
        selected_stocks.append(s2)
        pair=(s1,s2)
        selected_pairs_messd.append(pair)

    if len(selected_pairs_messd) == math.comb(len(cluster1_asset_list),2):
        break

opt_asset1=selected_stocks[0:len(selected_stocks)-1:2]
opt_asset2=selected_stocks[1:len(selected_stocks):2]

print(selected_pairs_messd)

####### optimal pairs through correlation strategy use the normalised log returns of DBSCAN cluster1 assets

#calculate pearson correlation for every possible pairing of assets
pearson_corr_list=[]

for i in range(0,len(pairs)):
    corr= pearsonr(clusters_norm_ret_df[pairs[i][0]],clusters_norm_ret_df[pairs[i][1]])[0]
    pearson_corr_list.append(corr)

#sort pairs by pearson correlation
sort_corr_list=list(zip(pairs,pearson_corr_list))
sort_corr_list.sort(key = lambda x: x[1])

sdd1=[]
sdd2=[]
for i in range(0,len(sort_corr_list)):
    sdd1.append(sort_corr_list[i][0][0])
    sdd2.append(sort_corr_list[i][0][1])

selected_stocks = []
selected_pairs_corr = []
opt_asset1=[]
opt_asset2=[]

for i in range(0,len(sort_corr_list)):
    s1=sdd1[i]
    s2=sdd2[i]

    if (s1 not in selected_stocks) and (s2 not in selected_stocks):
        selected_stocks.append(s1)
        selected_stocks.append(s2)
        pair=(s1,s2)
        selected_pairs_corr.append(pair)

    if len(selected_pairs_corr) == math.comb(len(cluster1_asset_list),2):
        break

opt_asset1=selected_stocks[0:len(selected_stocks)-1:2]
opt_asset2=selected_stocks[1:len(selected_stocks):2]

print(selected_pairs_corr)

###### check which asset pairs are cointegrated from DSCAN cluster 1
coint_pairs=[]
np.random.seed(107)
for i in range(0,len(pairs)):
    score, pvalue, _ = coint(np.cumsum(clusters_norm_ret_df[pairs[i][0]]),np.cumsum(clusters_norm_ret_df[pairs[i][1]]))
    confidence_level = 0.05
    if pvalue < confidence_level:
        coint_pairs.append(pairs[i])
print(coint_pairs)