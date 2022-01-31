import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('income.csv')

# visually check the distribution to select number of clusters
plt.scatter(df['Age'],df['Income($)'])

# create K means cluster obj with 3 clusters
km = KMeans(n_clusters=3)
# predict cluster based on age and income
df['Cluster'] = km.fit_predict(df[['Age','Income($)']])

# subset each cluster into different df
df0 = df[df.Cluster==0]
df1 = df[df.Cluster==1]
df2 = df[df.Cluster==2]

# plot different dfs
plt.scatter(df0['Age'],df0['Income($)'], color='green')
plt.scatter(df1['Age'],df1['Income($)'], color='red')
plt.scatter(df2['Age'],df2['Income($)'], color='blue')
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.legend(['Cluster 0','Cluster 1','Cluster 2'])

# clusters do not look optimal, scale data as a solution

# copy df to scale it
df_s = df
# scale data
scaler = MinMaxScaler()
scaler.fit(df_s[['Income($)']])
df_s['Income($)'] = scaler.transform(df[['Income($)']])
scaler.fit(df[['Age']])
df_s['Age'] = scaler.transform(df[['Age']])
# drop name col
df_s = df_s.drop('Name',axis='columns')

# create K means cluster obj with 3 clusters
km = KMeans(n_clusters=3)
# predict cluster based on scaled age and scaled income
df_s['cluster'] = km.fit_predict(df_s)

# subset each cluster into different df
df0 = df_s[df_s.cluster==0]
df1 = df_s[df_s.cluster==1]
df2 = df_s[df_s.cluster==2]

# plot different dfs
plt.scatter(df0['Age'],df0['Income($)'], color='green')
plt.scatter(df1['Age'],df1['Income($)'], color='red')
plt.scatter(df2['Age'],df2['Income($)'], color='blue')
# add cluster centers by taking the x and y of each (:) row
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color='yellow')
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.legend(['Cluster 0','Cluster 1','Cluster 2','Cluster centers'])

# clusters look optimal

# elbow technique to find optimal n of K
# create empty standard squared errors array
sse = []
# select k range to test, must start from 1 (cannot be 0 k)
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df_s)
    sse.append(km.inertia_)
# plot sse and k to find elbow
plt.plot(k_rng,sse)
plt.xlabel('k')
plt.ylabel('sse')
# elbow is at 3 k

