import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go


'''Reading dataset and selecting age, annual income and spending score columns'''
dataset = pd.read_csv(r'Dataset 3.csv')
X = dataset.iloc[:, 2:5].values

'''Using KMeans clustering to find the optimal number of clusters'''
from sklearn.cluster import KMeans
#within-cluster sum of squares (inertia)
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters= i, init='k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

'''Elbow method visualization'''
plt.style.use('fivethirtyeight')
plt.plot(range(1,11), wcss, 'o')
plt.plot(range(1,11), wcss, '-', alpha=0.5)
plt.title('The Elbow Method')
plt.xlabel('no of clusters')
plt.ylabel('wcss')
plt.savefig('figs/elbow.png', bbox_inches='tight')
plt.show()

'''Model building using KMeans clustering with 5 clusters (k=5 is the optimal number of clusters according to elbow method)'''
kmeansmodel = KMeans(n_clusters= 5, init='k-means++', random_state=0)
kmeansmodel.fit_predict(X)
labels = kmeansmodel.labels_
centroids = kmeansmodel.cluster_centers_

'''Clusters 3d visualization using graph objects'''
trace = go.Scatter3d(name='data', text=dataset.iloc[:, 0].values, hoverinfo='text', x=dataset['Age'], y=dataset['Spending Score (1-100)'], z=dataset['Annual Income (k$)'], mode='markers', marker=dict(color=labels))
ctrace = go.Scatter3d(name='centroids', x=centroids[:, 0], y=centroids[:, 1], z=centroids[:, 2], opacity=0.5, mode='markers', marker=dict(size=15, color='black'))
layout = go.Layout(title='Clusters', scene=dict(xaxis = dict(title  = 'Age'), yaxis = dict(title  = 'Spending Score'), zaxis = dict(title  = 'Annual Income')))
fig = go.Figure(data=[trace, ctrace],layout=layout)
fig.write_html('figs/3dmodel.html')
fig.show()
