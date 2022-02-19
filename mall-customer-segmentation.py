#dataset to dataframe library
import pandas as pd
#plotting libraries
import matplotlib.pyplot as plt
import plotly.graph_objects as go


'''Reading dataset and selecting age, annual income and spending score columns'''
dataset = pd.read_csv(r'Dataset 3.csv')
X = dataset.iloc[:, 2:5].values

'''Using KMeans clustering to find the optimal number of clusters'''
#import KMeans algorithm
from sklearn.cluster import KMeans
#create within-cluster sum of squares (inertia) list
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters= i, init='k-means++', random_state=0)
    #compute KMeans clustering
    kmeans.fit(X)
    #add KMeans inertia to wcss list
    wcss.append(kmeans.inertia_)

'''Elbow method visualization'''
#change default style to fivethirtyeight style
plt.style.use('fivethirtyeight')
#plot number of clusters (x axis) and wcss (y axis)
plt.plot(range(1,11), wcss, 'o')
plt.plot(range(1,11), wcss, '-', alpha=0.5)
#add plot info
plt.title('The Elbow Method')
plt.xlabel('no of clusters')
plt.ylabel('wcss')
#save and show plot
plt.savefig('figs/elbow.png', bbox_inches='tight')
plt.show()

'''Model building using KMeans clustering with 5 clusters (k=5 is the optimal number of clusters according to elbow method)'''
kmeansmodel = KMeans(n_clusters= 5, init='k-means++', random_state=0)
kmeansmodel.fit_predict(X)
labels = kmeansmodel.labels_
centroids = kmeansmodel.cluster_centers_

'''Clusters 3d visualization using graph objects'''
#find the data trace
trace = go.Scatter3d(name='data', text=dataset.iloc[:, 0].values, hoverinfo='text', x=dataset['Age'], y=dataset['Spending Score (1-100)'], z=dataset['Annual Income (k$)'], mode='markers', marker=dict(color=labels))
#find the centroids trace
ctrace = go.Scatter3d(name='centroids', x=centroids[:, 0], y=centroids[:, 1], z=centroids[:, 2], opacity=0.5, mode='markers', marker=dict(size=15, color='black'))
#find the layout given the title and scene options
layout = go.Layout(title='Clusters', scene=dict(xaxis = dict(title  = 'Age'), yaxis = dict(title  = 'Spending Score'), zaxis = dict(title  = 'Annual Income')))
#create 3d figure using trace and layout
fig = go.Figure(data=[trace, ctrace],layout=layout)
#save and show 3d figure
fig.write_html('figs/3dmodel.html')
fig.show()
