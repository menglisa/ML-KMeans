#!/usr/bin/env python
# coding: utf-8


# Lisa Meng and Shagun Gupta



#--- PART 1: Implementation from scratch ---


import pandas as pd
import numpy as np
import random
from copy import deepcopy



#--- K-Means Algo ---

data = pd.read_csv('clusters.txt', delimiter=',', header=None, names = ['X','Y']) # data structure for EM w/ GMM
training_data = np.loadtxt('clusters.txt', delimiter=',') # data structure for K-Means
#training_data


# Step 1: Pick k centroids at random
# k=3

def random_centroids(data, k):
    # x-coord of random centroids
    C_x = np.random.randint(np.min(training_data), np.max(training_data), size=k)
    # y-coord of random centroids
    C_y = np.random.randint(np.min(training_data), np.max(training_data), size=k)
    new_centroids = np.array(list(zip(C_x, C_y)), dtype=np.float32)
    return new_centroids


# Euclidean Distance Caculator
def dist(x1, x2, ax=1):
    return np.linalg.norm(x1 - x2, axis=ax)



def k_means_clustering(data, k):
    
    # Step 1: Ramdonly choose centroids
    new_centroids = random_centroids(data,k)
    old_centroids = np.zeros(new_centroids.shape)
    nearest_centroid = np.zeros(len(data))
    convergence = dist(new_centroids, old_centroids, None)
    # Repeat Steps 2 and 3 until convergence (distance from new centroids to old centroid equal 0)
    while convergence != 0:
        # Step 2: Assign each data point to its nearest centroid
        for i in range(len(data)):
            distances = dist(data[i], new_centroids)
            center_membership = np.argmin(distances)
            nearest_centroid[i] = center_membership
        # Step 3: Recompute centroids (by using mean)
        old_centroids = deepcopy(new_centroids)
        for i in range(k):
            total_points = [data[j] for j in range(len(data)) if nearest_centroid[j] == i]
            new_centroids[i] = np.mean(total_points, axis=0)
        convergence = dist(new_centroids, old_centroids, None)
    print(convergence)
    return new_centroids


print(k_means_clustering(training_data, 3))



#--- EM algo with GMM ---


k = 3
 
r = []

min(data['X'])


for i in range(data.shape[0]):
    point = []
    point.append(random.randint(1,10))
    point.append(random.randint(1,10))
    point.append(random.randint(1,10))
    r.append(point)
r = pd.DataFrame(r)
r.rename(columns={0:'ri1',1:'ri2',2:'ri3'}, inplace=True)
r['sum'] = r.apply(lambda x: x['ri1'] + x['ri2'] + x['ri3'], axis = 1)
r['ri1'] = r['ri1']/r['sum']
r['ri2'] = r['ri2']/r['sum']
r['ri3'] = r['ri3']/r['sum']
r['sum'] = r.apply(lambda x: x['ri1'] + x['ri2'] + x['ri3'], axis = 1)
r.drop(labels='sum', inplace=True, axis = 1)
# r



def mu(data, r):
    r.rename(columns={'ri1':0,'ri2':1,'ri3':2}, inplace=True)
    m = []
    for i in range(3):
        mu_x = sum(data['X'] * r[i])/sum(r[i])
        mu_y = sum(data['Y'] * r[i])/sum(r[i])
        m.append([mu_x,mu_y])
    return pd.DataFrame(m).transpose()



def calculate_covariance(data, mu, r):
    covariance = []
    vals = data.transpose().copy(deep = True)
    for i in range(3):
        new_val = pd.DataFrame(np.float64([[0,0],[0,0]]))
        for index in vals.iteritems():
            a = pd.DataFrame(vals[index[0]] - np.float64(mu[i]))
            a.rename(index={'X':0,'Y':1}, inplace=True)
            new_val = new_val + (r[i][index[0]] * a.dot(a.transpose()))
    
        new_val/=sum(r[i])
        covariance.append(new_val)
  
    return covariance



def calculate_pi(r):
    Pis = []
    for column in r:
        pi = (1/len(r))*sum(r[column])
        Pis.append(pi)
    return pd.DataFrame(Pis).transpose()



import math
def normal(x, covar, covar_inv, pi_c, mu):
    r = []
    for i in range(3):
        a = x.transpose()-mu[i]
        expo = -1/2*(a.transpose()).dot(covar_inv[i]).dot(a)
        n = 1/2/(math.pi)/(math.sqrt(np.linalg.det(covar[i])))*math.exp(expo)
        r.append(n)
    return r



def recompute_ric(data, mu_vals, covariance, pi_c, r):
    covar_inv = []
    data = data.rename(columns={'X':0,'Y':1})
    mu_vals = pd.DataFrame(np.float64(mu_vals))
    for i in range(3):
        df_inv = pd.DataFrame(np.linalg.inv(covariance[i].values), covariance[i].columns, covariance[i].index)
        covar_inv.append(df_inv)
    numerator = pd.DataFrame(data.apply(lambda x: normal(x, covariance, covar_inv, pi_c, mu_vals), axis=1).to_list())
    for i in range(3):
        numerator[i] = numerator[i].apply(lambda x: x*pi_c[i])
    numerator['sum'] = numerator.apply(lambda x: sum(x/sum(x)), axis = 1)
    if sum(numerator['sum']) != 150:
        print('not 1')
        print(numerator)
        return None
    numerator.drop(columns='sum', inplace=True)
    new_r = numerator.apply(lambda x: x/sum(x), axis=1)
    new_r.rename(columns={0:'ri1',1:'ri2',2:'ri3'}, inplace=True)
    return new_r



x = 0
while(x < 100):
    prev_r = r.copy(deep=True)
    mu_vals = mu(data,prev_r)
    covariance = calculate_covariance(data, mu_vals,prev_r)
    pi_c = calculate_pi(prev_r)
    r = recompute_ric(data,mu_vals,covariance, pi_c, prev_r)
    if not isinstance(r,pd.DataFrame):
        break
    print(x,'------------------------------------------------------------------------')
    if r.equals(prev_r):
        break
    x += 1
print(mu_vals)
print(covariance)
print(pi_c)



import matplotlib.pyplot as plt
# %matplotlib inline
cmap = plt.cm.Spectral
norm = plt.Normalize(vmin=4, vmax=5)
data['cluster'] = r.idxmax(axis=1)
data = data.replace({'cluster':{'ri1':'red','ri2':'green','ri3':'blue'}})
# data['cluster']
z = np.array(data['cluster'])
# z
data.plot.scatter(x='X',y='Y',c=z)



# label cluster membership
r.idxmax(axis=1)



#--- PART 2: Implementation with library ---



import matplotlib.pyplot as plt
# %matplotlib inline
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(training_data[:,0],training_data[:,1])



#--- K-Means algo ---

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(training_data)



centroids = kmeans.cluster_centers_
print(centroids)



centroid_label = kmeans.labels_
print(centroid_label)



plt.scatter(training_data[:,0], training_data[:,1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')



#--- EM algo with GMM -- 

from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3)
gmm.fit(training_data)



centroid_labels = gmm.predict(training_data)
print(centroid_labels)
plt.scatter(training_data[:, 0], training_data[:, 1], c=centroid_labels, cmap='rainbow')



# probabilistic cluster assignments
probs = gmm.predict_proba(training_data)
print(probs.round(3))



mus = gmm.means_
print(mus)



sigmas = gmm.covariances_
print(sigmas)



# amplitude
pis = gmm.weights_
print(pis)


# Was convergence reached?
print(gmm.converged_)

# How many itereations to reach convergence?
print(gmm.n_iter_)




