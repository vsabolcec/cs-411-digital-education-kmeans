# import numpy as np
import random
import math

# Numpy:
# def generate_data():
#     np.random.seed(7)
#     x1 = np.random.standard_normal((100,2))*0.6+np.ones((100,2))
#     x2 = np.random.standard_normal((100,2))*0.5-np.ones((100,2))
#     x3 = np.random.standard_normal((100,2))*0.4-2*np.ones((100,2))+5
#     X = np.concatenate((x1,x2,x3),axis=0) 
#     return X

def generate_data():
    random.seed(7)
    x1 = [[random.gauss(0, 1)*0.6 + 1, random.gauss(0, 1)*0.6 + 1] for _ in range(100)]
    x2 = [[random.gauss(0, 1)*0.5 - 1, random.gauss(0, 1)*0.5 - 1] for _ in range(100)]
    x3 = [[random.gauss(0, 1)*0.4 + 3, random.gauss(0, 1)*0.4 + 3] for _ in range(100)]
    x = x1 + x2 + x3
    return x

# Numpy:
# def generate_centroids(k):
#     np.random.seed(2)
#     cx = np.random.rand(k)*6 - 2
#     cy = np.random.rand(k)*5 - 2
#     centroids = np.zeros((k,2))
#     centroids[:,0] = cx
#     centroids[:,1] = cy
#     return centroids

def generate_centroids(k):
    random.seed(2)
    cx = [random.uniform(-2, 4) for _ in range(k)]
    cy = [random.uniform(-2, 3) for _ in range(k)]
    centroids = [[x, y] for x, y in zip(cx, cy)]
    return centroids

# Numpy:
# def eucledian_distance(point1, point2):
#     return np.linalg.norm(point1 - point2)

def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Numpy
# def plot_data(X,labels,centroids,s,ax):
#     ax.plot(X[labels==9,0],X[labels==9,1],'k.')
#     ax.plot(X[labels==0,0],X[labels==0,1],'r.', label='cluster 1')
#     ax.plot(X[labels==1,0],X[labels==1,1],'b.', label='cluster 2')
#     ax.plot(X[labels==2,0],X[labels==2,1],'g.', label='cluster 3')
#     ax.plot(centroids[:,0],centroids[:,1],'mo',markersize=8, label='centroids')
#     ax.legend()
#     ax.set_title(s)

def plot_data(X, labels, centroids, s, ax):
    ax.plot([point[0] for point in X if labels[X.index(point)] == 9], [point[1] for point in X if labels[X.index(point)] == 9], 'k.')
    ax.plot([point[0] for point in X if labels[X.index(point)] == 0], [point[1] for point in X if labels[X.index(point)] == 0], 'r.', label='cluster 1')
    ax.plot([point[0] for point in X if labels[X.index(point)] == 1], [point[1] for point in X if labels[X.index(point)] == 1], 'b.', label='cluster 2')
    ax.plot([point[0] for point in X if labels[X.index(point)] == 2], [point[1] for point in X if labels[X.index(point)] == 2], 'g.', label='cluster 3')
    ax.plot([centroid[0] for centroid in centroids], [centroid[1] for centroid in centroids], 'mo', markersize=8, label='centroids')
    ax.legend()
    ax.set_title(s)


# fig, axes = plt.subplots(1, 4, figsize=(30,5))
# 
# plot_data(data, np.array([9]*len(data)), centroids, 'Initialize cluster centers for K = 3', axes[0])
# labels = assignment_step(data, centroids)
# plot_data(data, labels, centroids, 'Assign points to nearest cluster center\n 1st iteration', axes[1])
# centroids = update_step(data, labels, centroids)
# plot_data(data, labels, centroids, 'Update cluster centers\n 1st iteration', axes[2])
# labels = assignment_step(data, centroids)
# plot_data(data, labels, centroids, 'Assign points to nearest cluster center\n 2nd iteration', axes[0])
# centroids = update_step(data, labels, centroids)
# plot_data(data, labels, centroids, 'Update cluster centers\n 2nd iteration', axes[1])
# labels = assignment_step(data, centroids)
# plot_data(data, labels, centroids, 'Assign points to nearest cluster center\n 3rd iteration', axes[2])
# centroids = update_step(data, labels, centroids)
# plot_data(data, labels, centroids, 'Update cluster centers\n 3rd iteration', axes[3])
# plt.show()