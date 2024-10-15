import numpy as np

def load_data():
    X = np.load("data/ex7_X.npy")
    return X

def compute_centroids(X, idx, K):
    """
    Computes the new centroids by calculating the means of the data points assigned to each centroid.
    """
    m, n = X.shape
    centroids = np.zeros((K, n))

    for k in range(K):
        points = X[idx == k]
        if points.size > 0:
            centroids[k] = np.mean(points, axis=0)

    return centroids

def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example using vectorization.
    """
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)  # (m, K)
    return np.argmin(distances, axis=1)

def run_kMeans(X, initial_centroids, max_iters=10):
    """
    Runs the K-Means algorithm on the data matrix X.
    """
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    idx = np.zeros(m)

    for i in range(max_iters):
        print(f"K-Means iteration {i + 1}/{max_iters}")

        # Assign closest centroids
        idx = find_closest_centroids(X, centroids)

        # Compute new centroids
        centroids = compute_centroids(X, idx, K)

    return centroids, idx

def kMeans_init_centroids(X, K):
    """
    Initializes K centroids randomly from the dataset X.
    """
    randidx = np.random.permutation(X.shape[0])
    centroids = X[randidx[:K]]
    return centroids
