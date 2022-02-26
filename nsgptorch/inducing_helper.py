from sklearn.cluster import KMeans


def get_cluster_centers(array, n_clusters, random_state=None):
    """
    Get cluster centers from a set of points (1d only).
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(
        array.reshape(-1, 1)
    )
    return kmeans.cluster_centers_
