import numpy as np
import scipy
import scipy.stats
from sklearn import (
    linear_model,
    metrics,
    mixture,
    model_selection,
    neighbors,
)

def knn_accuracy(embedding, labels, n_neighbors=15, verbal=False):
    """
    Calculate KNN classification accuracy.

    Parameters:
    - embedding: Feature vectors
    - labels: Corresponding labels
    - n_neighbors: Number of neighbors for KNN

    Returns:
    - Accuracy score of KNN classification
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        embedding, labels, test_size=.2)

    # Create and train KNN classifier
    knn = neighbors.KNeighborsClassifier(n_neighbors)
    knn_accuracy = knn.fit(X_train, y_train).score(X_test, y_test)

    # Print and return accuracy
    if verbal:
        print(f'kNN Accuracy: {knn_accuracy:.4f}')

    return knn_accuracy


def ari_score(embedding,  true_labels, n_clusters=None, verbal=False):
    """
    Calculate Adjusted Rand Index (ARI) score.

    Parameters:
    - true_labels: Ground truth labels
    - predicted_labels: Predicted cluster labels

    Returns:
    - ARI score
    """
    if n_clusters is None:
        n_clusters = np.unique(true_labels).shape[0]

    gmm = mixture.GaussianMixture(n_components=n_clusters,
                                  covariance_type='diag',
                                  random_state=42)
    gmm.fit(embedding)
    labels_predicted = gmm.predict(embedding)
    ari = metrics.adjusted_rand_score(true_labels, labels_predicted)

    # Print and return ARI score
    if verbal:
        print(f'ARI Score: {ari:.4f}')

    return ari

def corr_pdist(x, y, sample_size=500, seed=0, metric="euclidean", mode="spearmann", verbal=False):
    """
    Computes correlation between pairwise distances among the x's and among the y's
    :param x: data high dim [num_samples, time]
    :param y: data low dim [num_samples, num_dims]
    :param sample_size: number of points to subsample from x and y for pairwise distance computation
    :param seed: random seed
    :param metric: Metric used for distances of x, must be a metric available for sklearn.metrics.pairwise_distances    :return: tuple of Pearson and Spearman correlation coefficient
    :param mode: Can be one of ["spearmann", "pearson"]
    """

    np.random.seed(seed)
    sample_idx = np.random.randint(len(x), size=sample_size)
    x_sample = x[sample_idx]
    y_sample = y[sample_idx]

    x_dists = metrics.pairwise_distances(x_sample, metric=metric).flatten()
    y_dists = metrics.pairwise_distances(y_sample, metric="euclidean").flatten()
    if mode == "pearson":
        corr, _ = scipy.stats.pearsonr(x_dists, y_dists)
    elif mode == "spearmann":
        corr, _ = scipy.stats.spearmanr(x_dists, y_dists)
    if verbal:
        print(f"{mode} corr: {corr}")

    return corr


def score_r_linear(embedding, feature):
    """
    Compute the linear gradient correlation between the embedding and the (biological) feature.
    Args:
        embedding: The embedding to be correlated with the feature.
        feature: The feature to be correlated with the embedding.

    Returns:
        correlation: Correlation coefficient between the rotated embedding and the parameter.

    """
    # compute linear regression to estimate the angle to rotate the embedding space
    regressor = linear_model.LinearRegression()
    regressor.fit(embedding, feature)

    a, b = regressor.coef_
    gradient_angle = np.degrees(np.arctan2(b, a))

    # rotating factor
    theta = -np.radians(gradient_angle)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    # rotate the embedding
    embedding_rotated = embedding @ rotation_matrix.T

    # compute correlation
    correlation, p_value = scipy.stats.pearsonr(embedding_rotated[:, 0], feature)

    return correlation


def score_r_radial(embedding, feature):
    # compute the center of the embedding
    center = np.mean(embedding, axis=0)
    # compute radial distances
    distances = np.sqrt(np.sum((embedding - center) ** 2, axis=1))

    # compute correlation coefficient
    correlation, p_value = scipy.stats.pearsonr(distances, feature)

    if correlation < 0:
        correlation = abs(correlation)

    return correlation


def score_corr_metric(embedding, feature):
    corr_lin = score_r_linear(embedding=embedding, feature=feature)
    corr_rad = score_r_radial(embedding=embedding, feature=feature)
    max_corr = np.max([corr_lin, corr_rad])

    return max_corr

def compute_discriminability(X_in, labels, class1=1, class2=2, epsilon=1e-8):
    """
    Compute a simple discriminability measure between two classes in a PCA-reduced dataset.

    Parameters:
    X_in : np.ndarray
        The input data (samples x features).
    labels : np.ndarray
        Array of class labels corresponding to rows in X_in.
    class1 : int
        Label for the first class.
    class2 : int
        Label for the second class.
    epsilon : float
        Small constant to prevent division by zero.

    Returns:
    float
        The discriminability measure.
    """
    # Extract data for each class
    X_class1 = X_in[labels == class1]
    X_class2 = X_in[labels == class2]

    # Calculate means per feature
    mu1 = np.mean(X_class1, axis=0)
    mu2 = np.mean(X_class2, axis=0)

    # Calculate standard deviations per feature
    std1 = np.std(X_class1, axis=0)
    std2 = np.std(X_class2, axis=0)

    # Get difference
    pooled_std = 0.5 * (std1 + std2)

    # Avoid division by zero by adding a small constant
    normalized_diff = (mu1 - mu2) / (pooled_std + epsilon)

    # Normalize
    discrim = np.linalg.norm(normalized_diff)

    return discrim