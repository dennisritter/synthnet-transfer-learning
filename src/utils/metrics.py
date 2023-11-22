import numpy as np
from scipy.special import kl_div
from scipy.stats import gaussian_kde


def _kernel_matrix(X, Y, kernel_func):
    """Calculates the kernel matrix for two input datasets X and Y using the given kernel function."""
    n = X.shape[0]
    m = Y.shape[0]
    K = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            K[i, j] = kernel_func(X[i], Y[j])
    return K


def _gaussian_kernel(x, y, sigma=1):
    """Calculates the Gaussian kernel between two input vectors x and y with the given sigma value."""
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma**2))


def mmd(X, Y, kernel_func=_gaussian_kernel):
    """Calculates the maximum mean discrepancy (MMD) between two input datasets X and Y using the given kernel
    function."""
    K_XX = _kernel_matrix(X, X, kernel_func)
    K_YY = _kernel_matrix(Y, Y, kernel_func)
    K_XY = _kernel_matrix(X, Y, kernel_func)
    mmd = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
    return mmd


def kl_divergence(source_feature_vectors, target_feature_vectors):
    """Calculate KL divergence between distributions estimated from source and target feature vectors.

    Arguments:
    source_feature_vectors -- Feature vectors from the source domain (numpy array)
    target_feature_vectors -- Feature vectors from the target domain (numpy array)

    Returns:
    kl_divergence -- KL divergence value
    """
    # Perform KDE on source and target feature vectors
    source_kde = gaussian_kde(source_feature_vectors.T)
    target_kde = gaussian_kde(target_feature_vectors.T)

    source_density = source_kde(source_feature_vectors.T)
    target_density = target_kde(target_feature_vectors.T)

    # Compute KL divergence
    kl_divergence = kl_div(source_density, target_density)

    return kl_divergence
