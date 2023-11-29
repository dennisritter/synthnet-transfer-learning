import numpy as np
from scipy.special import kl_div
from scipy.stats import gaussian_kde
from torchmetrics.image.kid import KernelInceptionDistance


def _kernel_matrix(X, Y, kernel_func):
    """Calculates the kernel matrix for two input datasets X and Y using the given kernel function."""
    n = X.shape[0]
    m = Y.shape[0]
    K = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            K[i, j] = kernel_func(X[i], Y[j])
    return K


def _gaussian_kernel(x, y, sigma=1.0):
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


def kl_divergence(source_feature_vectors, target_feature_vectors, num_samples=2048):
    """Calculate KL divergence between distributions estimated from source and target feature vectors.

    Arguments:
    source_feature_vectors -- Feature vectors from the source domain (numpy array)
    target_feature_vectors -- Feature vectors from the target domain (numpy array)
    num_samples -- Number of samples for KDE estimation (default: 1000)

    Returns:
    kl_divergence -- KL divergence value
    """
    # Randomly sample points for KDE estimation
    source_samples = source_feature_vectors[
        np.random.choice(source_feature_vectors.shape[0], num_samples, replace=False)
    ]
    target_samples = target_feature_vectors[
        np.random.choice(target_feature_vectors.shape[0], num_samples, replace=False)
    ]

    # Perform KDE on source and target feature vectors
    source_kde = gaussian_kde(source_samples.T)
    target_kde = gaussian_kde(target_samples.T)

    source_density = source_kde(source_samples.T)
    target_density = target_kde(target_samples.T)

    # Compute KL divergence
    kl_divergence = kl_div(source_density, target_density)

    return np.sum(kl_divergence)


def calc_kid(model, train_loader, test_loader, debug=False):
    subset_size = min(len(train_loader), len(test_loader)) - 1 if debug else 1000
    kid = KernelInceptionDistance(feature=model, subset_size=subset_size)
    for batch in train_loader:
        train_imgs, _, _ = batch
        kid.update(train_imgs, real=False)
    for batch in test_loader:
        test_imgs, _, _ = batch
        kid.update(test_imgs, real=True)
    return kid.compute()
