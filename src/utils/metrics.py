import numpy as np


def maximum_mean_discrepancy(x, y):
    """Computes the Maximum Mean Discrepancy (MMD) between two sets of samples x and y.

    Args:
    - x, y: NumPy arrays of shape (n_samples, n_features) representing samples.

    Returns:
    - Float value representing the MMD between x and y.
    """

    def _gaussian_kernel(x, y, sigma=1.0):
        """Computes the Gaussian kernel similarity between two sets of samples x and y.

        Args:
        - x, y: NumPy arrays of shape (n_samples, n_features) representing samples.
        - sigma: Float, bandwidth parameter for the Gaussian kernel.

        Returns:
        - Gram matrix of shape (n_samples_x, n_samples_y) containing pairwise
        Gaussian kernel similarities between samples in x and y.
        """
        n_samples_x = x.shape[0]
        n_samples_y = y.shape[0]

        K = np.zeros((n_samples_x, n_samples_y))

        for i in range(n_samples_x):
            for j in range(n_samples_y):
                d = np.linalg.norm(x[i] - y[j])
                K[i, j] = np.exp(-(d**2) / (2 * (sigma**2)))

        return K

    x_kernel = _gaussian_kernel(x, x)
    y_kernel = _gaussian_kernel(y, y)
    xy_kernel = _gaussian_kernel(x, y)

    mmd = np.mean(x_kernel) + np.mean(y_kernel) - 2 * np.mean(xy_kernel)
    return mmd


# Example usage:
# Assuming x and y are NumPy arrays representing samples from source and target domains, respectively
# Calculate MMD between x and y
# mmd_value = maximum_mean_discrepancy(x, y)
