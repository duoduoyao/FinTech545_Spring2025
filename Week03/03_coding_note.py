import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, eigvals
from scipy.stats import multivariate_normal
from numpy.linalg import eigh

# Cholesky that assumes PD matrix
def chol_pd(root, a):
    n = a.shape[0]
    # Initialize the root matrix with 0 values
    root.fill(0.0)

    # Loop over columns
    for j in range(n):
        s = 0.0
        # If we are not on the first column, calculate the dot product of the preceding row values.
        if j > 0:
            s = np.dot(root[j, :j], root[j, :j])

        # Diagonal Element
        root[j, j] = np.sqrt(a[j, j] - s)

        ir = 1.0 / root[j, j]
        # Update off-diagonal rows of the column
        for i in range(j + 1, n):
            s = np.dot(root[i, :j], root[j, :j])
            root[i, j] = (a[i, j] - s) * ir

# Cholesky that assumes PSD
def chol_psd(root, a):
    n = a.shape[0]
    # Initialize the root matrix with 0 values
    root.fill(0.0)

    # Loop over columns
    for j in range(n):
        s = 0.0
        # If we are not on the first column, calculate the dot product of the preceding row values.
        if j > 0:
            s = np.dot(root[j, :j], root[j, :j])

        # Diagonal Element
        temp = a[j, j] - s
        if -1e-8 <= temp <= 0:
            temp = 0.0
        root[j, j] = np.sqrt(temp)

        # Check for the 0 eigenvalue. The column will already be 0, move to the next column
        if root[j, j] != 0.0:
            # Update off-diagonal rows of the column
            ir = 1.0 / root[j, j]
            for i in range(j + 1, n):
                s = np.dot(root[i, :j], root[j, :j])
                root[i, j] = (a[i, j] - s) * ir

# Generate some random numbers with missing values
def generate_with_missing(n, m, pmiss=0.25):
    x = np.empty((n, m), dtype=object)
    for i in range(n):
        for j in range(m):
            if np.random.rand() >= pmiss:
                x[i, j] = np.random.randn()
            else:
                x[i, j] = np.nan
    return x

# Calculate either the covariance or correlation function when there are missing values
def missing_cov(x, skip_miss=True, fun=np.cov):
    n, m = x.shape
    n_miss = np.sum(np.isnan(x), axis=0)

    # Nothing missing, just calculate it.
    if np.sum(n_miss) == 0:
        return fun(x, rowvar=False)

    idx_missing = [set(np.where(np.isnan(x[:, i]))[0]) for i in range(m)]

    if skip_miss:
        # Skipping Missing, get all the rows which have values and calculate the covariance
        rows = set(range(n))
        for c in range(m):
            rows -= idx_missing[c]
        rows = sorted(rows)
        return fun(x[rows, :], rowvar=False)
    else:
        # Pairwise, for each cell, calculate the covariance.
        out = np.zeros((m, m))
        for i in range(m):
            for j in range(i + 1):
                rows = set(range(n))
                for c in (i, j):
                    rows -= idx_missing[c]
                rows = sorted(rows)
                out[i, j] = fun(x[rows, :][:, [i, j]], rowvar=False)[0, 1]
                if i != j:
                    out[j, i] = out[i, j]
        return out

# Near PSD Matrix
def near_psd(a, epsilon=0.0):
    n = a.shape[0]

    inv_sd = None
    out = a.copy()

    # Calculate the correlation matrix if we got a covariance
    if np.sum(np.diag(out) == 1.0) != n:
        inv_sd = np.diag(1 / np.sqrt(np.diag(out)))
        out = inv_sd @ out @ inv_sd

    # SVD, update the eigenvalue and scale
    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals, epsilon)
    T = 1 / (vecs**2 @ vals)
    T = np.diag(np.sqrt(T))
    l = np.diag(np.sqrt(vals))
    B = T @ vecs @ l
    out = B @ B.T

    # Add back the variance
    if inv_sd is not None:
        inv_sd = np.diag(1 / np.diag(inv_sd))
        out = inv_sd @ out @ inv_sd
    return out

# PCA
def simulate_pca(a, nsim, nval=None):
    # Eigenvalue decomposition
    vals, vecs = np.linalg.eigh(a)

    # Flip them and the vectors
    flip = np.arange(vals.shape[0] - 1, -1, -1)
    vals = vals[flip]
    vecs = vecs[:, flip]

    tv = np.sum(vals)

    posv = np.where(vals >= 1e-8)[0]
    if nval is not None:
        if nval < len(posv):
            posv = posv[:nval]
    vals = vals[posv]
    vecs = vecs[:, posv]

    print(f"Simulating with {len(posv)} PC Factors: {np.sum(vals) / tv * 100}% total variance explained")
    B = vecs @ np.diag(np.sqrt(vals))

    m = len(vals)
    r = np.random.randn(m, nsim)

    return (B @ r).T

# Example usage
n = 5
sigma = np.full((n, n), 0.9)
np.fill_diagonal(sigma, 1.0)

root = np.empty_like(sigma)
chol_pd(root, sigma)

print(np.allclose(root @ root.T, sigma))

root2 = cholesky(sigma, lower=True)
print(np.allclose(root, root2))

# Make the matrix PSD
sigma[0, 1] = 1.0
sigma[1, 0] = 1.0
print(eigvals(sigma))

chol_psd(root, sigma)
print(np.allclose(root @ root.T, sigma))

# Generate some random numbers with missing values
np.random.seed(2)
x = generate_with_missing(10, 5, pmiss=0.2)
print(np.cov(x, rowvar=False))

skip_miss = missing_cov(x)
pairwise = missing_cov(x, skip_miss=False)
print(eigvals(pairwise))

chol_psd(root, skip_miss)
chol_psd(root, pairwise)

# Near PSD Matrix
near_pairwise = near_psd(pairwise)
chol_psd(root, near_pairwise)

# PCA
vals, vecs = eigh(near_pairwise)
tv = np.sum(vals)
# Keep values 2:5
vals = vals[2:5]
vecs = vecs[:, 2:5]
B = vecs @ np.diag(np.sqrt(vals))
r = (B @ np.random.randn(3, 100000000)).T
print(np.cov(r, rowvar=False))

# Simulate PCA
n = 5
sigma = np.full((n, n), 0.9)
np.fill_diagonal(sigma, 1.0)

sigma[0, 1] = 1
sigma[1, 0] = 1

v = np.diag(np.full(n, 0.5))
sigma = v @ sigma @ v

sim = simulate_pca(sigma, 10000)
print(np.cov(sim, rowvar=False))

sim = simulate_pca(sigma, 10000, nval=3)
print(np.cov(sim, rowvar=False))

sim = simulate_pca(sigma, 10000, nval=2)
print(np.cov(sim, rowvar=False))