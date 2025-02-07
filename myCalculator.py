import numpy as np
import pandas as pd
from scipy.linalg import eigh

# A. Calculate the pairwise covariance matrix
def calculate_covariance_matrix(data):
    return data.cov()

# B. Check if the matrix is positive semi-definite
def is_positive_semi_definite(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    return np.all(eigenvalues >= 0), eigenvalues

# C. Find the nearest positive semi-definite matrix using Higham’s method
def nearest_psd_higham(matrix):
    # Compute the symmetric part of the matrix
    sym_matrix = (matrix + matrix.T) / 2
    # Perform eigen-decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(sym_matrix)
    # Replace negative eigenvalues with zero
    eigenvalues[eigenvalues < 0] = 0
    # Reconstruct the matrix
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

# D. Calculate covariance matrix using only overlapping data
def covariance_overlapping(data):
    overlapping_data = data.dropna()  # Drop rows with missing values
    return overlapping_data.cov()

# Main script
# Load the data (assuming the file is named 'problem2.csv')
data = pd.read_csv("problem2.csv")
# 1. Calculate covariance matrix
cov_matrix = calculate_covariance_matrix(data)
# 2. Check if the matrix is positive semi-definite
psd_status, eigenvalues = is_positive_semi_definite(cov_matrix)
print("Is the covariance matrix PSD?", psd_status)
# 3. Find the nearest PSD matrix using Higham's method
nearest_psd_matrix = nearest_psd_higham(cov_matrix)

# 4. Calculate covariance matrix using overlapping data
cov_matrix_overlapping = covariance_overlapping(data)

# 5. Compare the results
print("Original Covariance Matrix:")
print(cov_matrix)

print("Nearest PSD Matrix (Higham):")
print(nearest_psd_matrix)

print("Covariance Matrix Using Overlapping Data:")
print(cov_matrix_overlapping)


def exponential_weights(n, lambda_):
    """
    Generate exponential weights for a time series of length n.

    Parameters:
        n (int): Length of the time series.
        lambda_ (float): Decay factor (0 < lambda_ < 1).

    Returns:
        weights (np.array): Array of exponential weights.
    """
    weights = (1 - lambda_) * (lambda_ ** np.arange(n - 1, -1, -1))
    weights /= weights.sum()  # Normalize weights to sum to 1
    return weights


def exponential_weighted_covariance(data, lambda_):
    """
    Calculate the exponentially weighted covariance matrix.

    Parameters:
        data (np.array or pd.DataFrame): Time series data (rows = observations, columns = variables).
        lambda_ (float): Decay factor (0 < lambda_ < 1).

    Returns:
        cov_matrix (np.array): Exponentially weighted covariance matrix.
    """
    if isinstance(data, pd.DataFrame):
        data = data.values

    n, m = data.shape
    weights = exponential_weights(n, lambda_)

    # Center the data (subtract the mean)
    centered_data = data - np.mean(data, axis=0)

    # Calculate the weighted covariance matrix
    cov_matrix = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            cov_matrix[i, j] = np.sum(weights * centered_data[:, i] * centered_data[:, j])

    return cov_matrix


# Example Usage
if __name__ == "__main__":
    # Example data (rows = time, columns = variables)
    test_data = pd.read_csv("testfiles/data/test2.csv")
    lambda_val = 0.97

    # Calculate the exponentially weighted covariance matrix
    ew_cov = exponential_weighted_covariance(test_data, lambda_val)
    print("Exponentially Weighted Covariance Matrix:")
    print(ew_cov)