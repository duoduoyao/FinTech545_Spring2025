import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import norm, skewnorm
from scipy.special import kv  # Modified Bessel function
from scipy.interpolate import interp1d
import statsmodels.api as sm
import problem1  # import code from problem 1
import problem2  # import code from problem 2


class NormalInverseGaussian:
    """
    Normal Inverse Gaussian distribution implementation

    Parameters:
    - alpha: steepness parameter (α > 0)
    - beta: asymmetry parameter (|β| < α)
    - mu: location parameter (μ ∈ ℝ)
    - delta: scale parameter (δ > 0)
    """

    def __init__(self, alpha=1, beta=0, mu=0, delta=1):
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.delta = delta

        # Validate parameters
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if abs(beta) >= alpha:
            raise ValueError("|beta| must be less than alpha")
        if delta <= 0:
            raise ValueError("delta must be positive")

        # Derived parameters
        self.gamma = np.sqrt(alpha ** 2 - beta ** 2)

    def pdf(self, x):
        """Probability density function"""

        alpha, beta, mu, delta, gamma = self.alpha, self.beta, self.mu, self.delta, self.gamma

        # Convert input to array
        x = np.asarray(x)
        scalar_input = False
        if x.ndim == 0:
            x = x[np.newaxis]
            scalar_input = True

        # Calculate components
        z = delta * gamma
        q = np.sqrt(1 + ((x - mu) / delta) ** 2)

        # PDF formula
        numer = alpha * delta * kv(1, alpha * delta * q) * np.exp(delta * gamma + beta * (x - mu))
        denom = np.pi * q

        result = numer / denom

        if scalar_input:
            return result[0]
        return result

    def fit(self, data):
        """Fit distribution parameters to data using maximum likelihood estimation"""

        def negative_log_likelihood(params):
            alpha, beta, mu, delta = params
            try:
                tmp_dist = NormalInverseGaussian(alpha, beta, mu, delta)
                pdf_values = tmp_dist.pdf(data)
                pdf_values = np.maximum(pdf_values, 1e-10)  # Avoid log(0)
                return -np.sum(np.log(pdf_values))
            except (ValueError, RuntimeWarning, np.linalg.LinAlgError):
                return np.inf

        # Initial estimates
        mu_init = np.mean(data)
        delta_init = np.std(data)
        alpha_init = 1.0
        beta_init = 0.0

        # Parameter bounds
        bounds = [
            (1e-4, None),  # alpha > 0
            (-np.inf, np.inf),  # beta
            (-np.inf, np.inf),  # mu
            (1e-4, None)  # delta > 0
        ]

        # Constraints: |beta| < alpha
        def constraint(params):
            alpha, beta, _, _ = params
            return alpha - abs(beta) - 1e-6

        constraints = {'type': 'ineq', 'fun': constraint}

        # Optimize
        result = minimize(
            negative_log_likelihood,
            [alpha_init, beta_init, mu_init, delta_init],
            bounds=bounds,
            constraints=constraints,
            method='SLSQP'
        )

        # Update parameters
        if result.success:
            self.alpha, self.beta, self.mu, self.delta = result.x
            self.gamma = np.sqrt(self.alpha ** 2 - self.beta ** 2)
        else:
            raise ValueError(f"Fitting failed: {result.message}")

        return self

    def cdf(self, x, n_points=1000):
        """Cumulative distribution function (numerical approximation)"""
        x = np.asarray(x)

        # For scalar input
        if x.ndim == 0 or (x.ndim == 1 and len(x) == 1):
            x_val = float(x.item() if hasattr(x, 'item') else x)
            x_array = np.linspace(x_val - 10, x_val, n_points)
            pdf_values = self.pdf(x_array)
            return np.trapz(pdf_values, x_array)

        # For array input
        result = np.zeros_like(x, dtype=float)
        for i, xi in enumerate(x):
            x_array = np.linspace(xi - 10, xi, n_points)
            pdf_values = self.pdf(x_array)
            result[i] = np.trapz(pdf_values, x_array)

        return result

    def ppf(self, q, n_points=1000):
        """Percent point function (inverse of CDF) - Approximate using interpolation"""
        # Generate a lookup table
        x_min, x_max = -10, 10
        x_values = np.linspace(x_min, x_max, n_points)
        cdf_values = self.cdf(x_values)

        # Interpolate
        ppf_func = interp1d(cdf_values, x_values, bounds_error=False, fill_value=(x_min, x_max))

        return ppf_func(q)

    def rvs(self, size=1, random_state=None):
        """Generate random variates from the distribution"""
        from scipy.stats import invgauss

        if random_state is not None:
            np.random.seed(random_state)

        # Parameters
        alpha, beta, mu, delta, gamma = self.alpha, self.beta, self.mu, self.delta, self.gamma

        # Generate inverse Gaussian random variates
        ig_samples = invgauss.rvs(mu=delta / gamma, scale=delta ** 2, size=size)

        # Generate normal random variates
        z = norm.rvs(size=size)

        # Combine to get NIG random variates
        x = mu + beta * ig_samples + np.sqrt(ig_samples) * z

        return x


class GeneralizedTDistribution:
    """
    Generalized T Distribution

    Parameters:
    - mu: location parameter
    - sigma: scale parameter (σ > 0)
    - p: kurtosis parameter (p > 0)
    - q: kurtosis parameter (q > 0)
    """

    def __init__(self, mu=0, sigma=1, p=2, q=2):
        self.mu = mu
        self.sigma = sigma
        self.p = p
        self.q = q

        # Validate parameters
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if p <= 0:
            raise ValueError("p must be positive")
        if q <= 0:
            raise ValueError("q must be positive")

        # Import special functions
        from scipy.special import beta as beta_func
        self.beta_func = beta_func

    def pdf(self, x):
        """Probability density function"""
        mu, sigma, p, q = self.mu, self.sigma, self.p, self.q

        # Convert input to array
        x = np.asarray(x)
        scalar_input = False
        if x.ndim == 0:
            x = x[np.newaxis]
            scalar_input = True

        # Constants
        c = p / (2 * sigma * q * self.beta_func(p / 2, q))

        # Calculate PDF
        z = (x - mu) / sigma
        pdf = c * (1 + abs(z) ** p / q) ** (-q - 1) * abs(z) ** (p - 1)

        if scalar_input:
            return pdf[0]
        return pdf

    def fit(self, data):
        """Fit distribution parameters to data using maximum likelihood estimation"""

        def negative_log_likelihood(params):
            mu, sigma, p, q = params
            try:
                tmp_dist = GeneralizedTDistribution(mu, sigma, p, q)
                pdf_values = tmp_dist.pdf(data)
                pdf_values = np.maximum(pdf_values, 1e-10)  # Avoid log(0)
                return -np.sum(np.log(pdf_values))
            except (ValueError, RuntimeWarning, np.linalg.LinAlgError):
                return np.inf

        # Initial estimates
        mu_init = np.mean(data)
        sigma_init = np.std(data)
        p_init = 2.0  # Start with t-distribution
        q_init = 2.0

        # Parameter bounds
        bounds = [
            (-np.inf, np.inf),  # mu
            (1e-4, None),  # sigma > 0
            (1e-4, None),  # p > 0
            (1e-4, None)  # q > 0
        ]

        # Optimize
        result = minimize(
            negative_log_likelihood,
            [mu_init, sigma_init, p_init, q_init],
            bounds=bounds,
            method='L-BFGS-B'
        )

        # Update parameters
        if result.success:
            self.mu, self.sigma, self.p, self.q = result.x
        else:
            raise ValueError(f"Fitting failed: {result.message}")

        return self

    def cdf(self, x):
        """Cumulative distribution function (numerical approximation)"""
        x = np.asarray(x)

        # For scalar input
        if x.ndim == 0 or (x.ndim == 1 and len(x) == 1):
            x_val = float(x.item() if hasattr(x, 'item') else x)
            x_grid = np.linspace(-10, x_val, 1000)
            pdf_values = self.pdf(x_grid)
            return np.trapz(pdf_values, x_grid)

        # For array input
        result = np.zeros_like(x, dtype=float)
        for i, xi in enumerate(x):
            x_grid = np.linspace(-10, xi, 1000)
            pdf_values = self.pdf(x_grid)
            result[i] = np.trapz(pdf_values, x_grid)

        return result

    def ppf(self, q):
        """Percent point function (inverse of CDF) - Approximate using interpolation"""
        # Generate a lookup table
        x_values = np.linspace(-10, 10, 1000)
        cdf_values = self.cdf(x_values)

        # Interpolate
        ppf_func = interp1d(cdf_values, x_values, bounds_error=False, fill_value=(-10, 10))

        return ppf_func(q)

    def rvs(self, size=1, random_state=None):
        """Generate random variates from the distribution"""
        if random_state is not None:
            np.random.seed(random_state)

        # Use inverse transform sampling
        u = np.random.uniform(0, 1, size=size)
        return self.ppf(u)
