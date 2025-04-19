# Part 4: Advanced Distribution Risk Models and VaR/ES Analysis
import pandas as pd
import numpy as np
from scipy.special import gamma
from statsmodels.distributions.empirical_distribution import ECDF
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from scipy.optimize import minimize
from scipy.special import kv
from scipy.integrate import quad




# First, read and process the portfolio data
# Step 1: Load the data
daily_prices = pd.read_csv('../Projects/Final Project/DailyPrices.csv')
initial_portfolio = pd.read_csv('../Projects/Final Project/initial_portfolio.csv')
risk_free = pd.read_csv('../Projects/Final Project/rf.csv')

# Convert dates to datetime format
daily_prices['Date'] = pd.to_datetime(daily_prices['Date'])
risk_free['Date'] = pd.to_datetime(risk_free['Date'])

# Step 2: Split data into training (2023) and testing (2024-2025) periods
training_data = daily_prices[daily_prices['Date'] <= pd.Timestamp('2023-12-29')]
testing_data = daily_prices[daily_prices['Date'] >= pd.Timestamp('2023-12-29')]


# Step 3: Calculate returns
# Function to calculate daily returns
def calculate_returns(prices_df):
    returns_df = prices_df.copy()
    for column in prices_df.columns:
        if column != 'Date':
            returns_df[column] = prices_df[column].pct_change()
    return returns_df.dropna()


# Calculate returns for training and testing periods
training_returns = calculate_returns(training_data)
testing_returns = calculate_returns(testing_data)


# # Get unique portfolio names
portfolios = initial_portfolio['Portfolio'].unique().tolist()
print(f"Unique portfolios: {portfolios}")

# Calculate the total value of each portfolio
portfolio_values = {}
for portfolio in portfolios:
    portfolio_holdings = initial_portfolio[initial_portfolio['Portfolio'] == portfolio]
    total_value = portfolio_holdings['Holding'].sum()
    portfolio_values[portfolio] = total_value

# Create the portfolio_weights dictionary with the correct structure
portfolio_weights = {}
for portfolio in portfolios:
    portfolio_weights[portfolio] = {}

    # Get holdings for this portfolio
    portfolio_holdings = initial_portfolio[initial_portfolio['Portfolio'] == portfolio]

    # Calculate total portfolio value
    total_value = portfolio_holdings['Holding'].sum()

    # Calculate weight for each symbol
    for _, row in portfolio_holdings.iterrows():
        symbol = row['Symbol']
        holding = row['Holding']
        weight = holding / total_value if total_value > 0 else 0

        portfolio_weights[portfolio][symbol] = {
            'weight': weight,
            'holding': holding
        }

print("Portfolio weights successfully created.")

class NormalInverseGaussian:
    """
    Custom implementation of the Normal Inverse Gaussian distribution.
    """
    def __init__(self, alpha, beta, mu, delta):
        # Parameter checks
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if abs(beta) >= alpha:
            raise ValueError("abs(beta) must be less than alpha")
        if delta <= 0:
            raise ValueError("delta must be positive")

        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.delta = delta
        # Derived parameter
        self.gamma = np.sqrt(alpha**2 - beta**2)

    def pdf(self, x):
        """Probability density function"""
        alpha, beta, mu, delta = self.alpha, self.beta, self.mu, self.delta
        gamma = self.gamma

        # Handle array input
        if np.isscalar(x):
            x = np.array([x])
        else:
            x = np.asarray(x)

        # Calculate components
        arg = alpha * np.sqrt(delta**2 + (x - mu)**2)

        # Calculate PDF
        pdf_values = (alpha * delta * kv(1, arg) *
                      np.exp(delta * gamma + beta * (x - mu)) /
                      (np.pi * np.sqrt(delta**2 + (x - mu)**2)))

        # Handle potential numerical issues
        pdf_values = np.maximum(pdf_values, 1e-300)

        if len(pdf_values) == 1:
            return pdf_values[0]
        return pdf_values

    def fit(self, data):
        """Fit distribution parameters using MLE"""
        data = np.asarray(data)

        # Define negative log-likelihood function
        def neg_loglikelihood(params):
            alpha, beta, mu, delta = params
            if alpha <= 0 or delta <= 0 or abs(beta) >= alpha:
                return np.inf

            try:
                model = NormalInverseGaussian(alpha, beta, mu, delta)
                pdf_values = model.pdf(data)
                # Handle numerical issues
                pdf_values = np.maximum(pdf_values, 1e-300)
                return -np.sum(np.log(pdf_values))
            except:
                return np.inf

        # Initial parameter estimates
        mean = np.mean(data)
        var = np.var(data)
        skew = stats.skew(data)
        kurtosis = stats.kurtosis(data, fisher=False)

        # Initial estimates
        try:
            if kurtosis > 3:  # Must be leptokurtic for NIG
                delta_init = 3 * var / (kurtosis - 3)
                alpha_init = np.sqrt(3 * kurtosis / (var * (kurtosis - 3)))
                beta_init = skew / (var * np.sqrt(kurtosis - 3)) if skew != 0 else 0
                mu_init = mean - beta_init * delta_init / np.sqrt(alpha_init**2 - beta_init**2)
            else:
                # Fallback if kurtosis doesn't meet requirements
                delta_init = var
                alpha_init = 2.0 / np.sqrt(var)
                beta_init = skew / (2.0 * var) if skew != 0 else 0
                mu_init = mean
        except:
            # Simple fallback if moment-based estimates fail
            delta_init = np.std(data)
            alpha_init = 1.5 / delta_init
            beta_init = 0
            mu_init = mean

        # Ensure alpha > |beta|
        if abs(beta_init) >= alpha_init:
            alpha_init = abs(beta_init) + 0.1

        # Initial parameters
        initial_params = [alpha_init, beta_init, mu_init, delta_init]

        # Optimize
        try:
            result = minimize(neg_loglikelihood, initial_params,
                             method='Nelder-Mead',
                             bounds=[(0.001, None), (None, None), (None, None), (0.001, None)])

            if result.success:
                alpha, beta, mu, delta = result.x
                return alpha, beta, mu, delta
            else:
                return alpha_init, beta_init, mu_init, delta_init
        except:
            return alpha_init, beta_init, mu_init, delta_init

    def cdf(self, x):
        """Cumulative distribution function"""
        if np.isscalar(x):
            lower_bound = x - 50 * self.delta
            result, _ = quad(self.pdf, lower_bound, x)
            return result
        else:
            return np.array([self.cdf(xi) for xi in x])

    def ppf(self, q):
        """Percent point function (inverse CDF)"""
        if np.isscalar(q):
            if q <= 0: return -np.inf
            if q >= 1: return np.inf

            x_min, x_max = self.mu - 50 * self.delta, self.mu + 50 * self.delta

            # Expand range if needed
            attempts = 0
            while attempts < 10:
                if self.cdf(x_min) > q:
                    x_min -= 50 * self.delta
                elif self.cdf(x_max) < q:
                    x_max += 50 * self.delta
                else:
                    break
                attempts += 1

            # Binary search
            for _ in range(50):
                x_mid = (x_min + x_max) / 2
                cdf_mid = self.cdf(x_mid)

                if abs(cdf_mid - q) < 1e-6:
                    return x_mid

                if cdf_mid < q:
                    x_min = x_mid
                else:
                    x_max = x_mid

            return (x_min + x_max) / 2
        else:
            return np.array([self.ppf(qi) for qi in q])


# 2. Fit distributions to pre-holding period data
print("\nFitting distributions to pre-holding period stock returns...")

# Use training data for fitting
fit_results = {}
best_models = {}
model_params = {}
stock_returns = {}

# List of symbols excluding Date
symbols = [col for col in training_returns.columns if col not in ['Date', 'rf']]
symbols = [s for s in symbols if not s.endswith('_excess')]

# Define a function to evaluate model fit using AIC
def calculate_aic(log_likelihood, k):
    """Calculate AIC. Lower is better."""
    return 2 * k - 2 * log_likelihood

# Function to fit all distributions to a stock's returns
def fit_distributions(returns):
    result = {}

    # Filter out any NaN values - Fix for numpy arrays
    if isinstance(returns, np.ndarray):
        clean_returns = returns[~np.isnan(returns)]
    else:
        # If it's a pandas Series or DataFrame
        clean_returns = returns.dropna()

    # 1. Normal distribution
    try:
        norm_params = stats.norm.fit(clean_returns)
        mu, sigma = norm_params
        log_likelihood = np.sum(stats.norm.logpdf(clean_returns, mu, sigma))
        aic = calculate_aic(log_likelihood, 2)  # 2 parameters: mu, sigma
        result['Normal'] = {
            'params': norm_params,
            'aic': aic,
            'dist': stats.norm(*norm_params)
        }
    except:
        result['Normal'] = {'aic': np.inf}

    # 2. Generalized T distribution
    try:
        t_params = stats.t.fit(clean_returns)
        log_likelihood = np.sum(stats.t.logpdf(clean_returns, *t_params))
        aic = calculate_aic(log_likelihood, 3)  # 3 parameters: df, loc, scale
        result['GeneralizedT'] = {
            'params': t_params,
            'aic': aic,
            'dist': stats.t(*t_params)
        }
    except Exception as e:
        print(f"Error fitting GeneralizedT: {e}")
        result['GeneralizedT'] = {'aic': np.inf}

    try:
        nig = NormalInverseGaussian(1, 0, 0, 1)  # Default initialization
        alpha, beta, mu, delta = nig.fit(clean_returns)
        nig_params = (alpha, beta, mu, delta)
        nig_fitted = NormalInverseGaussian(*nig_params)

        # Calculate log-likelihood and AIC
        pdf_values = nig_fitted.pdf(clean_returns)
        pdf_values = np.maximum(pdf_values, 1e-300)  # Avoid log(0)
        log_likelihood = np.sum(np.log(pdf_values))
        aic = calculate_aic(log_likelihood, 4)  # 4 parameters: alpha, beta, mu, delta

        result['NIG'] = {
            'params': nig_params,
            'aic': aic,
            'dist': nig_fitted
        }
    except Exception as e:
        print(f"Error fitting NIG: {e}")
        import traceback
        traceback.print_exc()  # Print the full traceback for debugging
        result['NIG'] = {'aic': np.inf}


# 3. Skew Normal
    try:
        skewnorm_params = stats.skewnorm.fit(clean_returns)
        log_likelihood = np.sum(stats.skewnorm.logpdf(clean_returns, *skewnorm_params))
        aic = calculate_aic(log_likelihood, 3)  # 3 parameters: a, loc, scale
        result['SkewNormal'] = {
            'params': skewnorm_params,
            'aic': aic,
            'dist': stats.skewnorm(*skewnorm_params)
        }
    except Exception as e:
        print(f"Error fitting SkewNormal: {e}")
        result['SkewNormal'] = {'aic': np.inf}

    # Find best model based on AIC
    best_model = min(result.items(), key=lambda x: x[1]['aic'])[0]

    return result, best_model

# Fit distributions for each stock
for symbol in symbols:
    # Extract training period returns
    returns = training_returns[symbol].values
    stock_returns[symbol] = returns

    # Fit all distributions and find the best one
    try:
        fit_results[symbol], best_models[symbol] = fit_distributions(returns)
        model_params[symbol] = fit_results[symbol][best_models[symbol]]['params']
        # print(f"{symbol}: Best fit model is {best_models[symbol]}")
    except Exception as e:
        print(f"Error fitting distributions for {symbol}: {e}")
        best_models[symbol] = "Normal"  # Default to normal if fitting fails
        model_params[symbol] = stats.norm.fit(returns)

# Report best fit models and parameters
print("\nBest Fit Distribution Models and Parameters:")
print("=" * 100)
print(f"{'Symbol':<8} {'Best Model':<15} {'Parameters'}")
print("-" * 100)

for symbol in symbols:
    params_str = str(model_params[symbol])
    # For readability, truncate very long parameter strings
    if len(params_str) > 50:
        params_str = params_str[:47] + "..."
    print(f"{symbol:<8} {best_models[symbol]:<15} {params_str}")

# 3. Calculate VaR and ES using Gaussian Copula with fitted marginals
print("\nCalculating VaR and ES using Gaussian Copula with fitted marginals...")

# Function to calculate portfolio VaR and ES
def calculate_var_es(portfolio_name, weights, confidence_level=0.95, n_simulations=10000, method="GaussianCopula"):
    symbols_in_portfolio = list(weights.keys())

    if method == "GaussianCopula":
        # Step 1: Transform original returns to uniform using fitted distributions
        uniform_data = {}
        for symbol in symbols_in_portfolio:
            returns = stock_returns[symbol]
            best_model = best_models[symbol]
            dist = fit_results[symbol][best_model]['dist']

            # Calculate empirical CDFs
            try:
                u = np.array([dist.cdf(x) for x in returns])
                # Handle boundary cases
                u = np.minimum(np.maximum(u, 0.0001), 0.9999)
                uniform_data[symbol] = u
            except Exception as e:
                print(f"Error transforming {symbol} to uniform: {e}")
                # Fallback to empirical CDF
                ecdf = ECDF(returns)
                u = ecdf(returns)
                uniform_data[symbol] = u

        # Step 2: Transform uniform to standard normal
        normal_data = {}
        for symbol in symbols_in_portfolio:
            try:
                normal_data[symbol] = stats.norm.ppf(uniform_data[symbol])
            except:
                # Handle any numerical issues
                u_clean = np.clip(uniform_data[symbol], 0.0001, 0.9999)
                normal_data[symbol] = stats.norm.ppf(u_clean)

        # Step 3: Estimate correlation matrix of transformed data
        transformed_returns = pd.DataFrame({symbol: normal_data[symbol] for symbol in symbols_in_portfolio})
        correlation_matrix = transformed_returns.corr().values

        # Step 4: Generate correlated normal samples
        np.random.seed(42)  # For reproducibility
        simulated_normals = np.random.multivariate_normal(
            mean=np.zeros(len(symbols_in_portfolio)),
            cov=correlation_matrix,
            size=n_simulations
        )

        # Step 5: Transform back to original distribution
        simulated_returns = np.zeros((n_simulations, len(symbols_in_portfolio)))

        for i, symbol in enumerate(symbols_in_portfolio):
            z = simulated_normals[:, i]
            u = stats.norm.cdf(z)

            # Get correct distribution
            best_model = best_models[symbol]
            dist = fit_results[symbol][best_model]['dist']

            # Transform uniform back to returns using inverse CDF (ppf)
            try:
                simulated_returns[:, i] = dist.ppf(u)
            except Exception as e:
                print(f"Error in inverse transform for {symbol}: {e}")
                # Fallback to empirical inverse CDF
                x_sorted = np.sort(stock_returns[symbol])
                indices = np.floor(u * len(x_sorted)).astype(int)
                indices = np.minimum(indices, len(x_sorted)-1)
                simulated_returns[:, i] = x_sorted[indices]

        # Step 6: Calculate portfolio returns
        portfolio_returns = np.zeros(n_simulations)
        for i, symbol in enumerate(symbols_in_portfolio):
            portfolio_returns += simulated_returns[:, i] * weights[symbol]

    elif method == "MultivariateNormal":
        # Simpler approach: assume multivariate normal directly
        returns_data = np.column_stack([stock_returns[symbol] for symbol in symbols_in_portfolio])

        # Estimate mean and covariance
        mean_vector = np.zeros(len(symbols_in_portfolio))  # Assume 0% return as specified
        cov_matrix = np.cov(returns_data, rowvar=False)

        # Generate multivariate normal samples
        np.random.seed(42)  # For reproducibility
        simulated_returns = np.random.multivariate_normal(
            mean=mean_vector,
            cov=cov_matrix,
            size=n_simulations
        )

        # Calculate portfolio returns
        portfolio_returns = np.zeros(n_simulations)
        for i, symbol in enumerate(symbols_in_portfolio):
            portfolio_returns += simulated_returns[:, i] * weights[symbol]

    # Calculate VaR and ES
    sorted_returns = np.sort(portfolio_returns)
    var_index = int(n_simulations * (1 - confidence_level))
    var = -sorted_returns[var_index]
    es = -np.mean(sorted_returns[:var_index])

    return var, es

# Extract portfolio weights from original structure
# Note: We're using a different variable name to avoid conflict
portfolio_weight_data = {}
for portfolio in portfolios:
    weights = {}
    for symbol, info in portfolio_weights[portfolio].items():
        weights[symbol] = info['weight']
    portfolio_weight_data[portfolio] = weights

# Calculate VaR and ES for each portfolio using both methods
var_es_results = {}
confidence_level = 0.95  # 95% confidence level

for portfolio in portfolios:
    weights = portfolio_weight_data[portfolio]

    # Calculate using Gaussian Copula with fitted marginals
    var_gc, es_gc = calculate_var_es(
        portfolio, weights,
        confidence_level=confidence_level,
        method="GaussianCopula"
    )

    # Calculate using Multivariate Normal
    var_mvn, es_mvn = calculate_var_es(
        portfolio, weights,
        confidence_level=confidence_level,
        method="MultivariateNormal"
    )

    var_es_results[portfolio] = {
        'GaussianCopula': {'VaR': var_gc, 'ES': es_gc},
        'MultivariateNormal': {'VaR': var_mvn, 'ES': es_mvn}
    }

# Calculate for combined portfolio
combined_weights = {}
for portfolio in portfolios:
    portfolio_value = portfolio_values[portfolio]
    for symbol, info in portfolio_weights[portfolio].items():
        weight = info['weight'] * portfolio_value / sum(portfolio_values.values())
        if symbol in combined_weights:
            combined_weights[symbol] += weight
        else:
            combined_weights[symbol] = weight

# Calculate VaR and ES for combined portfolio
var_gc, es_gc = calculate_var_es(
    'Combined', combined_weights,
    confidence_level=confidence_level,
    method="GaussianCopula"
)

var_mvn, es_mvn = calculate_var_es(
    'Combined', combined_weights,
    confidence_level=confidence_level,
    method="MultivariateNormal"
)

var_es_results['Combined'] = {
    'GaussianCopula': {'VaR': var_gc, 'ES': es_gc},
    'MultivariateNormal': {'VaR': var_mvn, 'ES': es_mvn}
}

# Report VaR and ES results
print("\n1-Day VaR and ES Results at 95% Confidence Level:")
print("=" * 100)
print(f"{'Portfolio':<10} {'VaR (GC)':<15} {'ES (GC)':<15} {'VaR (MVN)':<15} {'ES (MVN)':<15} {'VaR Diff %':<15} {'ES Diff %':<15}")
print("-" * 100)

for portfolio in var_es_results:
    var_gc = var_es_results[portfolio]['GaussianCopula']['VaR']
    es_gc = var_es_results[portfolio]['GaussianCopula']['ES']
    var_mvn = var_es_results[portfolio]['MultivariateNormal']['VaR']
    es_mvn = var_es_results[portfolio]['MultivariateNormal']['ES']

    # Calculate percentage differences
    var_diff_pct = (var_gc - var_mvn) / var_mvn * 100 if var_mvn != 0 else float('inf')
    es_diff_pct = (es_gc - es_mvn) / es_mvn * 100 if es_mvn != 0 else float('inf')

    print(f"{portfolio:<10} {var_gc:<15.4f}  {es_gc:<15.4f}  {var_mvn:<15.4f} {es_mvn:<15.4f}  {var_diff_pct:<15.2f} {es_diff_pct:<15.2f}")

print("\nNote: GC = Gaussian Copula with fitted marginals, MVN = Multivariate Normal")
print("      Positive difference percentages indicate that the Gaussian Copula method gives higher risk estimates")

# Additional analysis of the differences
print("\nComparison of Distribution Models vs. Normal Distribution:")
print("=" * 100)
print("The differences between the two approaches can be attributed to:")
print("1. Skewness and fat tails captured by specialized distributions")
print("2. Non-linear dependency structures captured by the copula approach")
print("3. Different assumptions about the joint distribution of returns")

# Analyze which stocks deviate most from normality
print("\nStocks Deviating Most from Normality:")
print("=" * 80)
print(f"{'Symbol':<8} {'Best Model':<15} {'Skewness':<12} {'Excess Kurtosis':<15}")
print("-" * 80)

for symbol in symbols:
    returns = stock_returns[symbol]
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)  # Excess kurtosis (normal = 0)

    print(f"{symbol:<8} {best_models[symbol]:<15} {skewness:<12.4f} {kurtosis:<15.4f}")



from collections import Counter

distribution_counts = Counter(best_models.values())

print("\n分布类型对应的股票数量统计：")
print("=" * 40)
for dist, count in distribution_counts.items():
    print(f"{dist:<15} : {count} stocks")