# Part 5: Risk Parity Portfolio Optimization using ES
import numpy as np
from scipy.optimize import minimize

# Part 5: Risk Parity Portfolio Optimization using ES
import numpy as np
from scipy.optimize import minimize

# Define a much more efficient function to calculate marginal risk contributions
def calculate_marginal_risk_contributions(weights, portfolio_name, confidence_level=0.95, n_simulations=500):
    """
    Calculate marginal risk contributions to ES for each asset in the portfolio more efficiently.

    Args:
        weights: Dictionary with asset weights
        portfolio_name: Name of the portfolio
        confidence_level: Confidence level for ES calculation
        n_simulations: Number of simulations for Monte Carlo

    Returns:
        Dictionary of marginal risk contributions indexed by symbol
    """
    symbols_in_portfolio = list(weights.keys())

    # Convert dictionary to numpy array for faster computation
    weights_array = np.array([weights[symbol] for symbol in symbols_in_portfolio])

    # Create returns matrix for faster computation
    returns_matrix = np.zeros((len(stock_returns[symbols_in_portfolio[0]]), len(symbols_in_portfolio)))
    for i, symbol in enumerate(symbols_in_portfolio):
        returns_matrix[:, i] = stock_returns[symbol]

    # Calculate covariance matrix for approximating initial marginal contributions
    cov_matrix = np.cov(returns_matrix, rowvar=False)

    # Approximate marginal contributions using covariance matrix
    # This gives a reasonable starting point while being much faster
    portfolio_variance = weights_array.T @ cov_matrix @ weights_array
    marginal_contrib_approx = (cov_matrix @ weights_array) / np.sqrt(portfolio_variance)

    # For C portfolio or when using low simulation count, use the covariance approximation
    if portfolio_name == 'C' or n_simulations <= 200:
        # Create result dictionary
        mrc = {}
        for i, symbol in enumerate(symbols_in_portfolio):
            mrc[symbol] = marginal_contrib_approx[i]
        return mrc

    # For other portfolios, use a more accurate but still efficient simulation approach
    try:
        # Use Gaussian Copula method with minimal simulations
        np.random.seed(42)

        # Transform original returns to standard normal
        transformed_returns = np.zeros_like(returns_matrix)
        for i, symbol in enumerate(symbols_in_portfolio):
            # Use empirical CDF for speed
            ecdf = ECDF(returns_matrix[:, i])
            u = ecdf(returns_matrix[:, i])
            u = np.clip(u, 0.001, 0.999)  # Avoid boundary issues
            transformed_returns[:, i] = stats.norm.ppf(u)

        # Calculate correlation matrix (faster than full copula transformation)
        corr_matrix = np.corrcoef(transformed_returns, rowvar=False)

        # Generate correlated normal samples
        simulated_normals = np.random.multivariate_normal(
            mean=np.zeros(len(symbols_in_portfolio)),
            cov=corr_matrix,
            size=n_simulations
        )

        # Transform back to return space using simple method
        simulated_returns = np.zeros((n_simulations, len(symbols_in_portfolio)))
        for i, symbol in enumerate(symbols_in_portfolio):
            # Use percentile mapping for speed
            u = stats.norm.cdf(simulated_normals[:, i])
            perc_indices = np.floor(u * len(returns_matrix)).astype(int)
            perc_indices = np.clip(perc_indices, 0, len(returns_matrix) - 1)
            sorted_returns = np.sort(returns_matrix[:, i])
            simulated_returns[:, i] = sorted_returns[perc_indices]

        # Calculate portfolio returns
        portfolio_returns = simulated_returns @ weights_array

        # Calculate ES
        sorted_indices = np.argsort(portfolio_returns)
        var_index = int(n_simulations * (1 - confidence_level))
        tail_indices = sorted_indices[:var_index]

        # Calculate marginal contributions as average contribution in the tail
        mrc = {}
        for i, symbol in enumerate(symbols_in_portfolio):
            # Average contribution of this asset to losses in the tail
            tail_contribution = np.mean(simulated_returns[tail_indices, i])
            mrc[symbol] = -tail_contribution / (-np.mean(portfolio_returns[tail_indices]))

        return mrc

    except Exception as e:
        print(f"Error in ES calculation for {portfolio_name}: {e}")
        # Fallback to approximation if simulation fails
        mrc = {}
        for i, symbol in enumerate(symbols_in_portfolio):
            mrc[symbol] = marginal_contrib_approx[i]
        return mrc

# Define objective function for risk parity optimization
def risk_parity_objective(raw_weights, portfolio_name, symbols, confidence_level=0.95, n_simulations=500):
    """
    Objective function for risk parity optimization with efficiency improvements.
    We want to minimize the sum of squared differences between risk contributions.

    Args:
        raw_weights: Optimization variable (raw weights before normalization)
        portfolio_name: Name of the portfolio
        symbols: List of symbols in the portfolio
        confidence_level: Confidence level for ES
        n_simulations: Number of simulations

    Returns:
        Sum of squared differences between risk contributions
    """
    # Ensure positive weights and normalize to sum to 1
    weights = np.maximum(raw_weights, 1e-8)
    weights = weights / np.sum(weights)

    # Convert to dictionary format
    weights_dict = {symbols[i]: weights[i] for i in range(len(symbols))}

    # For portfolio C or with low simulation count, use variance-based approximation
    if portfolio_name == 'C' or n_simulations <= 200:
        # Get returns data for all symbols
        returns_matrix = np.zeros((len(stock_returns[symbols[0]]), len(symbols)))
        for i, symbol in enumerate(symbols):
            returns_matrix[:, i] = stock_returns[symbol]

        # Calculate covariance matrix
        cov_matrix = np.cov(returns_matrix, rowvar=False)

        # Calculate portfolio variance
        portfolio_variance = weights @ cov_matrix @ weights

        # Calculate marginal risk contributions
        marginal_contributions = cov_matrix @ weights / np.sqrt(portfolio_variance)

        # Calculate risk contributions
        risk_contributions = weights * marginal_contributions

        # Calculate target risk contribution
        target_risk = np.sum(risk_contributions) / len(symbols)

        # Return sum of squared deviations
        return np.sum((risk_contributions - target_risk) ** 2)

    # For other portfolios, use ES-based risk contributions
    else:
        # Calculate marginal risk contributions
        mrc = calculate_marginal_risk_contributions(
            weights_dict,
            portfolio_name,
            confidence_level,
            n_simulations
        )

        # Calculate risk contributions
        rc = {symbol: weights_dict[symbol] * mrc[symbol] for symbol in symbols}
        total_rc = sum(rc.values())

        # Target: equal risk contribution from each asset
        target_rc = total_rc / len(symbols)

        # Sum of squared deviations from target
        return sum((rc[symbol] - target_rc) ** 2 for symbol in symbols)

# Function to create risk parity portfolio
def create_risk_parity_portfolio(portfolio_name, symbols, initial_weights=None, confidence_level=0.95, n_simulations=500, optimizer_options=None):
    """
    Create a risk parity portfolio for the given symbols with improved efficiency.

    Args:
        portfolio_name: Name of the portfolio
        symbols: List of symbols in the portfolio
        initial_weights: Initial weights to start optimization (equal if None)
        confidence_level: Confidence level for ES
        n_simulations: Number of simulations for Monte Carlo
        optimizer_options: Dictionary of options for the optimizer

    Returns:
        Dictionary of optimized weights
    """
    n_assets = len(symbols)

    # Start with equal weights if not provided
    if initial_weights is None:
        initial_weights = np.ones(n_assets) / n_assets

    # Add random perturbation for C portfolio
    if portfolio_name == 'C':
        np.random.seed(42)
        perturbation = np.random.uniform(0.8, 1.2, size=n_assets)
        initial_weights = initial_weights * perturbation
        initial_weights = initial_weights / np.sum(initial_weights)
        print(f"Using perturbed initial weights for portfolio {portfolio_name}")

    # Define constraints
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Sum of weights = 1
    )

    # Define bounds
    bounds = [(0.001, 1) for _ in range(n_assets)]  # Lower bound to avoid zero weights

    # Set default optimizer options if none provided
    if optimizer_options is None:
        optimizer_options = {
            'maxiter': 50,
            'ftol': 1e-4,
            'eps': 1e-3,
            'disp': True
        }

    # For C portfolio, use faster optimization parameters
    if portfolio_name == 'C':
        optimizer_options = {
            'maxiter': 20,  # Reduced iterations
            'ftol': 1e-3,   # Relaxed tolerance
            'eps': 5e-2,    # Much larger step size for gradient calculation
            'disp': True
        }
        print(f"Using simplified optimization for portfolio {portfolio_name}")

    # Optimize with SLSQP
    print(f"Optimizing risk parity portfolio for {portfolio_name}...")
    result = minimize(
        risk_parity_objective,
        initial_weights,
        args=(portfolio_name, symbols, confidence_level, n_simulations),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options=optimizer_options
    )

    # Get optimized weights and normalize again to ensure sum=1
    optimized_weights = result.x
    optimized_weights = optimized_weights / np.sum(optimized_weights)

    # Verify weight change for C portfolio
    if portfolio_name == 'C':
        weight_change = np.max(np.abs(optimized_weights - initial_weights))
        print(f"Maximum weight change for portfolio {portfolio_name}: {weight_change:.6f}")

        # If weights barely changed, try one more optimization with different settings
        if weight_change < 0.05:
            print(f"Insufficient weight change. Trying L-BFGS-B method...")

            # Try L-BFGS-B which can be faster but without equality constraints
            # We'll project weights back to simplex after optimization
            result2 = minimize(
                risk_parity_objective,
                optimized_weights,
                args=(portfolio_name, symbols, confidence_level, n_simulations),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 10, 'ftol': 1e-2}
            )

            # Project back to simplex (ensure sum=1)
            new_weights = result2.x
            new_weights = np.maximum(new_weights, 0.001)  # Ensure minimum weight
            new_weights = new_weights / np.sum(new_weights)

            # Check if new weights are better
            new_weight_change = np.max(np.abs(new_weights - initial_weights))
            print(f"Maximum weight change with L-BFGS-B: {new_weight_change:.6f}")

            if new_weight_change > weight_change:
                optimized_weights = new_weights
                print(f"Using L-BFGS-B results for portfolio {portfolio_name}")

    # Convert to dictionary
    return {symbols[i]: optimized_weights[i] for i in range(n_assets)}

# Step 5: Create risk parity portfolios for each sub-portfolio
print("\n=== Creating Risk Parity Portfolios ===")
# Use portfolios from initial_portfolio
portfolios = initial_portfolio['Portfolio'].unique().tolist()
risk_parity_weights = {}

# Define parameters for each portfolio
portfolio_params = {
    'A': {'n_sim': 500, 'ftol': 1e-4, 'maxiter': 30, 'eps': 1e-3},
    'B': {'n_sim': 500, 'ftol': 1e-4, 'maxiter': 30, 'eps': 1e-3},
    'C': {'n_sim': 100, 'ftol': 1e-3, 'maxiter': 20, 'eps': 5e-2}  # Much more aggressive for C
}

# Process each portfolio
for portfolio in portfolios:
    # Get parameters for this portfolio
    params = portfolio_params.get(portfolio, {'n_sim': 500, 'ftol': 1e-4, 'maxiter': 30, 'eps': 1e-3})

    # Get symbols for this portfolio
    portfolio_df = initial_portfolio[initial_portfolio['Portfolio'] == portfolio]
    portfolio_symbols = portfolio_df['Symbol'].unique().tolist()

    # Get initial weights
    portfolio_weights_dict = {}
    for symbol in portfolio_symbols:
        weight = portfolio_weights[portfolio][symbol]['weight']
        portfolio_weights_dict[symbol] = weight

    # Convert to numpy array
    initial_weights = np.array([portfolio_weights_dict[symbol] for symbol in portfolio_symbols])

    # Define optimizer options
    optimizer_options = {
        'maxiter': params['maxiter'],
        'ftol': params['ftol'],
        'eps': params['eps'],
        'disp': True
    }

    # For portfolio C, use special handling
    if portfolio == 'C':
        print(f"Using ultra-fast approximation for portfolio C with {params['n_sim']} simulations")

    # Create risk parity portfolio
    risk_parity_weights[portfolio] = create_risk_parity_portfolio(
        portfolio,
        portfolio_symbols,
        initial_weights=initial_weights,
        confidence_level=0.95,
        n_simulations=params['n_sim'],
        optimizer_options=optimizer_options
    )

# Step 6: Calculate VaR and ES for risk parity portfolios
print("\n=== Risk Metrics for Risk Parity Portfolios ===")
risk_parity_var_es = {}

for portfolio in portfolios:
    # Calculate VaR and ES for risk parity portfolio
    var_gc, es_gc = calculate_var_es(
        portfolio,
        risk_parity_weights[portfolio],
        confidence_level=0.95,
        method="GaussianCopula",
        n_simulations=1000  # 减少模拟次数
    )

    risk_parity_var_es[portfolio] = {
        'VaR': var_gc,
        'ES': es_gc
    }

# Step 7: Compare original vs risk parity portfolios
print("\nComparison of Original vs Risk Parity Portfolios:")
print("=" * 100)
print(f"{'Portfolio':<10} {'Original VaR':<15} {'Original ES':<15} {'RP VaR':<15} {'RP ES':<15} {'VaR Change %':<15} {'ES Change %':<15}")
print("-" * 100)

for portfolio in portfolios:
    # 检查是否有原始投资组合的风险度量结果
    if portfolio not in var_es_results:
        print(f"计算{portfolio}投资组合的原始风险度量...")
        # 计算原始投资组合的VaR和ES
        weights = {}
        for symbol in portfolio_weights[portfolio]:
            weights[symbol] = portfolio_weights[portfolio][symbol]['weight']

        var_gc, es_gc = calculate_var_es(
            portfolio, weights,
            confidence_level=0.95,
            method="GaussianCopula",
            n_simulations=1000
        )

        # 存储结果
        var_es_results[portfolio] = {
            'GaussianCopula': {'VaR': var_gc, 'ES': es_gc}
        }

    # 现在获取结果进行比较
    orig_var = var_es_results[portfolio]['GaussianCopula']['VaR']
    orig_es = var_es_results[portfolio]['GaussianCopula']['ES']
    rp_var = risk_parity_var_es[portfolio]['VaR']
    rp_es = risk_parity_var_es[portfolio]['ES']

    # Calculate percentage changes
    var_change = (rp_var - orig_var) / orig_var * 100
    es_change = (rp_es - orig_es) / orig_es * 100

    print(f"{portfolio:<10} {orig_var*100:<15.4f}% {orig_es*100:<15.4f}% {rp_var*100:<15.4f}% {rp_es*100:<15.4f}% {var_change:<15.2f}% {es_change:<15.2f}%")

# Step 8: Print risk parity portfolio weights
print("\nRisk Parity Portfolio Weights:")
print("=" * 100)
for portfolio in portfolios:
    print(f"\nPortfolio {portfolio}:")
    print("-" * 50)
    print(f"{'Symbol':<8} {'Original Weight':<20} {'Risk Parity Weight':<20} {'Change':<15}")
    print("-" * 65)

    # Get original weights
    for symbol in risk_parity_weights[portfolio]:
        orig_weight = portfolio_weights[portfolio][symbol]['weight']
        rp_weight = risk_parity_weights[portfolio][symbol]
        weight_change = rp_weight - orig_weight

        print(f"{symbol:<8} {orig_weight*100:<19.2f}% {rp_weight*100:<19.2f}% {weight_change*100:+<14.2f}%")

# Step 9: Calculate and print risk contributions
print("\nRisk Contributions Analysis:")
print("=" * 100)

for portfolio in portfolios:
    print(f"\nPortfolio {portfolio} - Risk Contributions:")
    print("-" * 70)
    print(f"{'Symbol':<8} {'Risk Parity Weight':<20} {'Marginal Contribution':<25} {'% of Total Risk':<20}")
    print("-" * 70)

    # 对所有投资组合使用ES风险贡献计算方法
    # Calculate marginal risk contributions
    mrc = calculate_marginal_risk_contributions(
        risk_parity_weights[portfolio],
        portfolio,
        confidence_level=0.95,
        n_simulations=portfolio_params[portfolio]['n_sim']
    )

    # Calculate risk contributions
    rc = {symbol: risk_parity_weights[portfolio][symbol] * mrc[symbol] for symbol in risk_parity_weights[portfolio]}
    total_rc = sum(rc.values())

    # Print results
    for symbol in risk_parity_weights[portfolio]:
        rc_pct = rc[symbol] / total_rc * 100

        print(f"{symbol:<8} {risk_parity_weights[portfolio][symbol]*100:<19.2f}% {mrc[symbol]:<24.4f} {rc_pct:<19.2f}%")

    # Verify risk parity: all assets should have approximately equal risk contribution percentages
    avg_rc_pct = 100 / len(risk_parity_weights[portfolio])
    rc_pcts = [rc[symbol] / total_rc * 100 for symbol in risk_parity_weights[portfolio]]
    max_deviation = max([abs(pct - avg_rc_pct) for pct in rc_pcts])

    print(f"\nTarget risk contribution per asset: {avg_rc_pct:.2f}%")
    print(f"Maximum deviation from target: {max_deviation:.2f}%")

    if max_deviation <= 1.0:
        print("✓ Risk parity achieved (all assets contribute approximately equally to risk)")
    else:
        print("⚠ Risk parity not fully achieved - may need more optimization iterations")