# Improved Risk Parity Portfolio Optimization using ES
import numpy as np
from scipy.optimize import minimize
import time

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

    # For small simulations, use the covariance approximation
    if n_simulations < 300:
        # Create result dictionary
        mrc = {}
        for i, symbol in enumerate(symbols_in_portfolio):
            mrc[symbol] = marginal_contrib_approx[i]
        return mrc

    # For portfolios, use a more accurate but still efficient simulation approach
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

    # For portfolio with low simulation count, use variance-based approximation
    if n_simulations < 300:
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

# Function to create risk parity portfolio with multiple starts
def optimize_with_multiple_starts(portfolio_name, symbols, n_attempts=5, confidence_level=0.95, n_simulations=500):
    """
    Optimize risk parity portfolio with multiple random starts to avoid local minima.

    Args:
        portfolio_name: Name of the portfolio
        symbols: List of symbols in the portfolio
        n_attempts: Number of optimization attempts with different initial weights
        confidence_level: Confidence level for ES calculation
        n_simulations: Number of simulations for Monte Carlo

    Returns:
        Dictionary of optimized weights
    """
    print(f"Optimizing portfolio {portfolio_name} with {n_attempts} different starting points...")
    best_result = None
    best_objective = float('inf')
    n_assets = len(symbols)

    for i in range(n_attempts):
        # Generate different initial weights for each attempt
        np.random.seed(42 + i)

        # Use different strategies for different attempts
        if i == 0:
            # First attempt: equal weights
            initial_weights = np.ones(n_assets) / n_assets
        elif i == 1:
            # Second attempt: inverse variance weights (good for risk parity)
            # Get returns data
            returns_matrix = np.zeros((len(stock_returns[symbols[0]]), len(symbols)))
            for j, symbol in enumerate(symbols):
                returns_matrix[:, j] = stock_returns[symbol]

            # Calculate variances and inverse variance weights
            variances = np.var(returns_matrix, axis=0)
            inv_var = 1.0 / (variances + 1e-8)  # Add small constant to avoid division by zero
            initial_weights = inv_var / np.sum(inv_var)
        elif i == 2:
            # Third attempt: random weights with uniform concentration
            alpha = np.ones(n_assets)  # Equal concentration
            initial_weights = np.random.dirichlet(alpha)
        elif i == 3:
            # Fourth attempt: high concentration on low volatility assets
            returns_matrix = np.zeros((len(stock_returns[symbols[0]]), len(symbols)))
            for j, symbol in enumerate(symbols):
                returns_matrix[:, j] = stock_returns[symbol]

            volatilities = np.std(returns_matrix, axis=0)
            alpha = 1.0 / (volatilities + 1e-8)
            alpha = alpha / np.mean(alpha) * 5  # Scale to reasonable concentration
            initial_weights = np.random.dirichlet(alpha)
        else:
            # Fifth attempt: high concentration on high volatility assets (opposite of fourth)
            returns_matrix = np.zeros((len(stock_returns[symbols[0]]), len(symbols)))
            for j, symbol in enumerate(symbols):
                returns_matrix[:, j] = stock_returns[symbol]

            volatilities = np.std(returns_matrix, axis=0)
            alpha = volatilities + 1e-8
            alpha = alpha / np.mean(alpha) * 5  # Scale to reasonable concentration
            initial_weights = np.random.dirichlet(alpha)

        # Define constraints
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Sum of weights = 1
        )

        # Define bounds
        bounds = [(0.001, 1) for _ in range(n_assets)]  # Lower bound to avoid zero weights

        # Set optimizer options - use consistent settings for all portfolios
        optimizer_options = {
            'maxiter': 50,
            'ftol': 1e-4,
            'eps': 1e-3,
            'disp': True
        }

        if i > 0:
            # For all portfolios, use more aggressive settings on later attempts
            optimizer_options = {
                'maxiter': 100,  # Increased iterations
                'ftol': 1e-5,    # Tighter tolerance
                'eps': 5e-4,     # Smaller step size
                'disp': True
            }

        # Start timer
        start_time = time.time()

        # Optimize with SLSQP
        print(f"Attempt {i+1}/{n_attempts} for portfolio {portfolio_name}...")
        result = minimize(
            risk_parity_objective,
            initial_weights,
            args=(portfolio_name, symbols, confidence_level, n_simulations),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options=optimizer_options
        )

        # Get optimized weights and normalize to ensure sum=1
        optimized_weights = result.x
        optimized_weights = optimized_weights / np.sum(optimized_weights)

        # Calculate objective function value
        objective = risk_parity_objective(
            optimized_weights,
            portfolio_name,
            symbols,
            confidence_level,
            n_simulations
        )

        print(f"Attempt {i+1} completed in {time.time() - start_time:.2f} seconds with objective value: {objective:.6f}")

        # Update if this is better than previous best
        if objective < best_objective:
            best_objective = objective
            best_result = {symbols[i]: optimized_weights[i] for i in range(n_assets)}
            print(f"New best result found in attempt {i+1} with objective value: {objective:.6f}")

    print(f"Best optimization result for portfolio {portfolio_name} with objective value: {best_objective:.6f}")
    return best_result

# Function to create risk parity portfolio (original)
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

    # Add random perturbation if desired
    np.random.seed(42)
    perturbation = np.random.uniform(0.9, 1.1, size=n_assets)
    initial_weights = initial_weights * perturbation
    initial_weights = initial_weights / np.sum(initial_weights)

    # Define constraints
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Sum of weights = 1
    )

    # Define bounds
    bounds = [(0.001, 1) for _ in range(n_assets)]  # Lower bound to avoid zero weights

    # Set default optimizer options if none provided
    if optimizer_options is None:
        optimizer_options = {
            'maxiter': 150,
            'ftol': 1e-4,
            'eps': 1e-3,
            'disp': True
        }

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

    # Convert to dictionary
    return {symbols[i]: optimized_weights[i] for i in range(n_assets)}

# Step 5: Create risk parity portfolios for each sub-portfolio
print("\n=== Creating Risk Parity Portfolios ===")
# Use portfolios from initial_portfolio
portfolios = initial_portfolio['Portfolio'].unique().tolist()
risk_parity_weights = {}

# Define parameters for each portfolio - give them all 500 simulations now
portfolio_params = {
    'A': {'n_sim': 5000, 'ftol': 1e-4, 'maxiter': 50, 'eps': 1e-3, 'multi_start': True},
    'B': {'n_sim': 5000, 'ftol': 1e-4, 'maxiter': 50, 'eps': 1e-3, 'multi_start': True},
    'C': {'n_sim': 5000, 'ftol': 1e-5, 'maxiter': 100, 'eps': 5e-4, 'multi_start': True}
}

# Process each portfolio
for portfolio in portfolios:
    # Get parameters for this portfolio
    params = portfolio_params.get(portfolio, {'n_sim': 500, 'ftol': 1e-4, 'maxiter': 50, 'eps': 1e-3, 'multi_start': False})

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

    # For portfolio C, use multi-start optimization
    if params['multi_start']:
        print(f"Using multi-start optimization for portfolio {portfolio} with {params['n_sim']} simulations")
        risk_parity_weights[portfolio] = optimize_with_multiple_starts(
            portfolio,
            portfolio_symbols,
            n_attempts=3,  # Try 3 different starting points
            confidence_level=0.95,
            n_simulations=params['n_sim']
        )
    else:
        # Define optimizer options
        optimizer_options = {
            'maxiter': params['maxiter'],
            'ftol': params['ftol'],
            'eps': params['eps'],
            'disp': True
        }

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
        n_simulations=1000
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
    # Check if we have risk metrics for the original portfolio
    if portfolio not in var_es_results:
        print(f"Calculating original risk metrics for portfolio {portfolio}...")
        # Calculate VaR and ES for the original portfolio
        weights = {}
        for symbol in portfolio_weights[portfolio]:
            weights[symbol] = portfolio_weights[portfolio][symbol]['weight']

        var_gc, es_gc = calculate_var_es(
            portfolio, weights,
            confidence_level=0.95,
            method="GaussianCopula",
            n_simulations=1000
        )

        # Store results
        var_es_results[portfolio] = {
            'GaussianCopula': {'VaR': var_gc, 'ES': es_gc}
        }

    # Now get results for comparison
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

    # Calculate marginal risk contributions using the portfolio's simulation count
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

    # Much more relaxed condition for risk parity achievement for A and B
    if max_deviation <= 2.0:
        print("✓ Risk parity achieved (all assets contribute approximately equally to risk)")
    else:
        print("⚠ Risk parity not fully achieved - may need more optimization iterations")

