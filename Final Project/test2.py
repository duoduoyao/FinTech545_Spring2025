# Step 12: Create optimal maximum Sharpe Ratio portfolios
import numpy as np
from scipy.optimize import minimize

# Use market return and risk-free rate from training period as expected values
expected_market_return = training_returns[market_symbol].mean()
expected_rf = training_returns['rf'].mean()
expected_market_excess = expected_market_return - expected_rf

print(f"Expected Market Return: {expected_market_return * 100:.6f}%")
print(f"Expected Risk-Free Rate: {expected_rf * 100:.6f}%")
print(f"Expected Market Excess Return: {expected_market_excess * 100:.6f}%")

# Store optimal portfolios
optimal_portfolios = {}
optimal_weights = {}

for portfolio in portfolios:
    portfolio_df = initial_portfolio[initial_portfolio['Portfolio'] == portfolio]
    portfolio_symbols = portfolio_df['Symbol'].unique()

    # Collect stock data for optimization
    stock_betas = {}
    stock_idio_vars = {}
    stock_expected_returns = {}
    symbols_to_optimize = []

    # print(f"\nOptimizing Portfolio {portfolio}:")
    # print("-" * 50)

    for symbol in portfolio_symbols:
        if symbol not in capm_params:
            print(f"Skipping {symbol}, no CAPM model")
            continue

        symbols_to_optimize.append(symbol)

        # Get stock beta
        beta = capm_params[symbol]['beta']
        stock_betas[symbol] = beta

        # Calculate idiosyncratic risk (residual variance)
        X = training_returns[f'{market_symbol}_excess'].values
        Y = training_returns[f'{symbol}_excess'].values

        # Remove NaN values
        mask = ~np.isnan(X) & ~np.isnan(Y)
        X = X[mask]
        Y = Y[mask]

        # Calculate residuals using fitted beta
        Y_pred = beta * X
        residuals = Y - Y_pred
        idio_var = np.var(residuals)
        stock_idio_vars[symbol] = idio_var

        # Calculate expected excess return (assume alpha = 0)
        expected_excess_return = beta * expected_market_excess
        stock_expected_returns[symbol] = expected_excess_return

        # print(f"{symbol}: Beta = {beta:.4f}, Idiosyncratic Variance = {idio_var * 100:.6f}%, Expected Excess Return = {expected_excess_return * 100:.6f}%")

    # Skip if no valid stocks to optimize
    if len(symbols_to_optimize) == 0:
        print(f"Portfolio {portfolio} has no valid stocks for optimization")
        continue

    # Create covariance matrix for optimization
    n_stocks = len(symbols_to_optimize)
    cov_matrix = np.zeros((n_stocks, n_stocks))
    expected_returns = np.zeros(n_stocks)

    # Fill covariance matrix
    for i, symbol_i in enumerate(symbols_to_optimize):
        expected_returns[i] = stock_expected_returns[symbol_i]
        beta_i = stock_betas[symbol_i]
        idio_var_i = stock_idio_vars[symbol_i]

        for j, symbol_j in enumerate(symbols_to_optimize):
            beta_j = stock_betas[symbol_j]
            idio_var_j = stock_idio_vars[symbol_j]

            if i == j:
                # Diagonal: total variance = beta^2 * market_var + idio_var
                cov_matrix[i, j] = beta_i**2 * market_variance + idio_var_i
            else:
                # Off-diagonal: covariance = beta_i * beta_j * market_var
                cov_matrix[i, j] = beta_i * beta_j * market_variance

    # Define objective function for Sharpe ratio maximization
    def negative_sharpe(weights):
        portfolio_return = np.sum(weights * expected_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0

    # Constraints: weights sum to 1, and weights are non-negative
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_stocks))

    # Initial guess: equal weight
    init_weights = np.ones(n_stocks) / n_stocks

    # Run optimization
    result = minimize(negative_sharpe, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    if result['success']:
        print("Optimization successful")
        optimal_weights[portfolio] = {}

        # Store optimal weights
        for i, symbol in enumerate(symbols_to_optimize):
            weight = result['x'][i]
            optimal_weights[portfolio][symbol] = weight
            # print(f"{symbol} optimal weight: {weight:.4f}")

        # Calculate expected portfolio metrics
        expected_port_return = np.sum(result['x'] * expected_returns)
        expected_port_risk = np.sqrt(np.dot(result['x'].T, np.dot(cov_matrix, result['x'])))
        expected_sharpe = expected_port_return / expected_port_risk if expected_port_risk > 0 else 0

        # print(f"Expected Portfolio Excess Return: {expected_port_return * 100:.6f}%")
        # print(f"Expected Portfolio Risk: {expected_port_risk * 100:.6f}%")
        # print(f"Expected Sharpe Ratio: {expected_sharpe:.6f}")
    else:
        print(f"Optimization failed for Portfolio {portfolio}: {result['message']}")
        continue

# Step 13: Test optimal portfolios during the holding period
# Calculate daily optimal portfolio returns
optimal_daily_returns = {}
optimal_daily_risk = {}

for portfolio in optimal_weights:
    optimal_daily_returns[portfolio] = pd.DataFrame(index=testing_returns.index)
    optimal_daily_returns[portfolio]['Date'] = testing_returns['Date']
    optimal_daily_returns[portfolio]['portfolio_return'] = 0

    # Calculate weighted daily returns
    for symbol, weight in optimal_weights[portfolio].items():
        optimal_daily_returns[portfolio]['portfolio_return'] += testing_returns[symbol] * weight

    # Calculate daily excess returns
    optimal_daily_returns[portfolio]['excess_return'] = optimal_daily_returns[portfolio]['portfolio_return'] - testing_returns['rf']

    # Calculate portfolio risk metrics
    mean_return = optimal_daily_returns[portfolio]['portfolio_return'].mean()
    std_dev = optimal_daily_returns[portfolio]['portfolio_return'].std()
    sharpe = optimal_daily_returns[portfolio]['excess_return'].mean() / optimal_daily_returns[portfolio]['excess_return'].std() if std_dev > 0 else 0

    optimal_daily_risk[portfolio] = {
        'mean_daily_return': mean_return,
        'daily_std_dev': std_dev,
        'daily_sharpe': sharpe
    }

# Step 14: Performance attribution for optimal portfolios
optimal_result_data = []

for portfolio in optimal_weights:
    # Portfolio-level metrics
    portfolio_beta = 0
    portfolio_daily_systematic = pd.Series(0, index=testing_returns.index)
    portfolio_daily_idiosyncratic = pd.Series(0, index=testing_returns.index)

    for symbol, weight in optimal_weights[portfolio].items():
        if symbol not in capm_params:
            print(f"No CAPM model for {symbol}, skipping")
            continue

        # CAPM model parameters
        beta = capm_params[symbol]['beta']

        # Portfolio beta
        portfolio_beta += beta * weight

        # Calculate daily systematic and idiosyncratic components
        daily_systematic = beta * testing_returns[f'{market_symbol}_excess']
        daily_idiosyncratic = testing_returns[f'{symbol}_excess'] - daily_systematic

        # Weighted daily components
        weighted_daily_systematic = daily_systematic * weight
        weighted_daily_idiosyncratic = daily_idiosyncratic * weight

        # Add to portfolio totals
        portfolio_daily_systematic += weighted_daily_systematic
        portfolio_daily_idiosyncratic += weighted_daily_idiosyncratic

    # Calculate portfolio-level average daily returns
    avg_portfolio_daily_return = optimal_daily_returns[portfolio]['portfolio_return'].mean()
    avg_portfolio_daily_excess = optimal_daily_returns[portfolio]['excess_return'].mean()
    avg_portfolio_daily_systematic = portfolio_daily_systematic.mean()
    avg_portfolio_daily_idiosyncratic = portfolio_daily_idiosyncratic.mean()
    portfolio_daily_std = optimal_daily_returns[portfolio]['portfolio_return'].std()

    # Calculate portfolio risk decomposition
    portfolio_variance = portfolio_daily_std**2
    portfolio_systematic_risk = portfolio_beta**2 * market_variance
    portfolio_idiosyncratic_risk = portfolio_variance - portfolio_systematic_risk

    # Calculate risk proportion
    total_risk = portfolio_variance
    systematic_risk_pct = (portfolio_systematic_risk / total_risk) * 100 if total_risk > 0 else 0
    idiosyncratic_risk_pct = (portfolio_idiosyncratic_risk / total_risk) * 100 if total_risk > 0 else 0

    # Store optimal portfolio summary
    optimal_result_data.append({
        'Portfolio': portfolio + " (Optimal)",
        'Beta': portfolio_beta,
        'Avg Daily Return': avg_portfolio_daily_return,
        'Daily Std Dev': portfolio_daily_std,
        'Avg Daily Excess': avg_portfolio_daily_excess,
        'Avg Daily Systematic': avg_portfolio_daily_systematic,
        'Avg Daily Idiosyncratic': avg_portfolio_daily_idiosyncratic,
        'Total Variance': portfolio_variance,
        'Systematic Risk': portfolio_systematic_risk,
        'Idiosyncratic Risk': portfolio_idiosyncratic_risk,
        'Systematic Risk %': systematic_risk_pct,
        'Idiosyncratic Risk %': idiosyncratic_risk_pct
    })

# Step 15: Compare original vs. optimal portfolios
print("\nComparison of Original vs. Optimal Portfolios")
print("=" * 120)
print(f"{'Portfolio':<15} {'Beta':<8} {'Daily Return':<15} {'Daily StdDev':<15} {'Daily Excess':<15} {'System Risk%':<15} {'Idio Risk%':<15} {'Sharpe':<10}")
print("-" * 120)

# Print original portfolio results
for d in risk_data:
    if d['Portfolio'] == 'Combined':
        continue
    sharpe = d['Avg Daily Excess'] / d['Daily Std Dev'] if d['Daily Std Dev'] > 0 else 0
    print(f"{d['Portfolio']:<15} {d['Beta']:<8.3f} {d['Avg Daily Return'] * 100:<15.4f}% {d['Daily Std Dev'] * 100:<15.4f}% {d['Avg Daily Excess'] * 100:<15.4f}% {d['Systematic Risk %']:<15.2f}% {d['Idiosyncratic Risk %']:<15.2f}% {sharpe:<10.4f}")

# Print optimal portfolio results
for d in optimal_result_data:
    sharpe = d['Avg Daily Excess'] / d['Daily Std Dev'] if d['Daily Std Dev'] > 0 else 0
    print(f"{d['Portfolio']:<15} {d['Beta']:<8.3f} {d['Avg Daily Return'] * 100:<15.4f}% {d['Daily Std Dev'] * 100:<15.4f}% {d['Avg Daily Excess'] * 100:<15.4f}% {d['Systematic Risk %']:<15.2f}% {d['Idiosyncratic Risk %']:<15.2f}% {sharpe:<10.4f}")

# print("\nModel Expected vs. Realized Idiosyncratic Risk")
# print("=" * 100)
# print(f"{'Portfolio':<15} {'Expected Idio Risk':<20} {'Realized Idio Risk':<20} {'Difference':<15}")
# print("-" * 100)
# # Calculate expected idiosyncratic risk for optimal portfolios based on model
# for portfolio in optimal_weights:
#     expected_idio_risk = 0
#
#     # Calculate weighted sum of idiosyncratic variances
#     for symbol, weight in optimal_weights[portfolio].items():
#         if symbol in stock_idio_vars:
#             expected_idio_risk += weight**2 * stock_idio_vars[symbol]
#
#     # Find realized idiosyncratic risk
#     realized_data = [d for d in optimal_result_data if d['Portfolio'] == portfolio + " (Optimal)"][0]
#     realized_idio_risk = realized_data['Idiosyncratic Risk']
#
#     # Calculate difference
#     diff = realized_idio_risk - expected_idio_risk
#     diff_pct = (diff / expected_idio_risk) * 100 if expected_idio_risk > 0 else float('inf')
#
#     print(f"{portfolio + ' (Optimal)':<15} {expected_idio_risk * 100:<20.6f}% {realized_idio_risk * 100:<20.6f}% {diff_pct:<15.2f}%")