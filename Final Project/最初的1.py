import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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

# Step 4: Merge risk-free rate with returns data
# Use first training return date to filter risk-free data
first_return_date = training_returns['Date'].min()
risk_free_filtered = risk_free[risk_free['Date'] >= first_return_date].copy()

# Merge risk-free rates with returns
training_returns = pd.merge(training_returns, risk_free_filtered, on='Date', how='left')
testing_returns = pd.merge(testing_returns, risk_free_filtered, on='Date', how='left')


# Calculate excess returns
for column in training_returns.columns:
    if column not in ['Date', 'rf']:
        training_returns[f'{column}_excess'] = training_returns[column] - training_returns['rf']
        testing_returns[f'{column}_excess'] = testing_returns[column] - testing_returns['rf']


# print(f"\nFirst few rows of training excess returns for AAPL:")
# print(training_returns[['Date', 'AAPL', 'rf', 'AAPL_excess']].head())

# Step 5: Fit CAPM models using training data (2023)
market_symbol = 'SPY'
symbols = [col for col in daily_prices.columns if col != 'Date']

# Store CAPM parameters
capm_params = {}

for symbol in symbols:
    if symbol == market_symbol:
        continue

    # Linear regression for each stock against the market
    X = training_returns[f'{market_symbol}_excess'].values
    Y = training_returns[f'{symbol}_excess'].values

    # Remove NaN values
    mask = ~np.isnan(X) & ~np.isnan(Y)
    X = X[mask]
    Y = Y[mask]

    # Fit linear regression: excess_return = alpha + beta * market_excess_return
    slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)

    # Store parameters
    capm_params[symbol] = {
        'alpha': intercept,
        'beta': slope,
        'r_squared': r_value ** 2,
        'p_value': p_value,
        'std_err': std_err
    }

# Step 6: Calculate portfolio composition and value
portfolio_symbols = initial_portfolio['Symbol'].unique()
portfolio_weights = {}

# Get unique portfolios
portfolios = initial_portfolio['Portfolio'].unique()

# Calculate initial investment for each portfolio
portfolio_values = {}
portfolio_weights = {}

last_training_date = training_data['Date'].max()

for portfolio in portfolios:
    portfolio_df = initial_portfolio[initial_portfolio['Portfolio'] == portfolio]
    total_value = 0
    weights = {}

    # First pass: calculate total portfolio value
    for _, row in portfolio_df.iterrows():
        symbol = row['Symbol']
        holdings = row['Holding']
        price = training_data.loc[training_data['Date'] == last_training_date, symbol].values[0]
        value = holdings * price
        total_value += value
        weights[symbol] = {'holdings': holdings, 'value': value}  # 临时保存 value，等下再转成权重

    # Second pass: calculate weights
    for symbol in weights:
        weights[symbol]['weight'] = weights[symbol]['value'] / total_value
        del weights[symbol]['value']

    portfolio_values[portfolio] = total_value
    portfolio_weights[portfolio] = weights


# Step 7: Analyze portfolio performance during the holding period
# Focus on daily returns rather than cumulative returns

# Calculate daily portfolio returns
portfolio_daily_returns = {}
portfolio_daily_risk = {}
# Calculate cumulative returns for the holding period (for reference)
total_testing_returns = {}
for symbol in symbols:
    if symbol == 'Date':
        continue

    # Calculate cumulative return over the testing period
    first_price = testing_data.iloc[0][symbol]
    last_price = testing_data.iloc[-1][symbol]
    total_return = (last_price / first_price) - 1

    total_testing_returns[symbol] = total_return

for portfolio in portfolios:
    portfolio_df = initial_portfolio[initial_portfolio['Portfolio'] == portfolio]
    portfolio_daily_returns[portfolio] = pd.DataFrame(index=testing_returns.index)
    portfolio_daily_returns[portfolio]['Date'] = testing_returns['Date']
    portfolio_daily_returns[portfolio]['portfolio_return'] = 0

    # Calculate weighted daily returns
    for _, row in portfolio_df.iterrows():
        symbol = row['Symbol']
        weight = portfolio_weights[portfolio][symbol]['weight']
        portfolio_daily_returns[portfolio]['portfolio_return'] += testing_returns[symbol] * weight

    # Calculate daily excess returns
    portfolio_daily_returns[portfolio]['excess_return'] = portfolio_daily_returns[portfolio]['portfolio_return'] - \
                                                          testing_returns['rf']

    # Calculate portfolio risk metrics
    mean_return = portfolio_daily_returns[portfolio]['portfolio_return'].mean()
    std_dev = portfolio_daily_returns[portfolio]['portfolio_return'].std()
    sharpe = portfolio_daily_returns[portfolio]['excess_return'].mean() / portfolio_daily_returns[portfolio][
        'excess_return'].std()

    portfolio_daily_risk[portfolio] = {
        'mean_daily_return': mean_return,
        'daily_std_dev': std_dev,
        'daily_sharpe': sharpe
    }

# Calculate market-related metrics for analysis
market_daily_excess = testing_returns[f'{market_symbol}_excess']
market_mean_excess = market_daily_excess.mean()
market_std_excess = market_daily_excess.std()

# Step 8: Performance attribution using daily returns
result_data = []

for portfolio in portfolios:
    portfolio_df = initial_portfolio[initial_portfolio['Portfolio'] == portfolio]
    portfolio_symbols = portfolio_df['Symbol'].unique()

    # Portfolio-level metrics
    portfolio_beta = 0
    portfolio_daily_systematic = pd.Series(0, index=testing_returns.index)
    portfolio_daily_idiosyncratic = pd.Series(0, index=testing_returns.index)

    for symbol in portfolio_symbols:
        if symbol not in capm_params:
            print(f"No CAPM model for {symbol}, skipping")
            continue

        weight = portfolio_weights[portfolio][symbol]['weight']

        # CAPM model parameters
        alpha = capm_params[symbol]['alpha']  # Daily alpha
        beta = capm_params[symbol]['beta']

        # Calculate daily systematic and idiosyncratic components
        daily_systematic = beta * testing_returns[f'{market_symbol}_excess']
        daily_idiosyncratic = testing_returns[f'{symbol}_excess'] - daily_systematic

        # Weighted daily components
        weighted_daily_systematic = daily_systematic * weight
        weighted_daily_idiosyncratic = daily_idiosyncratic * weight

        # Add to portfolio totals
        portfolio_beta += beta * weight
        portfolio_daily_systematic += weighted_daily_systematic
        portfolio_daily_idiosyncratic += weighted_daily_idiosyncratic

        # Calculate average daily returns for this stock
        avg_daily_return = testing_returns[symbol].mean()
        avg_daily_excess = testing_returns[f'{symbol}_excess'].mean()
        avg_daily_systematic = daily_systematic.mean()
        avg_daily_idiosyncratic = daily_idiosyncratic.mean()
        daily_std = testing_returns[symbol].std()

        # Store individual stock data
        result_data.append({
            'Portfolio': portfolio,
            'Symbol': symbol,
            'Weight': weight,
            'Beta': beta,
            'Avg Daily Return': avg_daily_return,
            'Daily Std Dev': daily_std,
            'Avg Daily Excess': avg_daily_excess,
            'Avg Daily Systematic': avg_daily_systematic,
            'Avg Daily Idiosyncratic': avg_daily_idiosyncratic
        })

    # Calculate portfolio-level average daily returns
    avg_portfolio_daily_return = portfolio_daily_returns[portfolio]['portfolio_return'].mean()
    avg_portfolio_daily_excess = portfolio_daily_returns[portfolio]['excess_return'].mean()
    avg_portfolio_daily_systematic = portfolio_daily_systematic.mean()  # Take the mean here
    avg_portfolio_daily_idiosyncratic = portfolio_daily_idiosyncratic.mean()  # Take the mean here
    portfolio_daily_std = portfolio_daily_returns[portfolio]['portfolio_return'].std()

    # Check that the decomposition adds up (excess = systematic + idiosyncratic)
    decomp_check = avg_portfolio_daily_excess - (avg_portfolio_daily_systematic + avg_portfolio_daily_idiosyncratic)
    if abs(decomp_check) > 1e-10:
        print(f"Warning: Daily return decomposition for Portfolio {portfolio} doesn't balance exactly")
        print(
            f"  Excess: {avg_portfolio_daily_excess:.8f}, Systematic: {avg_portfolio_daily_systematic:.8f}, Idiosyncratic: {avg_portfolio_daily_idiosyncratic:.8f}")
        print(f"  Difference: {decomp_check:.8f}")

    # Store portfolio summary
    portfolio_summary = {
        'Portfolio': portfolio,
        'Beta': portfolio_beta,
        'Avg Daily Return': avg_portfolio_daily_return,
        'Daily Std Dev': portfolio_daily_std,
        'Avg Daily Excess': avg_portfolio_daily_excess,
        'Avg Daily Systematic': avg_portfolio_daily_systematic,
        'Avg Daily Idiosyncratic': avg_portfolio_daily_idiosyncratic,
        'Daily Sharpe': avg_portfolio_daily_excess / portfolio_daily_std if portfolio_daily_std > 0 else 0,
        'Initial Value': portfolio_values[portfolio]
    }

    # Add portfolio summary to result data
    result_data.append(portfolio_summary)

# Calculate combined portfolio (weighted average of portfolios)
total_value = sum(portfolio_values.values())
portfolio_weights_combined = {p: portfolio_values[p] / total_value for p in portfolios}

# Filter portfolio summaries (no stocks)
portfolio_summaries = [d for d in result_data if 'Symbol' not in d]

# Create combined portfolio summary using scalar values
combined_portfolio = {
    'Portfolio': 'Combined',
    'Beta': sum(d['Beta'] * portfolio_weights_combined[d['Portfolio']] for d in portfolio_summaries),
    'Avg Daily Return': sum(
        d['Avg Daily Return'] * portfolio_weights_combined[d['Portfolio']] for d in portfolio_summaries),
    'Daily Std Dev': sum(d['Daily Std Dev'] * portfolio_weights_combined[d['Portfolio']] for d in portfolio_summaries),
    # Approximation
    'Avg Daily Excess': sum(
        d['Avg Daily Excess'] * portfolio_weights_combined[d['Portfolio']] for d in portfolio_summaries),
    'Avg Daily Systematic': sum(
        d['Avg Daily Systematic'] * portfolio_weights_combined[d['Portfolio']] for d in portfolio_summaries),
    'Avg Daily Idiosyncratic': sum(
        d['Avg Daily Idiosyncratic'] * portfolio_weights_combined[d['Portfolio']] for d in portfolio_summaries),
    'Initial Value': total_value
}

# Calculate Sharpe ratio for combined portfolio
if combined_portfolio['Daily Std Dev'] > 0:
    combined_portfolio['Daily Sharpe'] = combined_portfolio['Avg Daily Excess'] / combined_portfolio['Daily Std Dev']
else:
    combined_portfolio['Daily Sharpe'] = 0

# Add combined portfolio to result data
result_data.append(combined_portfolio)

# Calculate average daily risk-free rate for reference
avg_daily_rf = testing_returns['rf'].mean()

# Output results focusing on daily metrics
print("\nPortfolio Performance Attribution (Daily Returns):")
print("=" * 100)
print(
    f"{'Portfolio':<10} {'Beta':<8} {'Daily Return':<15} {'Daily StdDev':<15} {'Daily Excess':<15} {'Daily Systematic':<15} {'Daily Idiosyncratic':<15}")
print("-" * 100)

# Filter out stock-level entries and only print portfolio summaries
portfolio_rows = [d for d in result_data if 'Symbol' not in d]

for d in portfolio_rows:
    print(
        f"{d['Portfolio']:<10} {d['Beta']:<8.3f} {d['Avg Daily Return'] * 100:<15.4f}% {d['Daily Std Dev'] * 100:<15.4f}% {d['Avg Daily Excess'] * 100:<15.4f}% {d['Avg Daily Systematic'] * 100:<15.4f}% {d['Avg Daily Idiosyncratic'] * 100:<15.4f}%")

print(f"\nAverage Daily Risk-Free Rate: {avg_daily_rf * 100:.6f}%")
# print("\nNote: Daily Excess Return = Daily Systematic Return + Daily Idiosyncratic Return")

# Step 8: Performance attribution for each portfolio
result_data = []

# Calculate average daily risk-free rate during testing period
avg_rf = testing_returns['rf'].mean()
# Calculate cumulative risk-free rate for the entire testing period
total_rf = (1 + avg_rf) ** len(testing_returns) - 1
# print(f"\nTotal risk-free rate for the testing period: {total_rf * 100:.2f}%")

for portfolio in portfolios:
    portfolio_df = initial_portfolio[initial_portfolio['Portfolio'] == portfolio]
    portfolio_symbols = portfolio_df['Symbol'].unique()

    portfolio_total_return = 0
    portfolio_systematic_return = 0
    portfolio_idiosyncratic_return = 0

    for symbol in portfolio_symbols:
        if symbol not in capm_params:
            print(f"No CAPM model for {symbol}, skipping")
            continue

        weight = portfolio_weights[portfolio][symbol]['weight']
        total_return = total_testing_returns[symbol]

        # CAPM model parameters
        alpha = capm_params[symbol]['alpha']
        beta = capm_params[symbol]['beta']

        # Market return over the testing period
        market_return = total_testing_returns[market_symbol]

        # Calculate systematic and idiosyncratic components
        # Systematic return is beta * (market return - rf)
        systematic_return = beta * (market_return - total_rf)

        # Idiosyncratic return is (total return - rf) - systematic return
        # This ensures Total Return = RF + Systematic + Idiosyncratic
        idiosyncratic_return = (total_return - total_rf) - systematic_return

        # Weight by portfolio allocation
        weighted_total_return = weight * total_return
        weighted_systematic_return = weight * systematic_return
        weighted_idiosyncratic_return = weight * idiosyncratic_return

        # Add to portfolio totals
        portfolio_total_return += weighted_total_return
        portfolio_systematic_return += weighted_systematic_return
        portfolio_idiosyncratic_return += weighted_idiosyncratic_return

        # Store individual stock data
        result_data.append({
            'Portfolio': portfolio,
            'Symbol': symbol,
            'Weight': weight,
            'Total Return': total_return,
            'RF': total_rf,
            'Excess Return': total_return - total_rf,
            'Systematic Return': systematic_return,
            'Idiosyncratic Return': idiosyncratic_return,
            'Beta': beta,
            'Alpha': alpha
        })

# Step 9: Create summary results for portfolios
portfolio_summary = []

for portfolio in portfolios:
    initial_portfolio = [d for d in result_data if d['Portfolio'] == portfolio]

    total_return = sum(d['Weight'] * d['Total Return'] for d in initial_portfolio)
    systematic_return = sum(d['Weight'] * d['Systematic Return'] for d in initial_portfolio)
    idiosyncratic_return = sum(d['Weight'] * d['Idiosyncratic Return'] for d in initial_portfolio)
    excess_return = total_return - total_rf

    # Calculate portfolio beta (weighted average of stock betas)
    portfolio_beta = sum(d['Weight'] * d['Beta'] for d in initial_portfolio)

    portfolio_summary.append({
        'Portfolio': portfolio,
        'Total Return': total_return,
        'RF': total_rf,
        'Excess Return': excess_return,
        'Systematic Return': systematic_return,
        'Idiosyncratic Return': idiosyncratic_return,
        'Beta': portfolio_beta,
        'Initial Value': portfolio_values[portfolio]
    })

# Calculate total portfolio (combined A, B, C)
total_value = sum(portfolio_values.values())
total_weights = {p: portfolio_values[p] / total_value for p in portfolios}

# Calculate combined portfolio metrics
combined_total_return = sum(p['Total Return'] * total_weights[p['Portfolio']] for p in portfolio_summary)
combined_systematic_return = sum(p['Systematic Return'] * total_weights[p['Portfolio']] for p in portfolio_summary)
combined_idiosyncratic_return = sum(
    p['Idiosyncratic Return'] * total_weights[p['Portfolio']] for p in portfolio_summary)
combined_beta = sum(p['Beta'] * total_weights[p['Portfolio']] for p in portfolio_summary)

# Add combined portfolio to summary
portfolio_summary.append({
    'Portfolio': 'Combined',
    'Total Return': combined_total_return,
    'RF': total_rf,
    'Excess Return': combined_total_return - total_rf,
    'Systematic Return': combined_systematic_return,
    'Idiosyncratic Return': combined_idiosyncratic_return,
    'Beta': combined_beta,
    'Initial Value': total_value
})

# Sort by portfolio and then by weight (descending)
sorted_data = sorted(result_data, key=lambda x: (x['Portfolio'], -x['Weight']))

# # Step 10: Output results
# print("\nPortfolio Performance Attribution Summary:")
# print("=" * 80)
# print(
#     f"{'Portfolio':<10} {'Beta':<8} {'Total Return':<15} {'RF':<15} {'Excess Return':<15} {'Systematic':<15} {'Idiosyncratic':<15}")
# print("-" * 90)
#
# for p in portfolio_summary:
#     print(
#         f"{p['Portfolio']:<10} {p['Beta']:<8.2f} {p['Total Return'] * 100:<15.2f}% {p['RF'] * 100:<15.2f}% {p['Excess Return'] * 100:<15.2f}% {p['Systematic Return'] * 100:<15.2f}% {p['Idiosyncratic Return'] * 100:<15.2f}%")
#
# print(f"\nNote: Excess Return = Total Return - Risk Free Rate ({total_rf * 100:.2f}%)")
# print(f"      Excess Return should equal Systematic Return + Idiosyncratic Return")

# Step 11: Perform portfolio risk decomposition (daily basis)
print("\nPortfolio Risk Decomposition (Daily):")
print("=" * 120)

# Calculate market variance for the testing period
market_variance = testing_returns[market_symbol].var()
market_excess_variance = testing_returns[f'{market_symbol}_excess'].var()

print(f"Market (SPY) Daily Variance: {market_variance * 100:.6f}%")

# Create a new dataset for results
risk_data = []
initial_portfolio = pd.read_csv('../Projects/Final Project/initial_portfolio.csv')

for portfolio in portfolios:
    portfolio_df = initial_portfolio[initial_portfolio['Portfolio'] == portfolio]
    portfolio_symbols = portfolio_df['Symbol'].unique()

    # Get portfolio beta (already calculated in previous steps)
    portfolio_beta = [d['Beta'] for d in portfolio_rows if d['Portfolio'] == portfolio][0]

    # Calculate systematic risk (β² × market variance)
    portfolio_systematic_risk = portfolio_beta ** 2 * market_variance

    # Get total portfolio variance (already calculated)
    portfolio_variance = [d['Daily Std Dev'] ** 2 for d in portfolio_rows if d['Portfolio'] == portfolio][0]

    # Calculate idiosyncratic risk (total variance - systematic risk)
    portfolio_idiosyncratic_risk = portfolio_variance - portfolio_systematic_risk

    # Calculate risk proportion
    total_risk = portfolio_variance
    systematic_risk_pct = (portfolio_systematic_risk / total_risk) * 100 if total_risk > 0 else 0
    idiosyncratic_risk_pct = (portfolio_idiosyncratic_risk / total_risk) * 100 if total_risk > 0 else 0

    # Find the existing portfolio summary data
    existing_data = [d for d in portfolio_rows if d['Portfolio'] == portfolio][0]

    # Add risk metrics to results
    risk_data.append({
        'Portfolio': portfolio,
        'Beta': portfolio_beta,
        'Total Variance': portfolio_variance,
        'Systematic Risk': portfolio_systematic_risk,
        'Idiosyncratic Risk': portfolio_idiosyncratic_risk,
        'Systematic Risk %': systematic_risk_pct,
        'Idiosyncratic Risk %': idiosyncratic_risk_pct,
        # Include original return metrics
        'Avg Daily Return': existing_data['Avg Daily Return'],
        'Daily Std Dev': existing_data['Daily Std Dev'],
        'Avg Daily Excess': existing_data['Avg Daily Excess'],
        'Avg Daily Systematic': existing_data['Avg Daily Systematic'],
        'Avg Daily Idiosyncratic': existing_data['Avg Daily Idiosyncratic']
    })

# Add combined portfolio to risk data
combined_data = [d for d in portfolio_rows if d['Portfolio'] == 'Combined'][0]
combined_beta = combined_data['Beta']
combined_variance = combined_data['Daily Std Dev'] ** 2
combined_systematic_risk = combined_beta ** 2 * market_variance
combined_idiosyncratic_risk = combined_variance - combined_systematic_risk

systematic_risk_pct = (combined_systematic_risk / combined_variance) * 100 if combined_variance > 0 else 0
idiosyncratic_risk_pct = (combined_idiosyncratic_risk / combined_variance) * 100 if combined_variance > 0 else 0

risk_data.append({
    'Portfolio': 'Combined',
    'Beta': combined_beta,
    'Total Variance': combined_variance,
    'Systematic Risk': combined_systematic_risk,
    'Idiosyncratic Risk': combined_idiosyncratic_risk,
    'Systematic Risk %': systematic_risk_pct,
    'Idiosyncratic Risk %': idiosyncratic_risk_pct,
    # Include original return metrics
    'Avg Daily Return': combined_data['Avg Daily Return'],
    'Daily Std Dev': combined_data['Daily Std Dev'],
    'Avg Daily Excess': combined_data['Avg Daily Excess'],
    'Avg Daily Systematic': combined_data['Avg Daily Systematic'],
    'Avg Daily Idiosyncratic': combined_data['Avg Daily Idiosyncratic']
})

# Prepare combined output table with returns and risk
print(
    f"{'Portfolio':<10} {'Beta':<8} {'Daily Return':<15} {'Daily StdDev':<15} {'System Risk':<15} {'Idio Risk':<15} {'System Risk%':<15} {'Idio Risk%':<15}")
print("-" * 120)

for d in risk_data:
    print(f"{d['Portfolio']:<10} {d['Beta']:<8.3f} "
          f"{d['Avg Daily Return'] * 100:<15.4f}% {d['Daily Std Dev'] * 100:<15.4f}% "
          f"{d['Systematic Risk'] * 100:<15.4f}% {d['Idiosyncratic Risk'] * 100:<15.4f}% "
          f"{d['Systematic Risk %']:<15.2f}% {d['Idiosyncratic Risk %']:<15.2f}%")

# print(f"\nAverage Daily Risk-Free Rate: {avg_daily_rf * 100:.6f}%")
print("\nNotes:")
print("1. Daily Excess Return = Daily Systematic Return + Daily Idiosyncratic Return")
print("2. Daily Total Return = Daily Excess Return + Risk-Free Rate")
print("3. System Risk = β² × Market Variance")
print("4. Idio Risk = Total Variance - System Risk")
print("5. Total Variance = Daily StdDev²")

# # For verification: Calculate expected total variance using decomposition
# print("\nVariance Decomposition Verification:")
# print("=" * 80)
# print(f"{'Portfolio':<10} {'Total Var':<15} {'System+Idio':<15} {'Difference':<15}")
# print("-" * 80)
#
# for d in risk_data:
#     total_var = d['Total Variance']
#     sum_components = d['Systematic Risk'] + d['Idiosyncratic Risk']
#     diff = total_var - sum_components
#     print(f"{d['Portfolio']:<10} {total_var * 100:<15.6f}% {sum_components * 100:<15.6f}% {diff * 100:<15.6f}%")

# Step 11: Create visualization
plt.figure(figsize=(12, 8))

# Plot returns by component for each portfolio
portfolios_to_plot = [p['Portfolio'] for p in portfolio_summary]
systematic_returns = [p['Systematic Return']*100 for p in portfolio_summary]
idiosyncratic_returns = [p['Idiosyncratic Return']*100 for p in portfolio_summary]

x = range(len(portfolios_to_plot))
width = 0.35

plt.bar(x, systematic_returns, width, label='Systematic Return')
plt.bar(x, idiosyncratic_returns, width, bottom=systematic_returns, label='Idiosyncratic Return')

plt.xlabel('Portfolio')
plt.ylabel('Return (%)')
plt.title('Portfolio Returns Attribution')
plt.xticks(x, portfolios_to_plot)
plt.legend()

plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.savefig('portfolio_attribution.png')
plt.show()
plt.close()