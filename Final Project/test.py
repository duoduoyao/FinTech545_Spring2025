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

# Step 8: Performance attribution for each portfolio
result_data = []

# Calculate average daily risk-free rate during testing period
avg_rf = testing_returns['rf'].mean()
# Calculate cumulative risk-free rate for the entire testing period
total_rf = (1 + avg_rf) ** len(testing_returns) - 1

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

        # 使用实际测试期间的总回报
        total_return = total_testing_returns[symbol]

        # CAPM model parameters
        alpha = capm_params[symbol]['alpha']
        beta = capm_params[symbol]['beta']

        # 市场在测试期间的回报
        market_return = total_testing_returns[market_symbol]

        # 修改: 系统性回报包含风险无关收益
        systematic_return = beta * (market_return - total_rf) + total_rf

        # 修改: 特质性回报是总回报减去系统性回报
        idiosyncratic_return = total_return - systematic_return

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
    portfolio_df = initial_portfolio[initial_portfolio['Portfolio'] == portfolio]

    # 计算测试期开始时的价值 (已有)
    initial_value = portfolio_values[portfolio]

    # 计算测试期结束时的价值
    final_value = 0
    for _, row in portfolio_df.iterrows():
        symbol = row['Symbol']
        holdings = row['Holding']
        # 获取测试期结束时的价格
        final_price = testing_data.iloc[-1][symbol]
        final_value += holdings * final_price

    # 计算总回报率
    portfolio_total_return = (final_value - initial_value) / initial_value

    # 获取投资组合的Beta(加权平均)
    portfolio_stocks = [d for d in result_data if d['Portfolio'] == portfolio and 'Symbol' in d]
    portfolio_beta = sum(d['Weight'] * d['Beta'] for d in portfolio_stocks)

    # 市场回报
    market_return = total_testing_returns[market_symbol]

    # 计算系统性回报(基于Beta和市场回报)
    systematic_return = portfolio_beta * (market_return - total_rf) + total_rf

    # 计算特质性回报(总回报减去系统性回报)
    idiosyncratic_return = portfolio_total_return - systematic_return

    # 获取投资组合的波动率 (使用已经计算好的portfolio_daily_risk字典)
    portfolio_volatility = portfolio_daily_risk[portfolio]['daily_std_dev']

    portfolio_summary.append({
        'Portfolio': portfolio,
        'Total Return': portfolio_total_return,
        'RF': total_rf,
        'Excess Return': portfolio_total_return - total_rf,
        'Systematic Return': systematic_return,
        'Idiosyncratic Return': idiosyncratic_return,
        'Beta': portfolio_beta,
        'Initial Value': initial_value,
        'Final Value': final_value,
        'Volatility': portfolio_volatility
    })

# 计算组合投资组合的波动率 (使用加权平均作为近似值)
total_value = sum(portfolio_values.values())
total_weights = {p: portfolio_values[p] / total_value for p in portfolios}

# 为了更准确，应该使用组合投资组合的实际日度回报计算波动率
# 但如果没有计算过，可以使用加权平均作为估计
combined_volatility = sum(p['Volatility'] * total_weights[p['Portfolio']] for p in portfolio_summary)

total_final_value = sum(p['Final Value'] for p in portfolio_summary)
combined_total_return = (total_final_value - total_value) / total_value
combined_beta = sum(p['Beta'] * total_weights[p['Portfolio']] for p in portfolio_summary)
combined_systematic_return = combined_beta * (total_testing_returns[market_symbol] - total_rf) + total_rf
combined_idiosyncratic_return = combined_total_return - combined_systematic_return

# Add combined portfolio to summary
portfolio_summary.append({
    'Portfolio': 'Combined',
    'Total Return': combined_total_return,
    'RF': total_rf,
    'Excess Return': combined_total_return - total_rf,
    'Systematic Return': combined_systematic_return,
    'Idiosyncratic Return': combined_idiosyncratic_return,
    'Beta': combined_beta,
    'Initial Value': total_value,
    'Final Value': total_final_value,
    'Volatility': combined_volatility
})


    # portfolio_stocks = [d for d in result_data if d['Portfolio'] == portfolio and 'Symbol' in d]
    #
    #
    # total_return = sum(d['Weight'] * d['Total Return'] for d in portfolio_stocks)
    # systematic_return = sum(d['Weight'] * d['Systematic Return'] for d in portfolio_stocks)
    # idiosyncratic_return = sum(d['Weight'] * d['Idiosyncratic Return'] for d in portfolio_stocks)
    # excess_return = total_return - total_rf
    #
    # # 计算组合的平均alpha (加权)
    # portfolio_alpha = sum(d['Weight'] * d['Alpha'] for d in portfolio_stocks)
    #
    # # Calculate portfolio beta (weighted average of stock betas)
    # portfolio_beta = sum(d['Weight'] * d['Beta'] for d in portfolio_stocks)
    #
    # # 计算组合的波动率
    # # 获取此组合的日度回报数据
    # portfolio_returns = portfolio_daily_returns[portfolio]['portfolio_return']
    # portfolio_volatility = portfolio_returns.std()
    #
    # portfolio_summary.append({
    #     'Portfolio': portfolio,
    #     'Total Return': total_return,
    #     'RF': total_rf,
    #     'Excess Return': excess_return,
    #     'Systematic Return': systematic_return,
    #     'Idiosyncratic Return': idiosyncratic_return,
    #     'Beta': portfolio_beta,
    #     'Alpha': portfolio_alpha,
    #     'Volatility': portfolio_volatility,
    #     'Initial Value': portfolio_values[portfolio]
    # })
#
# # Calculate total portfolio (combined A, B, C)
# total_value = sum(portfolio_values.values())
# total_weights = {p: portfolio_values[p] / total_value for p in portfolios}
#
# # Calculate combined portfolio metrics
# combined_total_return = sum(p['Total Return'] * total_weights[p['Portfolio']] for p in portfolio_summary)
# combined_systematic_return = sum(p['Systematic Return'] * total_weights[p['Portfolio']] for p in portfolio_summary)
# combined_idiosyncratic_return = sum(
#     p['Idiosyncratic Return'] * total_weights[p['Portfolio']] for p in portfolio_summary)
# combined_beta = sum(p['Beta'] * total_weights[p['Portfolio']] for p in portfolio_summary)
# combined_alpha = sum(p['Alpha'] * total_weights[p['Portfolio']] for p in portfolio_summary)
#
# # 计算组合的波动率贡献
# market_volatility = testing_returns[market_symbol].std()
# combined_volatility = sum(p['Volatility'] * total_weights[p['Portfolio']] for p in portfolio_summary)
#
# # Add combined portfolio to summary
# portfolio_summary.append({
#     'Portfolio': 'Combined',
#     'Total Return': combined_total_return,
#     'RF': total_rf,
#     'Excess Return': combined_total_return - total_rf,
#     'Systematic Return': combined_systematic_return,
#     'Idiosyncratic Return': combined_idiosyncratic_return,
#     'Beta': combined_beta,
#     'Alpha': combined_alpha,
#     'Volatility': combined_volatility,
#     'Initial Value': total_value
# })


# 创建图片中格式的输出函数
def create_attribution_table(title, data):
    # 市场回报
    market_return = total_testing_returns[market_symbol]

    # 计算回报归因 (基于beta和市场超额回报)
    return_attribution = data['Beta'] * (market_return - total_rf) + total_rf

    # 计算波动归因 (基于beta和市场波动率)
    market_vol = testing_returns[market_symbol].std()
    vol_attribution = data['Beta'] * market_vol

    # 计算alpha贡献 (实际回报与基于beta预测的回报之差)
    alpha_contribution = data['Total Return'] - return_attribution

    # 计算波动率的alpha贡献 (实际波动率与基于beta预测的波动率之差)
    vol_alpha_contribution = data['Volatility'] - vol_attribution

    print(f"\n# {title}")
    print("#    | Value              | SPY        | Alpha      | Portfolio")
    print("# ---|--------------------|------------|------------|------------")
    print(f"#  1 | Total Return       |  {market_return:.6f}  | {alpha_contribution:.6f} | {data['Total Return']:.6f}")
    print(
        f"#  2 | Return Attribution |  {return_attribution:.6f}  | {alpha_contribution:.6f} | {data['Total Return']:.6f}")
    print(f"#  3 | Vol Attribution    |  {vol_attribution:.6f}  | {vol_alpha_contribution:.8f} | {data['Volatility']:.8f}")


# 为每个投资组合创建表格，包括总组合
# 先创建总组合的表格
combined_portfolio = next((p for p in portfolio_summary if p['Portfolio'] == 'Combined'), None)
if combined_portfolio:
    create_attribution_table("Total Portfolio Attribution", combined_portfolio)

# 分别为A、B、C组合创建表格
for portfolio in portfolios:
    portfolio_data = next((p for p in portfolio_summary if p['Portfolio'] == portfolio), None)
    if portfolio_data:
        create_attribution_table(f"Portfolio {portfolio} Attribution", portfolio_data)




for portfolio in portfolios:
    portfolio_df = initial_portfolio[initial_portfolio['Portfolio'] == portfolio]

    # 旧方法：使用加权平均计算
    portfolio_stocks = [d for d in result_data if d['Portfolio'] == portfolio and 'Symbol' in d]
    old_method_return = sum(d['Weight'] * d['Total Return'] for d in portfolio_stocks)

    # 新方法：使用价值变化计算
    initial_value = portfolio_values[portfolio]
    final_value = 0
    for _, row in portfolio_df.iterrows():
        symbol = row['Symbol']
        holdings = row['Holding']
        final_price = testing_data.iloc[-1][symbol]
        final_value += holdings * final_price

    new_method_return = (final_value - initial_value) / initial_value

    print(f"Portfolio {portfolio}:")
    print(f"  旧方法回报率: {old_method_return:.6f}")
    print(f"  新方法回报率: {new_method_return:.6f}")
    print(f"  差异: {(new_method_return - old_method_return):.6f}")
    print()