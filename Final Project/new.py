import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize


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


# Step 4: Prepare the data for CAPM regression
# Merge risk-free rate with returns
def prepare_capm_data(returns_df, rf_df):
    # 确保日期格式一致
    merged_df = pd.merge(returns_df, rf_df[['Date', 'rf']], on='Date', how='left')

    # 计算超额收益率 (不需要在这里使用，因为我们选择了Option 1)
    # 但为了完整性，我们还是计算一下
    excess_returns = merged_df.copy()
    for column in merged_df.columns:
        if column not in ['Date', 'rf']:
            excess_returns[column + '_excess'] = merged_df[column] - merged_df['rf']

    return merged_df


# 准备训练数据
capm_training_data = prepare_capm_data(training_returns, risk_free)


# Step 5: Run CAPM regression for each stock
def run_capm_regression(data, market_ticker='SPY'):
    # 创建一个字典来存储每只股票的回归结果
    capm_results = {}

    # 获取市场收益率
    market_returns = data[market_ticker]

    for column in data.columns:
        if column not in ['Date', 'Rf', market_ticker] and 'excess' not in column:
            # 根据选择的Option 1，我们使用原始收益率而不是超额收益率
            # 创建X变量（市场收益率）和Y变量（股票收益率）
            X = market_returns.values.reshape(-1, 1)
            y = data[column].values

            # 添加常数项
            X = sm.add_constant(X)

            # 运行回归
            model = sm.OLS(y, X)
            results = model.fit()

            # 存储结果（alpha和beta）
            capm_results[column] = {
                'alpha': results.params[0],
                'beta': results.params[1],
                'r_squared': results.rsquared
            }

    return capm_results


# 运行CAPM回归
capm_results = run_capm_regression(capm_training_data)


# Step 6: Perform risk attribution for the holding period
def perform_risk_attribution(capm_results, test_returns, portfolio_df):
    # 创建一个空的DataFrame来存储归因结果
    attribution_results = pd.DataFrame(index=portfolio_df['Symbol'].unique())

    # 计算持有期间的总收益率（累积收益率）
    test_cumulative_returns = (1 + test_returns.drop('Date', axis=1)).prod() - 1

    # 获取投资组合信息
    portfolios = portfolio_df['Portfolio'].unique()

    # 创建一个字典来存储每个投资组合的归因结果
    portfolio_attribution = {portfolio: {} for portfolio in portfolios}
    portfolio_attribution['Total'] = {}

    # 对每只股票进行归因分析
    for stock in portfolio_df['Symbol'].unique():
        if stock in capm_results and stock in test_cumulative_returns.index:
            # 获取CAPM参数
            beta = capm_results[stock]['beta']

            # 获取股票和市场的实际收益率
            stock_return = test_cumulative_returns[stock]
            market_return = test_cumulative_returns['SPY']

            # 计算系统性收益率（Option 1：保留无风险利率在系统性桶中）
            systematic_return = beta * market_return

            # 计算非系统性收益率
            idiosyncratic_return = stock_return - systematic_return

            # 存储结果
            attribution_results.loc[stock, 'Beta'] = beta
            attribution_results.loc[stock, 'Total Return'] = stock_return
            attribution_results.loc[stock, 'Systematic Return'] = systematic_return
            attribution_results.loc[stock, 'Idiosyncratic Return'] = idiosyncratic_return

    # 对每个投资组合进行加权归因分析
    for portfolio in portfolios:
        # 获取投资组合中的股票
        portfolio_stocks = portfolio_df[portfolio_df['Portfolio'] == portfolio]

        # 计算投资组合的加权收益率
        total_weight = portfolio_stocks['Holding'].sum()
        portfolio_total_return = 0
        portfolio_systematic_return = 0
        portfolio_idiosyncratic_return = 0

        for _, row in portfolio_stocks.iterrows():
            stock = row['Symbol']
            weight = row['Holding'] / total_weight  # 归一化权重

            if stock in attribution_results.index:
                portfolio_total_return += weight * attribution_results.loc[stock, 'Total Return']
                portfolio_systematic_return += weight * attribution_results.loc[stock, 'Systematic Return']
                portfolio_idiosyncratic_return += weight * attribution_results.loc[stock, 'Idiosyncratic Return']

        # 存储投资组合的归因结果
        portfolio_attribution[portfolio]['Total Return'] = portfolio_total_return
        portfolio_attribution[portfolio]['Systematic Return'] = portfolio_systematic_return
        portfolio_attribution[portfolio]['Idiosyncratic Return'] = portfolio_idiosyncratic_return

    # 计算总投资组合的归因结果
    total_portfolio_value = portfolio_df['Holding'].sum()
    total_return = 0
    total_systematic_return = 0
    total_idiosyncratic_return = 0

    for portfolio in portfolios:
        portfolio_weight = portfolio_df[portfolio_df['Portfolio'] == portfolio]['Holding'].sum() / total_portfolio_value
        total_return += portfolio_weight * portfolio_attribution[portfolio]['Total Return']
        total_systematic_return += portfolio_weight * portfolio_attribution[portfolio]['Systematic Return']
        total_idiosyncratic_return += portfolio_weight * portfolio_attribution[portfolio]['Idiosyncratic Return']

    portfolio_attribution['Total']['Total Return'] = total_return
    portfolio_attribution['Total']['Systematic Return'] = total_systematic_return
    portfolio_attribution['Total']['Idiosyncratic Return'] = total_idiosyncratic_return

    return attribution_results, portfolio_attribution


# 执行风险归因
stock_attribution, portfolio_attribution = perform_risk_attribution(capm_results, testing_returns, initial_portfolio)


# 打印结果
def print_attribution_results(stock_attr, portfolio_attr):
    print("股票级别风险归因结果:")
    print(stock_attr)
    print("\n投资组合级别风险归因结果:")
    for portfolio, results in portfolio_attr.items():
        print(f"投资组合 {portfolio}:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}")
        print()


print_attribution_results(stock_attribution, portfolio_attribution)


# Step 7: Calculate expected returns using CAPM
def calculate_expected_returns(capm_results, avg_market_return, avg_rf):
    expected_returns = {}

    for stock, params in capm_results.items():
        # 假设alpha为0，根据CAPM计算期望收益率
        # 由于选择了Option 1，我们实际上是在计算：E(Ri) = β * E(Rm)
        expected_returns[stock] = params['beta'] * avg_market_return

    return expected_returns


# 计算持有期前的平均市场收益率和无风险利率
avg_market_return = training_returns['SPY'].mean()
avg_rf = risk_free[risk_free['Date'] <= pd.Timestamp('2023-12-29')]['rf'].mean()

# 计算每只股票的期望收益率
expected_returns = calculate_expected_returns(capm_results, avg_market_return, avg_rf)


# Step 8: Calculate covariance matrix for returns
def calculate_covariance(returns_df, stocks):
    return returns_df[stocks].cov()


# Step 9: Optimize for maximum Sharpe ratio
def optimize_portfolio(expected_returns, cov_matrix, risk_free_rate, stocks):
    num_assets = len(stocks)

    # 初始权重：等权重
    initial_weights = np.array([1 / num_assets] * num_assets)

    # 定义约束条件：权重之和为1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    # 定义边界条件：权重非负（不允许卖空）
    bounds = tuple((0, 1) for asset in range(num_assets))

    # 定义目标函数：最大化夏普比率
    def neg_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate):
        # 投资组合期望收益率
        port_return = np.sum(weights * np.array([expected_returns[stock] for stock in stocks]))

        # 投资组合风险（标准差）
        port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        # 计算夏普比率（由于是最大化问题，我们返回负的夏普比率）
        return -(port_return - risk_free_rate) / port_risk

    # 执行优化
    optimal_results = minimize(neg_sharpe_ratio, initial_weights,
                               args=(expected_returns, cov_matrix, risk_free_rate),
                               method='SLSQP', bounds=bounds, constraints=constraints)

    # 获取最优权重
    optimal_weights = optimal_results['x']

    # 计算最优投资组合的期望收益率和风险
    optimal_return = np.sum(optimal_weights * np.array([expected_returns[stock] for stock in stocks]))
    optimal_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
    optimal_sharpe = (optimal_return - risk_free_rate) / optimal_risk

    # 创建结果字典
    results = {
        'weights': {stocks[i]: optimal_weights[i] for i in range(len(stocks))},
        'expected_return': optimal_return,
        'expected_risk': optimal_risk,
        'sharpe_ratio': optimal_sharpe
    }

    return results


# 为每个子投资组合创建最优夏普比率投资组合
def create_optimal_portfolios(expected_returns, training_returns, avg_rf, initial_portfolio):
    portfolios = initial_portfolio['Portfolio'].unique()
    optimal_portfolios = {}

    for portfolio in portfolios:
        # 获取该投资组合中的股票
        portfolio_stocks = initial_portfolio[initial_portfolio['Portfolio'] == portfolio]['Symbol'].tolist()

        # 计算协方差矩阵
        cov_matrix = calculate_covariance(training_returns, portfolio_stocks)

        # 优化投资组合
        optimal_results = optimize_portfolio(expected_returns, cov_matrix, avg_rf, portfolio_stocks)

        # 存储结果
        optimal_portfolios[portfolio] = optimal_results

    return optimal_portfolios


# 创建最优投资组合
optimal_portfolios = create_optimal_portfolios(expected_returns, training_returns, avg_rf, initial_portfolio)


# Step 10: Create a new portfolio DataFrame with optimal weights
def create_optimal_portfolio_df(initial_portfolio, optimal_portfolios):
    # 复制原始投资组合DataFrame
    optimal_portfolio_df = initial_portfolio.copy()

    # 更新权重为最优权重
    for i, row in optimal_portfolio_df.iterrows():
        portfolio = row['Portfolio']
        stock = row['Symbol']

        if portfolio in optimal_portfolios and stock in optimal_portfolios[portfolio]['weights']:
            optimal_portfolio_df.at[i, 'Holding'] = optimal_portfolios[portfolio]['weights'][stock]

    return optimal_portfolio_df


# 创建最优投资组合DataFrame
optimal_portfolio_df = create_optimal_portfolio_df(initial_portfolio, optimal_portfolios)

# Step 11: Rerun the attribution analysis with optimal portfolios
optimal_stock_attribution, optimal_portfolio_attribution = perform_risk_attribution(
    capm_results, testing_returns, optimal_portfolio_df)

# 打印最优投资组合的归因结果
print("最优夏普比率投资组合的风险归因结果:")
print_attribution_results(optimal_stock_attribution, optimal_portfolio_attribution)


# Step 12: Compare the results between initial and optimal portfolios
def compare_portfolios(initial_attr, optimal_attr):
    portfolios = list(initial_attr.keys())

    comparison = {}
    for portfolio in portfolios:
        if portfolio in initial_attr and portfolio in optimal_attr:
            initial_return = initial_attr[portfolio].get('Total Return', 0)
            optimal_return = optimal_attr[portfolio].get('Total Return', 0)

            initial_sys_return = initial_attr[portfolio].get('Systematic Return', 0)
            optimal_sys_return = optimal_attr[portfolio].get('Systematic Return', 0)

            initial_idio_return = initial_attr[portfolio].get('Idiosyncratic Return', 0)
            optimal_idio_return = optimal_attr[portfolio].get('Idiosyncratic Return', 0)

            comparison[portfolio] = {
                'Initial Total Return': initial_return,
                'Optimal Total Return': optimal_return,
                'Return Improvement': optimal_return - initial_return,
                'Initial Systematic Return': initial_sys_return,
                'Optimal Systematic Return': optimal_sys_return,
                'Initial Idiosyncratic Return': initial_idio_return,
                'Optimal Idiosyncratic Return': optimal_idio_return
            }

    return comparison


# 比较原始投资组合和最优投资组合
portfolio_comparison = compare_portfolios(portfolio_attribution, optimal_portfolio_attribution)

# 打印比较结果
print("\n原始投资组合与最优投资组合比较:")
for portfolio, metrics in portfolio_comparison.items():
    print(f"投资组合 {portfolio}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    print()


# Step 13: Compare model expectations vs. realized idiosyncratic risk
def compare_idiosyncratic_risk(capm_results, stock_attribution):
    comparison = pd.DataFrame(index=stock_attribution.index)

    for stock in comparison.index:
        if stock in capm_results:
            # 模型预期的特异风险（1 - R^2）
            expected_idio_risk = 1 - capm_results[stock]['r_squared']

            # 实际特异风险贡献（绝对值）
            realized_idio_risk = abs(stock_attribution.loc[stock, 'Idiosyncratic Return'])

            comparison.loc[stock, 'Expected Idiosyncratic Risk'] = expected_idio_risk
            comparison.loc[stock, 'Realized Idiosyncratic Risk'] = realized_idio_risk
            comparison.loc[stock, 'Difference'] = realized_idio_risk - expected_idio_risk

    return comparison


# 比较特异风险模型预期与实际表现
idiosyncratic_risk_comparison = compare_idiosyncratic_risk(capm_results, stock_attribution)
print("\n特异风险模型预期与实际表现比较:")
print(idiosyncratic_risk_comparison)