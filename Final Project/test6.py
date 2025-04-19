import pandas as pd
import numpy as np

# 假设已经有testing_data和testing_returns
# 如果你需要重新加载数据，取消下面注释的代码
daily_prices = pd.read_csv('../Projects/Final Project/DailyPrices.csv')
daily_prices['Date'] = pd.to_datetime(daily_prices['Date'])
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

# 计算SPY的日收益率统计数据
spy_daily_returns = testing_returns['SPY']
spy_mean_daily_return = spy_daily_returns.mean()
spy_std_daily_return = spy_daily_returns.std()

# 计算年化数据（假设一年有252个交易日）
trading_days_per_year = 252
spy_annualized_return = (1 + spy_mean_daily_return) ** trading_days_per_year - 1
spy_annualized_volatility = spy_std_daily_return * np.sqrt(trading_days_per_year)

# 计算测试期间的总收益率
spy_first_price = testing_data.iloc[0]['SPY']
spy_last_price = testing_data.iloc[-1]['SPY']
spy_total_return = (spy_last_price / spy_first_price) - 1


# 计算测试期间的年化总收益率
# 计算测试期间的交易日数量
trading_days_in_period = len(testing_data)
years_in_period = trading_days_in_period / trading_days_per_year
spy_annualized_total_return = (1 + spy_total_return) ** (1 / years_in_period) - 1

# 打印结果
print(f"SPY测试期间统计数据:")
print(f"测试天数: {trading_days_in_period}天")
print(f"平均日收益率: {spy_mean_daily_return * 100:.4f}%")
print(f"日收益率标准差: {spy_std_daily_return * 100:.4f}%")
print(f"总收益率: {spy_total_return * 100:.2f}%")
print(f"年化总收益率: {spy_annualized_total_return * 100:.2f}%")
print(f"年化波动率: {spy_annualized_volatility * 100:.2f}%")

# 计算最大回撤
cumulative_returns = (1 + spy_daily_returns).cumprod()
rolling_max = cumulative_returns.cummax()
drawdown = (cumulative_returns / rolling_max) - 1
max_drawdown = drawdown.min()
print(f"最大回撤: {max_drawdown * 100:.2f}%")

# 计算夏普比率（假设无风险利率已经在testing_returns['rf']中）
if 'rf' in testing_returns.columns:
    avg_rf = testing_returns['rf'].mean()
    sharpe_ratio = (spy_mean_daily_return - avg_rf) / spy_std_daily_return
    annualized_sharpe = sharpe_ratio * np.sqrt(trading_days_per_year)
    print(f"日夏普比率: {sharpe_ratio:.4f}")
    print(f"年化夏普比率: {annualized_sharpe:.4f}")