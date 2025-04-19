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


# =====================  Part 2 – Optimal Max‑Sharpe Portfolios  ===================== #
import numpy as np
from scipy.optimize import minimize

# ------------------------------------------------------------------
# Step 10 :  训练期市场统计量（CAPM 预测用）
# ------------------------------------------------------------------
expected_market_return = training_returns[market_symbol].mean()        # 日度均值
expected_rf            = training_returns['rf'].mean()                 # 日度均值
expected_market_excess = expected_market_return - expected_rf
market_variance        = training_returns[f'{market_symbol}_excess'].var()

print(f"\n>>> 训练期平均市场日收益  : {expected_market_return*100:8.4f}%")
print(f">>> 训练期平均无风险日收益: {expected_rf*100:8.4f}%")
print(f">>> 训练期平均市场超额收益: {expected_market_excess*100:8.4f}%")

# ------------------------------------------------------------------
# Step 11 :  为每个子组合求最大 Sharpe 比权重
# ------------------------------------------------------------------
optimal_weights   = {}   # {portfolio: {symbol: weight}}
optimal_metrics   = {}   # 存储预期收益 / 风险 / Sharpe（基于模型预测）

for portfolio in portfolios:
    # 取该子组合包含的股票
    port_syms = initial_portfolio.loc[
        initial_portfolio['Portfolio'] == portfolio, 'Symbol'
    ].unique()

    # 只留下在 CAPM 里有 β 的股票
    symbols_to_opt = [s for s in port_syms if s in capm_params]
    if len(symbols_to_opt) == 0:
        print(f"⚠️  Portfolio {portfolio}: 没有可优化的股票，跳过")
        continue

    # ---------- 组装协方差矩阵 & 期望超额收益 ----------
    betas      = np.array([capm_params[s]['beta'] for s in symbols_to_opt])
    idio_vars  = np.array([               # residual variance
        training_returns[f'{s}_excess'] - betas[i] * training_returns[f'{market_symbol}_excess']
        for i, s in enumerate(symbols_to_opt)
    ])
    idio_vars  = idio_vars.var(axis=1)

    # 协方差矩阵：Σ = β_i β_j σ_m^2  +  δ_ij σ²_εi
    cov_mtx = np.outer(betas, betas) * market_variance
    np.fill_diagonal(cov_mtx, np.diag(cov_mtx) + idio_vars)

    exp_excess = betas * expected_market_excess       # 期望超额收益 (α=0)

    # ---------- Sharpe 最优化 ----------
    def neg_sharpe(w):
        port_ret  = np.dot(w, exp_excess)                     # 期望超额收益
        port_vol  = np.sqrt(np.dot(w.T, cov_mtx @ w))         # 期望年化？→ 先日度
        return - port_ret / port_vol if port_vol > 0 else 0

    cons   = ({'type': 'eq', 'fun': lambda w: w.sum() - 1})
    bounds = tuple((0, 1) for _ in symbols_to_opt)
    w0     = np.ones(len(symbols_to_opt)) / len(symbols_to_opt)

    res = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=cons)
    if not res.success:
        print(f"⚠️  Optimization failed for {portfolio}: {res.message}")
        continue

    w_opt = res.x
    optimal_weights[portfolio] = dict(zip(symbols_to_opt, w_opt))

    # 记录模型期望值（年化：×√252）
    model_ret  = np.dot(w_opt, exp_excess) * 252                     # 年化超额收益
    model_vol  = np.sqrt(np.dot(w_opt.T, cov_mtx @ w_opt)) * np.sqrt(252)
    model_sharpe = model_ret / model_vol if model_vol > 0 else 0

    optimal_metrics[portfolio] = {
        'Model Yearly Excess': model_ret,
        'Model Yearly Vol'  : model_vol,
        'Model Sharpe'      : model_sharpe,
    }

# ------------------------------------------------------------------
# Step 12 :  计算最优组合在持有期（2024‑25）的实际表现
# ------------------------------------------------------------------
optimal_result_data = []   # 归因结果以行存
optimal_daily       = {}   # 日度 DataFrame

for portfolio, w_dict in optimal_weights.items():

    # 1) 计算日度组合收益
    df = pd.DataFrame({'Date': testing_returns['Date']})
    df['portfolio_return'] = 0.0
    for sym, w in w_dict.items():
        df['portfolio_return'] += testing_returns[sym] * w
    df['excess_return'] = df['portfolio_return'] - testing_returns['rf']

    # 2) 实际总收益、年化收益 & 波动率
    cum_ret = (1 + df['portfolio_return']).prod() - 1
    yearly_ret = cum_ret                                           # 持有期 ≈ 1 年
    yearly_vol = df['portfolio_return'].std() * np.sqrt(252)
    yearly_excess = yearly_ret - total_rf
    sharpe_real = yearly_excess / yearly_vol if yearly_vol > 0 else 0

    # 3) 基于 β 的系统 / 特质风险归因
    port_beta = sum(capm_params[s]['beta'] * w for s, w in w_dict.items())
    sys_ret   = port_beta * (total_testing_returns[market_symbol] - total_rf) + total_rf
    idio_ret  = yearly_ret - sys_ret

    # 年化方差分解
    port_var  = yearly_vol**2
    sys_var   = (port_beta**2) * (market_variance * 252)   # 年化市场方差 = 日方差×252
    idio_var  = port_var - sys_var

    # 期望特质风险（训练期估值）
    expected_idio_var = sum((w**2) * idio_vars[i] * 252        # 年化
                            for i, (s, w) in enumerate(w_dict.items()))

    # ---- 保存到结果表 ----
    optimal_result_data.append({
        'Portfolio'            : f'{portfolio} (Optimal)',
        'Yearly Return'        : yearly_ret,
        'Yearly Excess'        : yearly_excess,
        'Yearly Vol'           : yearly_vol,
        'Sharpe Ratio'         : sharpe_real,
        'Systematic Return'    : sys_ret,
        'Idiosyncratic Return' : idio_ret,
        'System Risk %'        : sys_var / port_var * 100 if port_var>0 else 0,
        'Idio Risk %'          : idio_var / port_var * 100 if port_var>0 else 0,
        'Expected Idio Var'    : expected_idio_var,
        'Realized Idio Var'    : idio_var,
        'Beta'                 : port_beta
    })

# ------------------------------------------------------------------
# Step 13 :  把最优组合结果追加到 portfolio_summary，方便同表比较
# ------------------------------------------------------------------
for d in optimal_result_data:
    portfolio_summary.append({
        'Portfolio'            : d['Portfolio'],
        'Total Return'         : d['Yearly Return'],
        'Excess Return'        : d['Yearly Excess'],
        'Volatility'           : d['Yearly Vol'],
        'Sharpe Ratio'         : d['Sharpe Ratio'],
        'Systematic Return'    : d['Systematic Return'],
        'Idiosyncratic Return' : d['Idiosyncratic Return'],
        'System Risk %'        : d['System Risk %'],
        'Idiosyncratic Risk %' : d['Idio Risk %'],
        'Expected Idio Var'    : d['Expected Idio Var'],
        'Realized Idio Var'    : d['Realized Idio Var'],
        'Beta'                 : d['Beta']
    })

# ------------------------------------------------------------------
# Step 14 :  用同一模板打印 Attribution 表（原组合 & 最优组合）
# ------------------------------------------------------------------
def print_attr_table(title, record):
    mkt_ret  = total_testing_returns[market_symbol]
    mkt_vol  = testing_returns[market_symbol].std()

    return_attr = record['Beta'] * (mkt_ret - total_rf) + total_rf
    vol_attr    = record['Beta'] * mkt_vol
    alpha_ret   = record['Total Return'] - return_attr
    alpha_vol   = record['Volatility']   - vol_attr

    print(f"\n# {title}")
    print("#    | Value              | SPY        | Alpha      | Portfolio")
    print("# ---|--------------------|------------|------------|------------")
    print(f"#  1 | Total Return       |  {mkt_ret:.6f}  | {alpha_ret:.6f} | {record['Total Return']:.6f}")
    print(f"#  2 | Return Attribution |  {return_attr:.6f}  | {alpha_ret:.6f} | {record['Total Return']:.6f}")
    print(f"#  3 | Vol Attribution    |  {vol_attr:.6f}  | {alpha_vol:.8f} | {record['Volatility']:.8f}")
    print(f"#  4 | Sharpe Ratio       |              |            | {record['Sharpe Ratio']:.6f}")

# —— 原组合
for rec in portfolio_summary:
    if rec['Portfolio'] in portfolios:               # 只打印 A/B/C
        print_attr_table(f"Portfolio {rec['Portfolio']} Attribution (Original)", rec)

# —— 最优组合
for rec in portfolio_summary:
    if rec['Portfolio'].endswith('(Optimal)'):
        print_attr_table(f"Portfolio {rec['Portfolio']} Attribution", rec)

# ------------------------------------------------------------------
# Step 15 :  模型期望 vs 实现 特质风险对比一览
# ------------------------------------------------------------------
print("\n# Expected vs Realized Idiosyncratic Variance")
print("# -------------------------------------------")
print(f"# {'Portfolio':<20} | {'Expected':>12} | {'Realized':>12} | {'Diff %':>8}")
print("# --------------------|--------------|--------------|--------")
for rec in portfolio_summary:
    if rec['Portfolio'].endswith('(Optimal)'):
        exp_  = rec['Expected Idio Var']
        real_ = rec['Realized Idio Var']
        diff  = (real_ - exp_) / exp_ * 100 if exp_ > 0 else np.nan
        print(f"# {rec['Portfolio']:<20} | {exp_*100:10.4f}% | {real_*100:10.4f}% | {diff:7.2f}%")
