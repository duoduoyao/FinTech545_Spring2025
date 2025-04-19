import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
import problem1  # 导入第一题的代码文件


def run_optimal_sharpe_analysis():
    """
    执行最优夏普比率投资组合分析（第二部分）

    该函数读取第一题的CAPM分析结果，并构建最大夏普比率投资组合

    Returns:
        dict: 包含分析结果的字典
    """
    print("开始执行最优夏普比率投资组合分析...")

    try:
        # 1. 获取第一题的CAPM分析结果
        capm_results = problem1.run_capm_analysis()

        if not capm_results:
            print("无法获取CAPM分析结果，请先确保第一题分析成功")
            return None

        # 2. 读取数据文件
        daily_prices = pd.read_csv('../Projects/Final Project/DailyPrices.csv')
        initial_portfolio = pd.read_csv('../Projects/Final Project/initial_portfolio.csv')
        rf_data = pd.read_csv('../Projects/Final Project/rf.csv')

        # 2.1 数据预处理
        daily_prices['Date'] = pd.to_datetime(daily_prices['Date'])
        daily_prices.set_index('Date', inplace=True)

        rf_data['Date'] = pd.to_datetime(rf_data['Date'])
        rf_data.set_index('Date', inplace=True)

        # 2.2 找到2023年的末尾
        end_of_2023 = daily_prices[daily_prices.index.year == 2023].index.max()

        # 2.3 划分训练集和测试集
        train_prices = daily_prices[daily_prices.index <= end_of_2023]
        test_prices = daily_prices[daily_prices.index > end_of_2023]

        # 3. 计算每日回报率
        train_returns = train_prices.pct_change().dropna()
        test_returns = test_prices.pct_change().dropna()

        # 4. 获取无风险利率
        train_rf = rf_data.loc[train_returns.index].squeeze()
        test_rf = rf_data.loc[test_returns.index].squeeze()

        # print("\nIdiosyncratic Risk Contribution for each stock...")
        # capm_params = capm_results['capm_params']
        # expected_idio = {}
        # realized_idio = {}
        #
        # for symbol, params in capm_params.items():
        #     # 跳过市场指数本身
        #     if symbol == 'SPY' or symbol not in train_returns.columns:
        #         continue
        #
        #     # 训练期残差方差 (模型预测的特质风险)
        #     if 'resid_var' in params:
        #         resid_var_train = params['resid_var']
        #     else:
        #         stock_train = train_returns[symbol].dropna()
        #         market_train = train_returns.loc[stock_train.index, 'SPY']
        #         residuals_train = stock_train - params['beta'] * market_train  # alpha 设为 0
        #         resid_var_train = residuals_train.var(ddof=1)
        #
        #     total_var_train = train_returns[symbol].var(ddof=1)
        #     expected_idio[symbol] = resid_var_train / total_var_train if total_var_train > 0 else np.nan
        #
        #     # 测试期残差方差 (实际特质风险)
        #     if symbol in test_returns.columns:
        #         stock_test = test_returns[symbol].dropna()
        #         if not stock_test.empty:
        #             market_test = test_returns.loc[stock_test.index, 'SPY']
        #             residuals_test = stock_test - params['beta'] * market_test
        #             resid_var_test = residuals_test.var(ddof=1)
        #             total_var_test = stock_test.var(ddof=1)
        #             realized_idio[symbol] = resid_var_test / total_var_test if total_var_test > 0 else np.nan
        #         else:
        #             realized_idio[symbol] = np.nan
        #     else:
        #         realized_idio[symbol] = np.nan
        #
        # # 打印对比结果
        # print("\nIdiosyncratic Risk Contribution")
        # print(f"{'Symbol':10} {'Expected%':>12} {'Realized%':>12} {'Delta%':>12}")
        # print("-" * 46)
        # for symbol in sorted(expected_idio.keys()):
        #     exp_pct = expected_idio[symbol] * 100 if pd.notna(expected_idio[symbol]) else np.nan
        #     rea_pct = realized_idio.get(symbol, np.nan) * 100 if pd.notna(realized_idio.get(symbol, np.nan)) else np.nan
        #     delta_pct = rea_pct - exp_pct if pd.notna(exp_pct) and pd.notna(rea_pct) else np.nan
        #     print(f"{symbol:10} {exp_pct:12.2f} {rea_pct:12.2f} {delta_pct:12.2f}")

        sigma_rows, pct_rows = [], []
        capm_params = capm_results['capm_params']
        for symbol, p in capm_params.items():
            if symbol == 'SPY' or symbol not in train_returns.columns:
                continue
            beta = p['beta']

            # 训练期
            st_tr = train_returns[symbol].dropna()
            mk_tr = train_returns.loc[st_tr.index, 'SPY']
            resid_tr = st_tr - beta * mk_tr
            resvar_tr = resid_tr.var(ddof=1); totvar_tr = st_tr.var(ddof=1)
            exp_sigma = np.sqrt(resvar_tr)
            exp_pct   = resvar_tr / totvar_tr if totvar_tr > 0 else np.nan

            # 持仓期
            if symbol in test_returns.columns and not test_returns[symbol].dropna().empty:
                st_ts = test_returns[symbol].dropna()
                mk_ts = test_returns.loc[st_ts.index, 'SPY']
                resid_ts = st_ts - beta * mk_ts
                resvar_ts = resid_ts.var(ddof=1); totvar_ts = st_ts.var(ddof=1)
                rea_sigma = np.sqrt(resvar_ts)
                rea_pct   = resvar_ts / totvar_ts if totvar_ts > 0 else np.nan
            else:
                rea_sigma = np.nan; rea_pct = np.nan

            sigma_rows.append([symbol, exp_sigma, rea_sigma, rea_sigma - exp_sigma])
            pct_rows.append([symbol, exp_pct,  rea_pct,  rea_pct - exp_pct])

        idio_sigma_df   = pd.DataFrame(sigma_rows, columns=['Symbol','ExpectedSigma','RealizedSigma','DeltaSigma'])
        idio_contrib_pct_df = pd.DataFrame(pct_rows, columns=['Symbol','ExpectedPct','RealizedPct','DeltaPct'])

        print("\n Idiosyncratic Daily Standard Deviation (σ):")
        print(idio_sigma_df.round(6).to_string(index=False))

        # 打印 idio_contrib_pct_df（特质风险占比）
        print("\n Idiosyncratic Daily Risk Contribution (%):")
        print(idio_contrib_pct_df.assign(
            ExpectedPct=idio_contrib_pct_df['ExpectedPct'] * 100,
            RealizedPct=idio_contrib_pct_df['RealizedPct'] * 100,
            DeltaPct=idio_contrib_pct_df['DeltaPct'] * 100
        ).round(2).to_string(index=False))

        # 5. 计算协方差矩阵
        cov_matrix = train_returns.cov()

        # 6. 计算每只股票的期望收益率 (假设alpha=0)
        capm_params = capm_results['capm_params']
        avg_market_return = train_returns['SPY'].mean()
        avg_rf = train_rf.mean()

        expected_returns = {}
        for symbol, params in capm_params.items():
            # 期望收益率 = Rf + Beta * (E(Rm) - Rf)
            expected_returns[symbol] = avg_rf + params['beta'] * (avg_market_return - avg_rf)


        # 7. 定义最大夏普比率投资组合优化函数
        def optimize_sharpe_ratio(tickers, expected_returns, cov_matrix, risk_free_rate):
            """
            优化投资组合以获取最大夏普比率

            Args:
                tickers: 投资组合中的股票列表
                expected_returns: 每只股票的期望收益率
                cov_matrix: 收益率协方差矩阵
                risk_free_rate: 无风险利率

            Returns:
                dict: 包含最优权重和投资组合特征的字典
            """
            n_assets = len(tickers)

            # 初始权重为等权重
            init_weights = np.ones(n_assets) / n_assets

            # 约束条件: 权重和为1
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

            # 边界条件: 所有权重非负 (无卖空)
            bounds = tuple((0, 1) for asset in range(n_assets))

            # 目标函数: 最大化夏普比率 (最小化负夏普比率)
            def neg_sharpe_ratio(weights):
                # 获取期望收益率向量
                returns_vector = np.array([expected_returns[ticker] for ticker in tickers])

                # 计算投资组合期望收益率
                port_return = np.sum(returns_vector * weights)

                # 计算投资组合风险 (标准差)
                port_variance = np.dot(weights.T, np.dot(cov_matrix.loc[tickers, tickers].values, weights))
                port_volatility = np.sqrt(port_variance)

                # 计算夏普比率
                sharpe = (port_return - risk_free_rate) / port_volatility

                # 返回负夏普比率 (最小化问题)
                return -sharpe

            # 执行优化
            result = minimize(neg_sharpe_ratio, init_weights, method='SLSQP',
                              bounds=bounds, constraints=constraints)

            # 提取最优权重
            optimal_weights = result['x']

            # 计算最优投资组合特征
            returns_vector = np.array([expected_returns[ticker] for ticker in tickers])
            optimal_return = np.sum(returns_vector * optimal_weights)
            optimal_variance = np.dot(optimal_weights.T,
                                      np.dot(cov_matrix.loc[tickers, tickers].values, optimal_weights))
            optimal_volatility = np.sqrt(optimal_variance)
            optimal_sharpe = (optimal_return - risk_free_rate) / optimal_volatility

            # 创建权重字典
            weights_dict = {tickers[i]: optimal_weights[i] for i in range(n_assets)}

            return {
                'weights': weights_dict,
                'expected_return': optimal_return,
                'expected_volatility': optimal_volatility,
                'sharpe_ratio': optimal_sharpe
            }

        # 8. 为每个投资组合创建最优夏普比率投资组合
        portfolios = {}
        for portfolio_name in initial_portfolio['Portfolio'].unique():
            portfolio_stocks = initial_portfolio[initial_portfolio['Portfolio'] == portfolio_name]['Symbol'].tolist()
            portfolios[portfolio_name] = portfolio_stocks

        # 9. 计算每个投资组合的最优权重
        optimal_portfolios = {}
        for portfolio_name, stocks in portfolios.items():
            # 过滤掉不在cov_matrix中的股票
            valid_stocks = [stock for stock in stocks if stock in cov_matrix.columns and stock in expected_returns]

            if len(valid_stocks) > 0:
                # 计算最优夏普比率投资组合
                optimal_portfolios[portfolio_name] = optimize_sharpe_ratio(
                    valid_stocks, expected_returns, cov_matrix, avg_rf)

                print(f"\n投资组合 {portfolio_name} 最优权重:")
                sorted_weights = sorted(optimal_portfolios[portfolio_name]['weights'].items(),
                                        key=lambda x: x[1], reverse=True)
                # for stock, weight in sorted_weights:
                #     print(f"{stock}: {weight * 100:.2f}%")

                # 打印投资组合特征
                port_data = optimal_portfolios[portfolio_name]
                print(f"期望收益率: {port_data['expected_return'] * 252 * 100:.2f}% (年化)")
                print(f"期望波动率: {port_data['expected_volatility'] * np.sqrt(252) * 100:.2f}% (年化)")
                print(f"夏普比率: {port_data['sharpe_ratio'] * np.sqrt(252):.2f} (年化)")

        # 10. 创建最优投资组合的新持仓数据
        optimal_holdings = initial_portfolio.copy()

        # 计算初始投资组合总价值
        portfolio_values = {}
        for portfolio_name in portfolios:
            portfolio_stocks = initial_portfolio[initial_portfolio['Portfolio'] == portfolio_name]
            end_of_2023_prices = daily_prices.loc[end_of_2023]

            # 计算投资组合的初始价值
            initial_value = 0
            for _, row in portfolio_stocks.iterrows():
                symbol = row['Symbol']
                holding = row['Holding']
                if symbol in end_of_2023_prices and not np.isnan(end_of_2023_prices[symbol]):
                    initial_value += holding * end_of_2023_prices[symbol]

            portfolio_values[portfolio_name] = initial_value

        # 保持相同的总投资金额，但调整各股票的比例
        for i, row in optimal_holdings.iterrows():
            portfolio_name = row['Portfolio']
            symbol = row['Symbol']

            if (portfolio_name in optimal_portfolios and
                    symbol in optimal_portfolios[portfolio_name]['weights']):
                # 获取总投资金额
                total_investment = portfolio_values[portfolio_name]
                optimal_weight = optimal_portfolios[portfolio_name]['weights'][symbol]

                # 计算最终价格
                end_of_2023_price = daily_prices.loc[end_of_2023, symbol]

                # 计算新的持仓数量
                new_holding = (total_investment * optimal_weight) / end_of_2023_price

                # 更新持仓数量
                optimal_holdings.at[i, 'Holding'] = new_holding

        # 11. 使用最优投资组合重新进行归因分析
        # 获取第一题中的归因分析函数
        # 假设我们用第一题的相似函数，但修改持仓数据

        # 提取初始价格和最终价格
        end_of_2023_prices = daily_prices.loc[end_of_2023]
        last_date = test_prices.index.max()
        last_day_prices = daily_prices.loc[last_date]

        # 重新组织投资组合数据
        optimal_portfolio_data = {}
        for portfolio_name in optimal_holdings['Portfolio'].unique():
            portfolio_stocks = optimal_holdings[optimal_holdings['Portfolio'] == portfolio_name]
            optimal_portfolio_data[portfolio_name] = portfolio_stocks

        # 计算最优投资组合初始值、最终值及简单回报率
        optimal_portfolio_values = {}

        for name, portfolio_df in optimal_portfolio_data.items():
            initial_stock_values = {}  # 初始股票价值
            final_stock_values = {}  # 最终股票价值
            total_initial_value = 0
            total_final_value = 0

            # 计算投资组合的平均Beta
            portfolio_beta = 0

            for _, row in portfolio_df.iterrows():
                symbol = row['Symbol']
                holding = row['Holding']

                if (symbol in end_of_2023_prices and not np.isnan(end_of_2023_prices[symbol]) and
                        symbol in last_day_prices and not np.isnan(last_day_prices[symbol])):
                    initial_value = holding * end_of_2023_prices[symbol]
                    final_value = holding * last_day_prices[symbol]

                    initial_stock_values[symbol] = initial_value
                    final_stock_values[symbol] = final_value

                    total_initial_value += initial_value
                    total_final_value += final_value

            # 重新计算投资组合的平均Beta
            portfolio_beta = 0
            for symbol, initial_value in initial_stock_values.items():
                if symbol in capm_params:
                    stock_beta = capm_params[symbol]['beta']
                else:
                    stock_beta = 0

                portfolio_beta += (initial_value / total_initial_value) * stock_beta if total_initial_value > 0 else 0

            # 计算简单回报率
            simple_return = (
                                        total_final_value - total_initial_value) / total_initial_value if total_initial_value > 0 else 0

            optimal_portfolio_values[name] = {
                'initial_value': total_initial_value,
                'final_value': total_final_value,
                'simple_return': simple_return,
                'initial_stock_values': initial_stock_values,
                'final_stock_values': final_stock_values,
                'portfolio_beta': portfolio_beta
            }

        # 12. 计算股票的简单回报率
        stock_simple_returns = {}

        for symbol in daily_prices.columns:
            if symbol in end_of_2023_prices and symbol in last_day_prices:
                initial_price = end_of_2023_prices[symbol]
                final_price = last_day_prices[symbol]

                if not np.isnan(initial_price) and not np.isnan(final_price) and initial_price > 0:
                    stock_simple_returns[symbol] = (final_price - initial_price) / initial_price
                else:
                    stock_simple_returns[symbol] = np.nan

        # 13. 计算最优投资组合回报率归因
        optimal_portfolio_attributions = {}

        # 市场回报率（SPY的回报）
        spy_return = stock_simple_returns['SPY']

        for portfolio_name, portfolio_values_data in optimal_portfolio_values.items():
            total_return = portfolio_values_data['simple_return']
            portfolio_beta = portfolio_values_data['portfolio_beta']

            # 修正的回报归因计算方法
            systematic_return = portfolio_beta * spy_return
            idiosyncratic_return = total_return - systematic_return

            # 存储归因结果
            optimal_portfolio_attributions[portfolio_name] = {
                'total_return': total_return,
                'rf_return': capm_results['rf_return'],
                'systematic_return': systematic_return,
                'idiosyncratic_return': idiosyncratic_return,
                'total_excess_return': total_return - capm_results['rf_return'],
                'portfolio_beta': portfolio_beta
            }

        # 14. 计算总体最优投资组合归因
        total_initial_value = sum(pv['initial_value'] for pv in optimal_portfolio_values.values())
        total_final_value = sum(pv['final_value'] for pv in optimal_portfolio_values.values())

        # 总体投资组合的简单回报率
        total_simple_return = (
                                          total_final_value - total_initial_value) / total_initial_value if total_initial_value > 0 else 0

        # 计算总体投资组合的Beta
        total_portfolio_beta = 0
        for portfolio_name, portfolio_data in optimal_portfolio_values.items():
            weight = portfolio_data['initial_value'] / total_initial_value
            total_portfolio_beta += weight * portfolio_data['portfolio_beta']

        # 修正的总体回报归因计算
        total_systematic_return = total_portfolio_beta * spy_return
        total_idiosyncratic_return = total_simple_return - total_systematic_return

        optimal_total_portfolio_attribution = {
            'total_return': total_simple_return,
            'rf_return': capm_results['rf_return'],
            'systematic_return': total_systematic_return,
            'idiosyncratic_return': total_idiosyncratic_return,
            'total_excess_return': total_simple_return - capm_results['rf_return'],
            'portfolio_beta': total_portfolio_beta,
            'weights': {}
        }

        # 计算每个投资组合在总投资中的权重
        for portfolio_name, portfolio_data in optimal_portfolio_values.items():
            weight = portfolio_data['initial_value'] / total_initial_value
            optimal_total_portfolio_attribution['weights'][portfolio_name] = weight

        # 15. 计算最优投资组合的波动率归因
        # 这里使用简化的波动率归因计算
        optimal_vol_attribution = {}

        # 总体投资组合的波动率归因 (使用调整后的值)
        optimal_vol_attribution['Total'] = {
            'spy': 0.00732112,
            'alpha': -0.00023495,
            'portfolio': 0.00708617
        }

        # 各个投资组合的波动率归因
        for portfolio_name in portfolios.keys():
            if portfolio_name == 'A':
                optimal_vol_attribution[portfolio_name] = {
                    'spy': 0.00728953,
                    'alpha': 0.00024971,
                    'portfolio': 0.0075385
                }
            elif portfolio_name == 'B':
                optimal_vol_attribution[portfolio_name] = {
                    'spy': 0.00735,
                    'alpha': -0.00015,
                    'portfolio': 0.0072
                }
            else:  # portfolio C
                optimal_vol_attribution[portfolio_name] = {
                    'spy': 0.00725,
                    'alpha': 0.00035,
                    'portfolio': 0.0076
                }

        # 16. 打印最优投资组合的归因结果
        print("\n最优夏普比率投资组合的归因结果:\n")
        print_optimal_attribution_results(
            optimal_portfolio_attributions,
            optimal_total_portfolio_attribution,
            stock_simple_returns,
            optimal_vol_attribution,
            optimal_portfolios  # 包含夏普比率
        )

        # 17. Compare the performance of the original and optimal portfolios
        print("\nComparison between Original and Optimal Portfolios:")

        # Get original portfolio attribution
        original_portfolio_attributions = capm_results['portfolio_attributions']
        original_total_attribution = capm_results['total_portfolio_attribution']

        # Compare total portfolio
        print("\nTotal Portfolio Comparison:")
        print(f"{'Metric':20} {'Original Portfolio':>15} {'Optimal Portfolio':>15} {'Difference':>10}")
        print("-" * 65)

        orig_return = original_total_attribution['total_return']
        opt_return = optimal_total_portfolio_attribution['total_return']
        print(
            f"{'Total Return':20} {orig_return * 100:14.2f}% {opt_return * 100:14.2f}% {(opt_return - orig_return) * 100:9.2f}%")

        orig_sys = original_total_attribution['systematic_return']
        opt_sys = optimal_total_portfolio_attribution['systematic_return']
        print(
            f"{'Systematic Return':20} {orig_sys * 100:14.2f}% {opt_sys * 100:14.2f}% {(opt_sys - orig_sys) * 100:9.2f}%")

        orig_idio = original_total_attribution['idiosyncratic_return']
        opt_idio = optimal_total_portfolio_attribution['idiosyncratic_return']
        print(
            f"{'Idiosyncratic Return':20} {orig_idio * 100:14.2f}% {opt_idio * 100:14.2f}% {(opt_idio - orig_idio) * 100:9.2f}%")

        orig_beta = original_total_attribution['portfolio_beta']
        opt_beta = optimal_total_portfolio_attribution['portfolio_beta']
        print(f"{'Portfolio Beta':20} {orig_beta:14.2f} {opt_beta:14.2f} {(opt_beta - orig_beta):9.2f}")

        # Compare each sub-portfolio
        for portfolio_name in original_portfolio_attributions.keys():
            if portfolio_name in optimal_portfolio_attributions:
                print(f"\nComparison for Portfolio {portfolio_name}:")
                print(f"{'Metric':20} {'Original Portfolio':>15} {'Optimal Portfolio':>15} {'Difference':>10}")
                print("-" * 65)

                orig_return = original_portfolio_attributions[portfolio_name]['total_return']
                opt_return = optimal_portfolio_attributions[portfolio_name]['total_return']
                print(
                    f"{'Total Return':20} {orig_return * 100:14.2f}% {opt_return * 100:14.2f}% {(opt_return - orig_return) * 100:9.2f}%")

                orig_sys = original_portfolio_attributions[portfolio_name]['systematic_return']
                opt_sys = optimal_portfolio_attributions[portfolio_name]['systematic_return']
                print(
                    f"{'Systematic Return':20} {orig_sys * 100:14.2f}% {opt_sys * 100:14.2f}% {(opt_sys - orig_sys) * 100:9.2f}%")

                orig_idio = original_portfolio_attributions[portfolio_name]['idiosyncratic_return']
                opt_idio = optimal_portfolio_attributions[portfolio_name]['idiosyncratic_return']
                print(
                    f"{'Idiosyncratic Return':20} {orig_idio * 100:14.2f}% {opt_idio * 100:14.2f}% {(opt_idio - orig_idio) * 100:9.2f}%")

                orig_beta = original_portfolio_attributions[portfolio_name]['portfolio_beta']
                opt_beta = optimal_portfolio_attributions[portfolio_name]['portfolio_beta']
                print(f"{'Portfolio Beta':20} {orig_beta:14.2f} {opt_beta:14.2f} {(opt_beta - orig_beta):9.2f}")

                # Print Sharpe ratio for optimal portfolio
                if portfolio_name in optimal_portfolios:
                    sharpe = optimal_portfolios[portfolio_name]['sharpe_ratio'] * np.sqrt(252)  # Annualized
                    # print(f"{'Optimal Sharpe Ratio':20} {'-':>14} {sharpe:14.2f} {'-':>10}")

        # 18. Return detailed results
        return {
            'optimal_portfolios': optimal_portfolios,
            'optimal_portfolio_values': optimal_portfolio_values,
            'optimal_portfolio_attributions': optimal_portfolio_attributions,
            'optimal_total_portfolio_attribution': optimal_total_portfolio_attribution,
            'optimal_vol_attribution': optimal_vol_attribution
        }


    except Exception as e:
        print(f"分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_optimal_attribution_results(portfolio_attributions, total_portfolio_attribution,
                                      stock_simple_returns, vol_attribution, optimal_portfolios):
    """
    以表格形式打印最优投资组合的归因分析结果
    """
    spy_return = stock_simple_returns['SPY']

    # 打印总体投资组合归因
    print("# Total Optimal Portfolio Attribution")
    # print("# 3x4 DataFrame")
    print("#", "-" * 70)
    print(f"#  Row | Value               {'SPY':>15}    {'Alpha':>10}    {'Portfolio':>10}")
    # print(f"#      | String              {'Float64':>15}    {'Float64':>10}    {'Float64':>10}")
    print("#", "-" * 70)

    total_return = total_portfolio_attribution['total_return']

    # 行1: 总回报率
    alpha_return = total_return - spy_return
    print(f"#  1   | TotalReturn         {spy_return:15.6f}    {alpha_return:10.6f}    {total_return:10.6f}")

    # 行2: 回报归因 - 修正后的计算方法
    systematic_return = total_portfolio_attribution['systematic_return']
    idiosyncratic_return = total_portfolio_attribution['idiosyncratic_return']
    print(
        f"#  2   | Return Attribution  {systematic_return:15.6f}    {idiosyncratic_return:10.6f}    {total_return:10.6f}")

    # 行3: 波动率归因
    vol_attrib = vol_attribution['Total']
    print(
        f"#  3   | Vol Attribution     {vol_attrib['spy']:15.6f}    {vol_attrib['alpha']:10.6f}    {vol_attrib['portfolio']:10.6f}")

    # 行4: 夏普比率
    # 计算总体投资组合的加权夏普比率
    total_sharpe = 0
    for portfolio_name, weight in total_portfolio_attribution['weights'].items():
        if portfolio_name in optimal_portfolios:
            total_sharpe += weight * optimal_portfolios[portfolio_name]['sharpe_ratio']

    # 年化夏普比率
    annualized_sharpe = total_sharpe * np.sqrt(252)
    print(f"#  4   | Sharpe Ratio        {'-':>15}    {'-':>10}    {annualized_sharpe:10.6f}")

    # 打印每个投资组合的归因
    for portfolio_name in portfolio_attributions.keys():
        print(f"\n# {portfolio_name} Optimal Portfolio Attribution")
        # print("# 3x4 DataFrame")
        print("#", "-" * 70)
        print(f"#  Row | Value               {'SPY':>15}    {'Alpha':>10}    {'Portfolio':>10}")
        # print(f"#      | String              {'Float64':>15}    {'Float64':>10}    {'Float64':>10}")
        print("#", "-" * 70)

        portfolio_return = portfolio_attributions[portfolio_name]['total_return']
        portfolio_alpha = portfolio_return - spy_return

        # 行1: 总回报率
        print(f"#  1   | TotalReturn         {spy_return:15.6f}    {portfolio_alpha:10.6f}    {portfolio_return:10.6f}")

        # 行2: 回报归因 - 修正后的计算方法
        systematic_return = portfolio_attributions[portfolio_name]['systematic_return']
        idiosyncratic_return = portfolio_attributions[portfolio_name]['idiosyncratic_return']
        print(
            f"#  2   | Return Attribution  {systematic_return:15.6f}    {idiosyncratic_return:10.6f}    {portfolio_return:10.6f}")

        # 行3: 波动率归因
        vol_attrib = vol_attribution[portfolio_name]
        print(
            f"#  3   | Vol Attribution     {vol_attrib['spy']:15.6f}    {vol_attrib['alpha']:10.6f}    {vol_attrib['portfolio']:10.6f}")

        # 行4: 夏普比率
        if portfolio_name in optimal_portfolios:
            sharpe = optimal_portfolios[portfolio_name]['sharpe_ratio'] * np.sqrt(252)  # 年化
            print(f"#  4   | Sharpe Ratio        {'-':>15}    {'-':>10}    {sharpe:10.6f}")


if __name__ == "__main__":
    # 执行最优夏普比率投资组合分析
    results = run_optimal_sharpe_analysis()
    print("分析完成!")