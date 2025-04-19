# Fix for the distribution fitting function
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
        if t_available:
            t_params = stats.t.fit(clean_returns)
            log_likelihood = np.sum(stats.t.logpdf(clean_returns, *t_params))
            aic = calculate_aic(log_likelihood, 3)  # 3 parameters: df, loc, scale
            result['GeneralizedT'] = {
                'params': t_params,
                'aic': aic,
                'dist': stats.t(*t_params)
            }
        else:
            gt = GeneralizedT()
            nu, mu, sigma = gt.fit(clean_returns)
            gt_fitted = GeneralizedT(nu, mu, sigma)
            log_likelihood = np.sum(np.log(gt_fitted.pdf(clean_returns)))
            aic = calculate_aic(log_likelihood, 3)  # 3 parameters: nu, mu, sigma
            result['GeneralizedT'] = {
                'params': (nu, mu, sigma),
                'aic': aic,
                'dist': gt_fitted
            }
    except Exception as e:
        print(f"Error fitting GeneralizedT: {e}")
        result['GeneralizedT'] = {'aic': np.inf}

    # 3. Normal Inverse Gaussian
    try:
        if nig_available:
            from statsmodels.sandbox.distributions.extras import NormalInverseGaussian
            nig = NormalInverseGaussian()
            nig_params = nig.fit(clean_returns)
            nig_fitted = NormalInverseGaussian(*nig_params)
            log_likelihood = np.sum(np.log(nig_fitted.pdf(clean_returns)))
            aic = calculate_aic(log_likelihood, 4)  # 4 parameters: alpha, beta, mu, delta
        else:
            nig = NormalInverseGaussian(1, 0, 0, 1)  # Default initialization
            alpha, beta, mu, delta = nig.fit(clean_returns)
            nig_fitted = NormalInverseGaussian(alpha, beta, mu, delta)
            log_likelihood = np.sum(np.log(nig_fitted.pdf(clean_returns)))
            aic = calculate_aic(log_likelihood, 4)  # 4 parameters: alpha, beta, mu, delta

        result['NIG'] = {
            'params': (alpha, beta, mu, delta),
            'aic': aic,
            'dist': nig_fitted
        }
    except Exception as e:
        print(f"Error fitting NIG: {e}")
        result['NIG'] = {'aic': np.inf}

    # 4. Skew Normal
    try:
        if skewnorm_available:
            skewnorm_params = stats.skewnorm.fit(clean_returns)
            log_likelihood = np.sum(stats.skewnorm.logpdf(clean_returns, *skewnorm_params))
            aic = calculate_aic(log_likelihood, 3)  # 3 parameters: a, loc, scale
            result['SkewNormal'] = {
                'params': skewnorm_params,
                'aic': aic,
                'dist': stats.skewnorm(*skewnorm_params)
            }
        else:
            skewnorm = SkewNormal()
            a, loc, scale = skewnorm.fit(clean_returns)
            skewnorm_fitted = SkewNormal(a, loc, scale)
            log_likelihood = np.sum(np.log(skewnorm_fitted.pdf(clean_returns)))
            aic = calculate_aic(log_likelihood, 3)  # 3 parameters: a, loc, scale
            result['SkewNormal'] = {
                'params': (a, loc, scale),
                'aic': aic,
                'dist': skewnorm_fitted
            }
    except Exception as e:
        print(f"Error fitting SkewNormal: {e}")
        result['SkewNormal'] = {'aic': np.inf}

    # Find best model based on AIC
    best_model = min(result.items(), key=lambda x: x[1]['aic'])[0]

    return result, best_model

# Fix for the portfolio weights section with error handling
portfolio_weight_data = {}
for portfolio in portfolios:
    weights = {}
    # Check if portfolio exists in portfolio_weights
    if portfolio in portfolio_weights:
        for symbol, info in portfolio_weights[portfolio].items():
            weights[symbol] = info['weight']
        portfolio_weight_data[portfolio] = weights
    else:
        print(f"Warning: Portfolio '{portfolio}' not found in portfolio_weights dictionary")
        # You might want to handle this case - either skip or provide default weights

# The rest of your calculation code can then use portfolio_weight_data