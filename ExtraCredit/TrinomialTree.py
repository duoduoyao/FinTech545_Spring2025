import numpy as np
import time

from scipy.stats import norm


class TrinomialTree:
    def __init__(self, S0, K, T, r, sigma, N, is_american=False, dividend=0, option_type='call'):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.N = N
        self.is_american = is_american
        self.dividend = dividend
        self.option_type = option_type.lower()
        self.dt = T / N
        self.adjusted_r = r - dividend  # Adjust for continuous dividend

        # Trinomial tree parameters
        self.u = np.exp(sigma * np.sqrt(2 * self.dt))
        self.d = 1 / self.u
        self.m = 1.0

        # Probabilities
        numerator_pu = (np.exp(self.adjusted_r * self.dt / 2) - np.exp(-sigma * np.sqrt(self.dt / 2)))
        denominator = (np.exp(sigma * np.sqrt(self.dt / 2)) - np.exp(-sigma * np.sqrt(self.dt / 2)))
        self.pu = (numerator_pu / denominator) ** 2
        self.pd = ((np.exp(sigma * np.sqrt(self.dt / 2)) - np.exp(self.adjusted_r * self.dt / 2)) / denominator) ** 2
        self.pm = 1 - self.pu - self.pd

        # Validate probabilities
        if self.pu < 0 or self.pd < 0 or self.pm < 0:
            raise ValueError("Negative probabilities detected. Adjust parameters or increase N.")

    def build_stock_tree(self):
        stock_tree = [[round(self.S0, 10)]]
        for t in range(1, self.N + 1):
            prev_level = stock_tree[t - 1]
            current_level = []
            for s in prev_level:
                su = round(s * self.u, 10)
                sm = round(s * self.m, 10)
                sd = round(s * self.d, 10)
                current_level.extend([su, sm, sd])
            # Remove duplicates and sort
            current_level = sorted(list(set(current_level)), reverse=True)
            stock_tree.append(current_level)
        return stock_tree

    def calculate_option_price(self, stock_tree):
        option_tree = [[0 for _ in level] for level in stock_tree]

        # Terminal payoff
        for i, s in enumerate(stock_tree[-1]):
            if self.option_type == 'call':
                option_tree[-1][i] = max(s - self.K, 0)
            else:
                option_tree[-1][i] = max(self.K - s, 0)

        # Backward induction
        for t in range(self.N - 1, -1, -1):
            for i, s in enumerate(stock_tree[t]):
                next_level = np.array(stock_tree[t + 1])

                # Calculate next prices
                su = round(s * self.u, 10)
                sm = round(s * self.m, 10)
                sd = round(s * self.d, 10)

                # Find indices of child nodes
                idx_u = np.argmin(np.abs(next_level - su))
                idx_m = np.argmin(np.abs(next_level - sm))
                idx_d = np.argmin(np.abs(next_level - sd))

                # Expected value
                expected = (self.pu * option_tree[t + 1][idx_u] +
                            self.pm * option_tree[t + 1][idx_m] +
                            self.pd * option_tree[t + 1][idx_d]) * np.exp(-self.r * self.dt)

                # American option check
                if self.is_american:
                    intrinsic = max(s - self.K, 0) if self.option_type == 'call' else max(self.K - s, 0)
                    option_tree[t][i] = max(expected, intrinsic)
                else:
                    option_tree[t][i] = expected

        return option_tree[0][0]

def black_scholes(S0, K, T, r, sigma, dividend=0, option_type='call'):
    d1 = (np.log(S0 / K) + (r - dividend + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S0 * np.exp(-dividend * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-dividend * T) * norm.cdf(-d1)
    return price


if __name__ == "__main__":
    # Parameters
    S0 = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.2
    dividend = 0.03


    # European Call with Dividend
    print("European Call with Dividend:")
    bs_price = black_scholes(S0, K, T, r, sigma, dividend=dividend, option_type='call')
    print(f"Black-Scholes Price: {bs_price:.4f}")
    for N in [10, 50, 100, 200, 300]:
        start = time.time()
        model = TrinomialTree(S0, K, T, r, sigma, N, dividend=dividend, option_type='call')
        stock_tree = model.build_stock_tree()
        price = model.calculate_option_price(stock_tree)
        print(f"N={N}: Price={price:.4f}, Time={time.time() - start:.2f}s")

    # European Put with Dividend
    print("\nEuropean Put with Dividend:")
    bs_price = black_scholes(S0, K, T, r, sigma, dividend=dividend, option_type='put')
    print(f"Black-Scholes Price: {bs_price:.4f}")
    for N in [10, 50, 100, 200, 300]:
        start = time.time()
        model = TrinomialTree(S0, K, T, r, sigma, N, dividend=dividend, option_type='put')
        stock_tree = model.build_stock_tree()
        price = model.calculate_option_price(stock_tree)
        print(f"N={N}: Price={price:.4f}, Time={time.time() - start:.2f}s")

    # American Call with Dividend
    print("\nAmerican Call with Dividend:")
    for N in [10, 50, 100, 200, 300]:
        start = time.time()
        model = TrinomialTree(S0, K, T, r, sigma, N, is_american=True, dividend=dividend, option_type='call')
        stock_tree = model.build_stock_tree()
        price = model.calculate_option_price(stock_tree)
        print(f"N={N}: Price={price:.4f}, Time={time.time() - start:.2f}s")

    # American Put with Dividend
    print("\nAmerican Put with Dividend:")
    for N in [10, 50, 100, 200, 300]:
        start = time.time()
        model = TrinomialTree(S0, K, T, r, sigma, N, is_american=True, dividend=dividend, option_type='put')
        stock_tree = model.build_stock_tree()
        price = model.calculate_option_price(stock_tree)
        print(f"N={N}: Price={price:.4f}, Time={time.time() - start:.2f}s")


    # European Call without Dividend
    print("\nEuropean Call without Dividend：")
    bs_price = black_scholes(S0, K, T, r, sigma, dividend=0, option_type='call')
    print(f"Black-Scholes Price: {bs_price:.4f}")
    for N in [10, 50, 100, 200, 300]:
        start = time.time()
        model = TrinomialTree(S0, K, T, r, sigma, N, option_type='call')
        stock_tree = model.build_stock_tree()
        price = model.calculate_option_price(stock_tree)
        print(f"N={N}: Price={price:.4f}, Time={time.time() - start:.2f}s")

    # European Put without Dividend
    print("\nEuropean Put without Dividend:")
    bs_price = black_scholes(S0, K, T, r, sigma, dividend=0, option_type='put')
    print(f"Black-Scholes Price: {bs_price:.4f}")
    for N in [10, 50, 100, 200, 300]:
        start = time.time()
        model = TrinomialTree(S0, K, T, r, sigma, N, option_type='put')
        stock_tree = model.build_stock_tree()
        price = model.calculate_option_price(stock_tree)
        print(f"N={N}: Price={price:.4f}, Time={time.time() - start:.2f}s")

    # American Call with Dividend
    print("\nAmerican Call without Dividend:")
    for N in [10, 50, 100, 200, 300]:
        start = time.time()
        model = TrinomialTree(S0, K, T, r, sigma, N, is_american=True, option_type='call')
        stock_tree = model.build_stock_tree()
        price = model.calculate_option_price(stock_tree)
        print(f"N={N}: Price={price:.4f}, Time={time.time() - start:.2f}s")

    # American Put with Dividend
    print("\nAmerican Put without Dividend:")
    for N in [10, 50, 100, 200, 300]:
        start = time.time()
        model = TrinomialTree(S0, K, T, r, sigma, N, is_american=True, option_type='put')
        stock_tree = model.build_stock_tree()
        price = model.calculate_option_price(stock_tree)
        print(f"N={N}: Price={price:.4f}, Time={time.time() - start:.2f}s")