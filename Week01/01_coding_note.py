import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, kurtosis, ttest_1samp

# PDF Example
x = np.arange(-5, 5.01, 0.01)
pdf = norm.pdf(x, loc=0, scale=1)
df = pd.DataFrame({'x': x, 'pdf': pdf})

print(df.head())

plt.plot(df['x'], df['pdf'], label='PDF')
plt.legend()
plt.savefig("pdf.png")
plt.close()

# CDF
cdf = norm.cdf(x, loc=0, scale=1)
df['cdf'] = cdf

plt.plot(df['x'], df['cdf'], label='CDF')
plt.legend()
plt.savefig("cdf.png")
plt.close()

# Quick and dirty integration of the PDF
approx_cdf = np.sum(df['pdf'] * 0.01)
print(f"CDF actual {df['cdf'].iloc[-1]} vs calculated {approx_cdf} for F_x({df['x'].iloc[-1]})")

# Calculation of moments
n = 1000
sim = norm.rvs(loc=0, scale=1, size=n)

def first_4_moments(sample):
    n = len(sample)
    μ_hat = np.mean(sample)
    sim_corrected = sample - μ_hat
    cm2 = np.mean(sim_corrected**2)
    σ2_hat = np.var(sample, ddof=1)
    skew_hat = np.mean(sim_corrected**3) / (cm2**1.5)
    kurt_hat = np.mean(sim_corrected**4) / (cm2**2)
    excess_kurt_hat = kurt_hat - 3
    return μ_hat, σ2_hat, skew_hat, excess_kurt_hat

m, s2, sk, k = first_4_moments(sim)

print(f"Mean {m} ({np.mean(sim)})")
print(f"Variance {s2} ({np.var(sim, ddof=1)})")
print(f"Skew {sk} ({skew(sim)})")
print(f"Kurtosis {k} ({kurtosis(sim, fisher=True)})")

print(f"Mean diff = {m - np.mean(sim)}")
print(f"Variance diff = {s2 - np.var(sim, ddof=1)}")
print(f"Skewness diff = {sk - skew(sim)}")
print(f"Kurtosis diff = {k - kurtosis(sim, fisher=True)}")

# Study the limiting expected values from the estimators
sample_size = 1000
samples = 100

means, vars_, skews, kurts = [], [], [], []

for _ in range(samples):
    sample = norm.rvs(loc=0, scale=1, size=sample_size)
    μ, σ2, sk, k = first_4_moments(sample)
    means.append(μ)
    vars_.append(σ2)
    skews.append(sk)
    kurts.append(k)

print(f"Mean versus Expected {np.mean(means) - 0}")
print(f"Variance versus Expected {np.mean(vars_) - 1}")
print(f"Skewness versus Expected {np.mean(skews) - 0}")
print(f"Kurtosis versus Expected {np.mean(kurts) - 0}")

# Test the kurtosis function for bias in small sample sizes
sample_size = 100
samples = 100
kurts = []

for _ in range(samples):
    sample = norm.rvs(loc=0, scale=1, size=sample_size)
    kurts.append(kurtosis(sample, fisher=True))

kurts = np.array(kurts)
print(f"Kurtosis summary: mean={np.mean(kurts)}, variance={np.var(kurts)}")

# Hypothesis test using scipy
t_stat, p_val = ttest_1samp(kurts, popmean=0.0)
print(f"T-test p-value: {p_val}")


