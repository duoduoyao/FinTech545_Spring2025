using BenchmarkTools
using Distributions
using Random
using StatsBase
using DataFrames
using Plots
using StatsPlots
using LinearAlgebra
using JuMP
using Ipopt
using Dates
using ForwardDiff
using FiniteDiff
using CSV
using LoopVectorization
using Roots
using QuadGK
using StateSpaceModels

include("../../library/RiskStats.jl")
include("../../library/simulate.jl")
include("../../library/return_calculate.jl")
include("../../library/fitted_model.jl")
include("../../library/missing_cov.jl")
include("../../library/ewCov.jl")
include("../../library/expost_factor.jl")
include("../../library/optimizers.jl")
include("../../library/bt_american.jl")
include("../../library/gbsm.jl")

theme(:dark)

#1
#a. calculate mean, variance, skewness, kurtosis of the data
#b. given a choice between a normal distribution and a t-distribution, 
# which one would you choose to model the data? Why?
#c. fit both disitributions and prove or disprove your choice in b.

#Data Generation:
Random.seed!(123)
d = TDist(200)*.1 + .05
data = rand(d, 1000)
CSV.write("Project01/problem1.csv", DataFrame(:X=>data))

#Solution
data = CSV.read("Project01/problem1.csv",DataFrame)[!,:X]
#a
mean_data = mean(data)
var_data = var(data)
skew_data = skewness(data)
kurt_data = kurtosis(data)
println("Mean: ", mean_data)
println("Variance: ", var_data)
println("Skewness: ", skew_data)
println("Kurtosis: ", kurt_data)
# Mean: 0.050197957904769164
# Variance: 0.010332476407479594
# Skewness: 0.12044471191944014
# Kurtosis: 0.22292706745038604

#b Hard to argue either way. Given the Kurtosis is >0, I would choose the t-distribution to model the data.  However the Kurtosis is low enough the normal distribution could be a better fit.

#c 
normal = fit_normal(data)
t = fit_general_t(data)
n_aicc = AICC(normal.errorModel, data,2)
t_aicc = AICC(t.errorModel, data,3)
println("Normal AICC: ", n_aicc)
println("T AICC: ", t_aicc)
if n_aicc < t_aicc
    println("Normal Distribution is better")
else
    println("T-Distribution is better")
end
# Normal AICC: -1731.574192394598
# T AICC: -1731.3942725340037
# Normal Distribution is better

#2
#a. Calculate the pairwise covariance matrix of the data.
#b. Is the matrix as least positive semi-definite?  Why?
#c. If not, find the nearest positive semi-definite matrix using Higham's method and the near-psd method of Rebenato and Jackel.
#d. Calculate the covariance matrix of the data using only overlapping data.
#e. Compare the results of the covariance matrices in c and d.  Explain the differences.  Note: the true covariance matrix that
#   generated the data is has 1 on the diagonal and 0.99 elsewhere.

#Data Generation:
function generate_with_missing(n,m; pmiss=.25)
    x = Array{Union{Missing,Float64},2}(undef,n,m)

    c = fill(0.99,(m,m)) + .01*I(m)
    r = rand(MvNormal(fill(0,m),c),n)'
    cr = cor(r)
    println(Matrix(cr))
    for i in 1:n, j in 1:m
        if rand() >= pmiss
            x[i,j] = r[i,j]
        end
    end
    return x
end

Random.seed!(5)
x = generate_with_missing(50,5,pmiss=.25)
CSV.write("Project01/problem2.csv", DataFrame(x,:auto))

#a.
data=CSV.read("Project01/problem2.csv",DataFrame) |> Matrix
c = missing_cov(x; skipMiss=false, fun=cov)
# 5×5 Matrix{Float64}:
#  1.47048   1.45421   0.877269  1.90323  1.44436
#  1.45421   1.25208   0.539548  1.62192  1.23788
#  0.877269  0.539548  1.27242   1.17196  1.09191
#  1.90323   1.62192   1.17196   1.81447  1.58973
#  1.44436   1.23788   1.09191   1.58973  1.39619

#b
c2 = cov2cor(c)
println(min(eigvals(c2)...))
# -0.09482978874911373
# The matrix is not positive semi-definite as the smallest eigenvalue is negative.

#c
ch = higham_nearestPSD(c)
# 5×5 Matrix{Float64}:
#  1.47048   1.33236   0.884378  1.6276   1.39956
#  1.33236   1.25208   0.619028  1.4506   1.21445
#  0.884378  0.619028  1.27242   1.07685  1.05966
#  1.6276    1.4506    1.07685   1.81447  1.57793
#  1.39956   1.21445   1.05966   1.57793  1.39619

cnp = near_psd(c)
# 5×5 Matrix{Float64}:
#  1.47048   1.32701   0.842583  1.62446  1.36483
#  1.32701   1.25208   0.555421  1.43311  1.16591
#  0.842583  0.555421  1.27242   1.05279  1.06042
#  1.62446   1.43311   1.05279   1.81447  1.54499
#  1.36483   1.16591   1.06042   1.54499  1.39619

#d
csk = missing_cov(x; skipMiss=true, fun=cov)
# 5×5 Matrix{Float64}:
#  0.418604  0.394054  0.424457  0.416382  0.434287
#  0.394054  0.396786  0.409343  0.398401  0.422631
#  0.424457  0.409343  0.44136   0.428441  0.448957
#  0.416382  0.398401  0.428441  0.437274  0.440167
#  0.434287  0.422631  0.448957  0.440167  0.466272

#e
# The first thing we notice is the scale of the two matrices is different. The variance of the first matrix is much larger than the second.
fullSampleVar1 = var(skipmissing(x[:,1]))
d = ch[1,1] - fullSampleVar1
# 1.4704843700431658
# -2.220446049250313e-16
# the matrix using the overlapping samples contains the full sample variance in the diagonal.  The first difference between the two matrices 
#is the difference between the full sample variance and the variance of the overlapping samples.

crh = cov2cor(ch)
# 1.0       0.98192   0.646534  0.996422  0.976761
# 0.98192   1.0       0.490432  0.962406  0.918528
# 0.646534  0.490432  1.0       0.708701  0.79502
# 0.996422  0.962406  0.708701  1.0       0.991381
# 0.976761  0.918528  0.79502   0.991381  1.0
crnp = cov2cor(cnp)
# 1.0       0.977976  0.61598   0.994501  0.952527
# 0.977976  1.0       0.440038  0.950799  0.881812
# 0.61598   0.440038  1.0       0.692868  0.795595
# 0.994501  0.950799  0.692868  1.0       0.970688
# 0.952527  0.881812  0.795595  0.970688  1.0
crc = cov2cor(c)
# pairwise correlation matrix
# 1.0       1.0       0.641337  1.0       1.0
# 1.0       1.0       0.427463  1.0       0.936246
# 0.641337  0.427463  1.0       0.771297  0.819219
# 1.0       1.0       0.771297  1.0       0.998795
# 1.0       0.936246  0.819219  0.998795  1.0
crsk = cov2cor(csk)
# Overlapping sample correlation matrix
# 1.0       0.966888  0.987498  0.973227  0.983004
# 0.966888  1.0       0.978168  0.956458  0.982569
# 0.987498  0.978168  1.0       0.975255  0.989666
# 0.973227  0.956458  0.975255  1.0       0.974813
# 0.983004  0.982569  0.989666  0.974813  1.0

# The correlations from Higham and near_psd are much less stable knowing that the true correlation is 0.99.  The correlation matrix from the overlapping samples is much more stable.
# The large outliers in the correlation from the pairwise covariance matrix lead us to wonder why.  Consider the math:
# pairwise_cov[i,j] = sum( (x[k,i] - mean(x[S,i]))*(x[k,j] - mean(x[S,j])) )/(n-1)
#  for k ∈ S
#  where S is the set of rows where x[k,i] and x[k,j] are not missing.
#  and n = length(S)

# This means that the variance calculated on the diagonal is the full sample variance, but the covariance is calculated on a subset of the data.  This leads to a much larger variance in 
# the pairwise correlations when we use the pairwise covariance.

# If we change the covariance function to the correlation function, the pairwise correlation matrix is much more stable and closer to expected:
c = missing_cov(x; skipMiss=false, fun=cor)
# 1.0       0.993266  0.989351  0.994665  0.994846
# 0.993266  1.0       0.97822   0.992945  0.992887
# 0.989351  0.97822   1.0       0.994024  0.991846
# 0.994665  0.992945  0.994024  1.0       0.993074
# 0.994846  0.992887  0.991846  0.993074  1.0

min(eigvals(c)...)
# -0.00031924355522938046
# That matrix is still not psd so fix it.
ch2 = higham_nearestPSD(c)

cTrue = fill(0.99,(5,5)) + .01*I(5)
println("Norm Fixed Pairwise: $(sum( (cTrue - ch2).^2))")
println("Norm Hignam  : $(sum( (cTrue - crh).^2))")
println("Norm NearPSD : $(sum( (cTrue - crnp).^2))")
# Norm Fixed Pairwise: 0.0004735407389425922
# Norm Hignam  : 0.9816743826053165
# Norm NearPSD : 1.1672263871400355

# The fixed pairwise correlation matrix is much closer to the true correlation matrix than the Higham or near_psd methods.  
# In practice, I would use the pairwise correlation matrix, adjust to PSD if needed, and overlay the full sample variance to construct the covariance matrix.

#3
# a. Fit a multivariate normal to the data in problem3.csv
# b. Given that fit, what is the distribution of X2 given X1=0.6?  Use the 2 methods described in class.
# c. Given the properties of the Cholesky root, create a simulation that proves your distribution of X2 given X1=0.6 is correct.

Random.seed!(3)
r = rand(MvNormal([0.05,0.1],[0.01 0.005; 0.005 0.02]),1000)'
CSV.write("Project01/problem3.csv", DataFrame(r,:auto))

#a. 
data = CSV.read("Project01/problem3.csv",DataFrame) |> Matrix
means = mean(data,dims=1)
covar = cov(data)
# Means: 0.0460016  0.099915
# Covariance:  
# 0.0101622   0.00492354
# 0.00492354  0.0202844

#b. conditional expectation
m_ce = means[2] + covar[2,1]/covar[1,1]*(0.6 - means[1])
std_ce = sqrt(covar[2,2] - covar[2,1]^2/covar[1,1])
println("Conditional Expectation: N($(m_ce),$(std_ce))")
# Conditional Expectation: N(0.36832499586097733,0.133787030930085)

# OLS 
X = hcat(ones(size(data,1)),data[:,1])
b = inv(X'X)*X'data[:,2]

println("OLS: N($(b[1] + b[2]*0.6),$(std(data[:,2] - X*b)))")
# OLS: N(0.36832499586097733,0.133787030930085)

#c.
# The Cholesky root of the covariance matrix is the lower triangular matrix L such that LL' = Σ
L = cholesky(covar).L
# 2×2 LowerTriangular{Float64, Matrix{Float64}}:
# 0.100808    ⋅
# 0.0488409  0.133787

# X = L*Z + μ
# Given X1 = 0.6, we can simulate X2 as follows:
# L[1,1]*z1 + L[2,1]*z2 + means[1] = 0.6 -> z1 = (0.6-means[1])/L[1,1]
# X2 = L[2,1]*0.6/L[1,1] + L[2,2]*z2 + means[2]

z2 = rand(Normal(0,1),10000)
z1 = (0.6-means[1])/L[1,1]
x2 = L[2,1]*z1 .+ L[2,2]*z2 .+ means[2]

println("Simulation: N($(mean(x2)), $(std(x2)))")
# Simulation: N(0.36651407623261295, 0.13239991320968955)

# Alternatively, we know the limiting distribution because the simulation is a linear combination of normals.
m_lim =L[2,1]*z1 + means[2]
std_lim = L[2,2]
println("Limiting Distribution: N($(m_lim),$(std_lim))")
# Limiting Distribution: N(0.36832499586097744,0.133787030930085)
# The simulation is consistent with the limiting distribution, which is the same as the conditional expectation and OLS estimate.

#4
#a. Simulate an MA(1), MA(2), and MA(3) process and graph the ACF and PACF of each.  What do you notice?
#b. Simulate an AR(1), AR(2), and AR(3) process and graph the ACF and PACF of each.  What do you notice?
#c. Examine the data in problem4.csv.  What AR/MA process would you use to model the data?  Why?
#d. Fit the model you chose in c along with other AR/MA models.  Compare the AICc of each model.  What is the best model?

#a. AR models have PACF that cuts off after the order of the AR process.  The absolute value of the ACF is non-zero and decays towards 0.
#b. MA models have ACF that cuts off after the order of the MA process.  The absolute value of the PACF is non-zero and decays towards 0.

#Data Generation
Random.seed!(444)

function simulate_ar_process(n, θ1, θ2, θ3; burnin=100)
    # ε = randn(n + burnin)
    # y = zeros(n)
    # for t in (burnin+1):n+burnin
    #     y[t-3] = ε[t] + θ1 * ε[t-1] + θ3 * ε[t-3]
    # end

    y = Vector{Float64}(undef,n)

    yt_last = fill(1.0,3)
    d = Normal(0,0.1)
    ε = rand(d,n+burnin)
    
    for i in 1:(n+burnin)
        y_t = 1.0 + θ1*yt_last[1] +θ2*yt_last[2] + θ3*yt_last[3] + ε[i]
        yt_last = [y_t, yt_last[1], yt_last[2]]
        if i > burnin
            y[i-burnin] = y_t
        end
    end
    return y
end

# Parameters
n = 1000
θ1 = 0.5
θ2 = 0.0
θ3 = 0.3

# Simulate MA process
y = simulate_ar_process(n, θ1, θ2, θ3)
CSV.write("Project01/problem4.csv", DataFrame(:y=>y))


# Plot ACF and PACF
y = CSV.read("Project01/problem4.csv",DataFrame)[!,:y]
p1 = plot(autocor(y)[1:10], title = "ACF of The Process", legend = false,seriestype="bar")
p2 = plot(pacf(y,1:10), title = "PACF of the Process", legend = false,seriestype="bar")
p = plot(p1,p2,layout=(2,1))

# c. The ACF and PACF of the data in problem4.csv suggest an AR as the ACF decays towards 0.
# the PACF has large values at lags 1,2, and 3.  
# This suggests an AR(3) process.

#d. To test, I'll fit an AR(1)-AR(4), as well as an MA(1) to the data and compare the AICc of each model.
ar = []
for i in 1:4
    push!(ar,SARIMA(y,order=(i,0,0),include_mean=true))
    StateSpaceModels.fit!(ar[i];optimizer=optimizer = Optimizer(StateSpaceModels.Optim.NelderMead()))
    println("AICc AR($i): ", ar[i].results.aicc)
end
ma = []
for i in 1:1
    push!(ma,SARIMA(y,order=(0,0,i),include_mean=true))
    StateSpaceModels.fit!(ma[i])
    println("AICc MA($i): ", ma[i].results.aicc)
end

# AICc AR(1): -1669.0646858139185
# AICc AR(2): -1695.4906861196323
# AICc AR(3): -1746.2203526352735
# AICc AR(4): -1733.5939510900746
# AICc MA(1): -1508.902984747084

# The AR(3) model has the lowest AICc, which is consistent with the ACF and PACF of the data.

#5 
#Use the stock returns in DailyReturn.csv for this problem. DailyReturn.csv contains returns for
# 100 large US stocks and as well as the ETF, SPY which tracks the S&P500.
# Create a routine for calculating an exponentially weighted covariance matrix. If you have a
# package that calculates it for you, verify that it calculates the values you expect. This means
# you still have to implement it.
# Vary . Use PCA and plot the cumulative variance explained λ ∈ (0, 1) by each eigenvalue for
# each λ chosen.
# What does this tell us about values of λ and the effect it has on the covariance matrix?

returns = CSV.read("Project01/DailyReturn.csv",DataFrame)
returns = Matrix(select(returns,Not(:Date)))

lmbds = [.7,.8,.9,.95,.99]
cExp = Dict{Float64,Vector{Float64}}()

function pctExplained(λ)
    cumsum(λ[end:-1:1]/sum(λ))
end

for l in lmbds
    cExp[l] = pctExplained(eigvals(ewCovar(returns,l)))
end

plot(1:100,[cExp[l] for l in lmbds],label=hcat([string(l) for l in lmbds]...),
    xlabel="Eigenvalue",ylabel="Cumulative Variance Explained",title="EW Covariance Matrix")

#As lambda gets smaller, the cumulative variance explained by each eigenvalue increases.  This means that
# the amount of infomation in the matrix is descreased as lambda decreases.  Effectively, the number of days
# that the covariance matrix is calculated over is reduced as lambda decreases.  This makes sense as shown in 
# the class notes.  

#6
# Implement a multivariatve normal simulation using the Cholesky root of the covariance matrix.
# Implement a multivariate normal simulation using PCA where Percent Explained is an input.
# Use the covariance matrix in problem5.csv to simulate 10,000 observations in 2 ways.
# a. Use the Cholesky root method.
# b. Use the PCA method with percent explained = .75.
# c. Take the covaraince of the simulated data.  Compare the Frobenius norm of these 
#    matrices to the original covariance matrix.  What do you notice?
# d. Compare the cumulative variance explained by each eigenvalue of the covariance matrices.  
#    What do you notice?
# e. Compare the time it took to run both simulations.
# f. Discuss the tradeoffs between the two methods.

#Data Generation
Random.seed!(6)
function random_cov_matrix(n)
    # Generate random standard deviations
    std_devs = rand(Uniform(0.02, 0.1), n)
    
    # Generate random correlation matrix
    corr_matrix = Matrix{Float64}(I, n, n)
    for i in 1:n, j in i+1:n
        corr_matrix[i, j] = corr_matrix[j, i] = rand(Uniform(-1, 1))
    end
    
    # Convert correlation matrix to covariance matrix
    cov_matrix = Diagonal(std_devs) * corr_matrix * Diagonal(std_devs)
    return cov_matrix
end
cout = random_cov_matrix(500)
eigvals(cov2cor(cout))
cout = higham_nearestPSD(cout)
CSV.write("Project01/problem6.csv", DataFrame(cout,:auto))

#Solution
c = CSV.read("Project01/problem6.csv",DataFrame) |> Matrix
n = 10000
c_chol = cov(simulateNormal(n,c))
c_pca = cov(simulate_pca(c,n;pctExp=.75))

#c
println("Frobenius Norm Cholesky: ", sum( (c - c_chol).^2 ))
println("Frobenius Norm PCA: ", sum( (c - c_pca).^2 ))
# Frobenius Norm Cholesky: 0.00045140989356500357
# Frobenius Norm PCA: 0.006912441616933017
# The Cholesky method is much closer to the original covariance matrix than the PCA method.

#d 
cExpChol = pctExplained(eigvals(c_chol))
cExpPCA = pctExplained(eigvals(c_pca))
cExpOrg = pctExplained(eigvals(c))
plot(1:500,[cExpChol cExpPCA cExpOrg],label=["Cholesky" "PCA" "Orginal"],xlabel="Eigenvalue"
    ,ylabel="Cumulative Variance Explained",title="EW Covariance Matrix")
# The PCA method cumulative variance explained hits 100% at around eigenvalue 75, while the Cholesky method
# tracks the original covariance matrix much more closely.

#e 
@btime simulateNormal(n,c)
# 359.709 ms (367725 allocations: 519.21 MiB)
@btime simulate_pca(c,n;pctExp=.75)
# 187.265 ms (2050 allocations: 126.46 MiB)

# The PCA method is much faster than the Cholesky method requiring less memory and fewer allocations.  
# However, the PCA method is less accurate than the Cholesky method.  
# We trade speed for accuracy between these two methods.

