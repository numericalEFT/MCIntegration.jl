var documenterSearchIndex = {"docs":
[{"location":"lib/montecarlo/#Monte-Carlo","page":"Monte Carlo","title":"Monte Carlo","text":"","category":"section"},{"location":"lib/montecarlo/","page":"Monte Carlo","title":"Monte Carlo","text":"Modules = [MCIntegration]","category":"page"},{"location":"lib/montecarlo/#MCIntegration.Configuration","page":"Monte Carlo","title":"MCIntegration.Configuration","text":"mutable struct Configuration\n\nStruct that contains everything needed for MC.\n\nStatic parameters\n\nseed: seed to initialize random numebr generator, also serves as the unique pid of the configuration\nrng: a MersenneTwister random number generator, seeded by seed\npara: user-defined parameter, set to nothing if not needed\nvar: TUPLE of variables, each variable should be derived from the abstract type Variable, see variable.jl for details). Use a tuple rather than a vector improves the performance.\n\nintegrand properties\n\nneighbor::Vector{Tuple{Int, Int}} : vector of tuples that defines the neighboring integrands. Two neighboring integrands are directly connected in the Markov chain.    e.g., [(1, 2), (2, 3)] means the integrand 1 and 2 are neighbor, and 2 and 3 are neighbor.    The neighbor vector defines a undirected graph showing how the integrands are connected. Please make sure all integrands are connected.  By default, we assume the N integrands are in the increase order, meaning the neighbor will be set to [(N+1, 1), (1, 2), (2, 4), ..., (N-1, N)], where the first N entries are for diagram 1, 2, ..., N and the last entry is for the normalization diagram. Only the first diagram is connected to the normalization diagram.  Only highly correlated integrands are not highly correlated should be defined as neighbors. Otherwise, most of the updates between the neighboring integrands will be rejected and wasted.\ndof::Vector{Vector{Int}}: degrees of freedom of each integrand, e.g., [[0, 1], [2, 3]] means the first integrand has zero var#1 and one var#2; while the second integrand has two var#1 and 3 var#2. \nobservable: observables that is required to calculate the integrands, will be used in the measure function call.   It is either an array of any type with the common operations like +-*/^ defined. \nreweight: reweight factors for each integrands. The reweight factor of the normalization diagram is assumed to be 1. Note that you don't need to explicitly add the normalization diagram. \nreweight_goal: The expected distribution of visited times for each integrand after reweighting . If not set, then all factors will be initialized with one.\nvisited: how many times this integrand is visited by the Markov chain.\n\ncurrent MC state\n\nstep: the number of MC updates performed up to now\ncurr: the current integrand, initialize with 1\nnorm: the index of the normalization diagram. norm is larger than the index of any user-defined integrands \nnormalization: the accumulated normalization factor. Physical observable = Configuration.observable/Configuration.normalization.\nrelativeWeight: integrand(config)/absWeight/config.reweight[config.curr], which is the reweighted weight of the current integrand  \nabsWeight: the abolute weight of the current integrand. User is responsible to initialize it after the contructor is called.\npropose/accept: array to store the proposed and accepted updates for each integrands and variables.  Their shapes are (number of updates X integrand number X max(integrand number, variable number).  The last index will waste some memory, but the dimension is small anyway.\n\n\n\n\n\n","category":"type"},{"location":"lib/montecarlo/#MCIntegration.Configuration-Union{Tuple{}, Tuple{V}} where V","page":"Monte Carlo","title":"MCIntegration.Configuration","text":"function Configuration(var::V, dof, obs::O=length(dof) == 1 ? 0.0 : zeros(length(dof));\n    para::P=nothing,\n    reweight::Vector{Float64}=ones(length(dof) + 1),\n    seed::Int=rand(Random.RandomDevice(), 1:1000000),\n    neighbor::Union{Vector{Vector{Int}},Vector{Tuple{Int,Int}},Nothing}=nothing\n) where {V,P,O}\n\nCreate a Configuration struct\n\nArguments\n\nvar: TUPLE of variables, each variable should be derived from the abstract type Variable, see variable.jl for details). Use a tuple rather than a vector improves the performance.\n\nBy default, var = (Continuous(0.0, 1.0),), which is a single continuous variable.\n\ndof::Vector{Vector{Int}}: degrees of freedom of each integrand, e.g., [[0, 1], [2, 3]] means the first integrand has zero var#1 and one var#2; while the second integrand has two var#1 and 3 var#2. \n\nBy default, dof=[ones(length(var)), ], which means that there is only one integrand, and each variable has one degree of freedom.\n\nobs: observables that is required to calculate the integrands, will be used in the measure function call.\n\nIt is either an array of any type with the common operations like +-*/^ defined.  By default, it will be set to 0.0 if there is only one integrand (e.g., length(dof)==1); otherwise, it will be set to zeros(length(dof)).\n\npara: user-defined parameter, set to nothing if not needed\nreweight: reweight factors for each integrands. If not set, then all factors will be initialized with one.\nreweight_goal: The expected distribution of visited times for each integrand after reweighting . If not set, then all factors will be initialized with one.\nseed: seed to initialize random numebr generator, also serves as the unique pid of the configuration. If it is nothing, then use RandomDevice() to generate a random seed in [1, 1000_1000]\nneighbor::Vector{Tuple{Int, Int}} : vector of tuples that defines the neighboring integrands. Two neighboring integrands are directly connected in the Markov chain.    e.g., [(1, 2), (2, 3)] means the integrand 1 and 2 are neighbor, and 2 and 3 are neighbor.     The neighbor vector defines a undirected graph showing how the integrands are connected. Please make sure all integrands are connected.   By default, we assume the N integrands are in the increase order, meaning the neighbor will be set to [(N+1, 1), (1, 2), (2, 4), ..., (N-1, N)], where the first N entries are for diagram 1, 2, ..., N and the last entry is for the normalization diagram. Only the first diagram is connected to the normalization diagram.   Only highly correlated integrands are not highly correlated should be defined as neighbors. Otherwise, most of the updates between the neighboring integrands will be rejected and wasted.\n\n\n\n\n\n","category":"method"},{"location":"lib/montecarlo/#MCIntegration.Result","page":"Monte Carlo","title":"MCIntegration.Result","text":"struct Result{O,C}\n\nthe returned result of the MC integration.\n\nMembers\n\nmean: mean of the MC integration\nstdev: standard deviation of the MC integration\nchi2: chi-square of the MC integration\nneval: number of evaluations of the integrand\ndof: degrees of freedom of the MC integration (number of iterations - 1)\nconfig: configuration of the MC integration from the last iteration\niteractions: list of tuples [(data, error, Configuration), ...] from each iteration\n\n\n\n\n\n","category":"type"},{"location":"lib/montecarlo/#MCIntegration.average","page":"Monte Carlo","title":"MCIntegration.average","text":"function average(history, max=length(history))\n\naverage the history[1:max]. Return the mean, standard deviation and chi2 of the history.\n\nArguments\n\nhistory: a list of tuples, such as [(data, error, Configuration), ...]\nmax: the number of data to average over\n\n\n\n\n\n","category":"function"},{"location":"lib/montecarlo/#MCIntegration.create!-Tuple{Continuous, Int64, Any}","page":"Monte Carlo","title":"MCIntegration.create!","text":"create!(T::Continuous, idx::Int, rng=GLOBAL_RNG)\n\nPropose to generate new (uniform) variable randomly in [T.lower, T.lower+T.range), return proposal probability\n\nArguments\n\nT:  Continuous variable\nidx: T.data[idx] will be updated\n\n\n\n\n\n","category":"method"},{"location":"lib/montecarlo/#MCIntegration.create!-Tuple{Discrete, Int64, Any}","page":"Monte Carlo","title":"MCIntegration.create!","text":"create!(newIdx::Int, size::Int, rng=GLOBAL_RNG)\n\nPropose to generate new index (uniformly) randomly in [1, size]\n\nArguments\n\nnewIdx:  index ∈ [1, size]\nsize : up limit of the index\nrng=GLOBAL_RNG : random number generator\n\n\n\n\n\n","category":"method"},{"location":"lib/montecarlo/#MCIntegration.create!-Tuple{MCIntegration.RadialFermiK, Int64, Any}","page":"Monte Carlo","title":"MCIntegration.create!","text":"create!(K::RadialFermiK, idx::Int, rng=GLOBAL_RNG)\n\nPropose to generate new k randomly in [0, +inf), return proposal probability k is generated uniformly on [0, K.kF-K.δk), Lorentzianly on [K.kF-K.δk,K.kF+K.δk), and exponentially on [K.kF-K.δk, +inf).\n\nArguments\n\nK:  k variable\nidx: K.data[idx] will be updated\n\n\n\n\n\n","category":"method"},{"location":"lib/montecarlo/#MCIntegration.create!-Tuple{TauPair, Int64, Any}","page":"Monte Carlo","title":"MCIntegration.create!","text":"create!(T::TauPair, idx::Int, rng=GLOBAL_RNG)\n\nPropose to generate a new pair of tau (uniformly) randomly in [0, β), return proposal probability\n\nArguments\n\nT:  TauPair variable\nidx: T.data[idx] will be updated\n\n\n\n\n\n","category":"method"},{"location":"lib/montecarlo/#MCIntegration.create!-Union{Tuple{D}, Tuple{FermiK{D}, Int64, Any}} where D","page":"Monte Carlo","title":"MCIntegration.create!","text":"create!(K::FermiK{D}, idx::Int, rng=GLOBAL_RNG)\n\nPropose to generate new Fermi K in [Kf-δK, Kf+δK)\n\nArguments\n\nnewK:  vector of dimension of d=2 or 3\n\n\n\n\n\n","category":"method"},{"location":"lib/montecarlo/#MCIntegration.integrate-Tuple{Function}","page":"Monte Carlo","title":"MCIntegration.integrate","text":"function integrate(integrand::Function;\n    config::Union{Configuration,Nothing}=nothing,\n    measure::Function=simple_measure,\n    neval=1e5, \n    niter=10, \n    block=16, \n    alpha=1.0, \n    print=-1, \n    printio=stdout,\n    kwargs...\n)\n\nCalculate the integrals, collect statistics, and return a Result struct that contains the estimations and errors.\n\nRemarks\n\nUser may run the MC in parallel using MPI. Simply run mpiexec -n N julia userscript.jl where N is the number of workers. In this mode, only the root process returns meaningful results. All other workers return nothing, nothing. User is responsible to handle the returning results properly. If you have multiple number of mpi version, you can use \"mpiexecjl\" in your \"~/.julia/package/MPI/###/bin\" to make sure the version is correct. See https://juliaparallel.github.io/MPI.jl/stable/configuration/ for more detail.\nIn the MC, a normalization diagram is introduced to normalize the MC estimates of the integrands. More information can be found in the link: https://kunyuan.github.io/QuantumStatistics.jl/dev/man/important_sampling/#Important-Sampling. User don't need to explicitly specify this normalization diagram.Internally, normalization diagram will be added to each table that is related to the integrands.\n\nArguments\n\nintegrand: function call to evaluate the integrand. It should accept an argument of the type Configuration, and return a weight.   Internally, MC only samples the absolute value of the weight. Therefore, it is also important to define Main.abs for the weight if its type is user-defined. \nconfig: Configuration object to perform the MC integration. If nothing, it attempts to create a new one with Configuration(; kwargs...).\nmeasure: function call to measure. It should accept an argument the type Configuration. Then you can accumulate the measurements with Configuration.obs.   If every integral is expected to be a float number, you can use MCIntegration.simple_measure as the default.\nneval: number of evaluations of the integrand per iteration. \nniter: number of iterations. The reweight factor and the variables will be self-adapted after each iteration. \nblock: Number of blocks. Each block will be evaluated by about neval/block times. Each block is assumed to be statistically independent, and will be used to estimate the error.   In MPI mode, the blocks are distributed among the workers. If the numebr of workers N is larger than block, then block will be set to be N.\nalpha: Learning rate of the reweight factor after each iteraction. Note that alpha <=1, where alpha = 0 means no reweighting.  \nprint: -1 to not print anything, 0 to print minimal information, >0 to print summary for every print seconds\nprintio: io to print the information\nkwargs: keyword arguments. If config is nothing, you may need to provide arguments for the Configuration constructor, check Configuration docs for more details.\n\nExamples\n\njulia> integrate(c->(X=c.var[1]; X[1]^2+X[2]^2); var = (Continuous(0.0, 1.0), ), dof = [(2, ),], print=-1)\nIntegral 1 = 0.6830078240204353 ± 0.014960689298028415   (chi2/dof = 1.46)\n\n\n\n\n\n","category":"method"},{"location":"lib/montecarlo/#MCIntegration.remove!-Tuple{Continuous, Int64, Any}","page":"Monte Carlo","title":"MCIntegration.remove!","text":"remove(T::Continuous, idx::Int, rng=GLOBAL_RNG)\n\nPropose to remove old variable in [T.lower, T.lower+T.range), return proposal probability\n\nArguments\n\nT:  Continuous variable\nidx: T.data[idx] will be updated\n\n\n\n\n\n","category":"method"},{"location":"lib/montecarlo/#MCIntegration.remove!-Tuple{Discrete, Int64, Any}","page":"Monte Carlo","title":"MCIntegration.remove!","text":"remove!(newIdx::Int, size::Int, rng=GLOBAL_RNG)\n\nPropose to remove the old index in [1, size]\n\nArguments\n\noldIdx:  index ∈ [1, size]\nsize : up limit of the index\nrng=GLOBAL_RNG : random number generator\n\n\n\n\n\n","category":"method"},{"location":"lib/montecarlo/#MCIntegration.remove!-Tuple{MCIntegration.RadialFermiK, Int64, Any}","page":"Monte Carlo","title":"MCIntegration.remove!","text":"remove(K::RadialFermiK, idx::Int, rng=GLOBAL_RNG)\n\nPropose to remove old k in [0, +inf), return proposal probability\n\nArguments\n\nK:  K variable\nidx: K.data[idx] will be updated\n\n\n\n\n\n","category":"method"},{"location":"lib/montecarlo/#MCIntegration.remove!-Tuple{TauPair, Int64, Any}","page":"Monte Carlo","title":"MCIntegration.remove!","text":"remove(T::TauPair, idx::Int, rng=GLOBAL_RNG)\n\nPropose to remove an existing pair of tau in [0, β), return proposal probability\n\nArguments\n\nT:  Tau variable\nidx: T.data[idx] will be updated\n\n\n\n\n\n","category":"method"},{"location":"lib/montecarlo/#MCIntegration.remove!-Union{Tuple{D}, Tuple{FermiK{D}, Int64, Any}} where D","page":"Monte Carlo","title":"MCIntegration.remove!","text":"removeFermiK!(oldK, Kf=1.0, δK=0.5, rng=GLOBAL_RNG)\n\nPropose to remove an existing Fermi K in [Kf-δK, Kf+δK)\n\nArguments\n\noldK:  vector of dimension of d=2 or 3\n\n\n\n\n\n","category":"method"},{"location":"lib/montecarlo/#MCIntegration.shift!-Tuple{Continuous, Int64, Any}","page":"Monte Carlo","title":"MCIntegration.shift!","text":"shift!(T::Continuous, idx::Int, rng=GLOBAL_RNG)\n\nPropose to shift an existing variable to a new one, both in [T.lower, T.lower+T.range), return proposal probability\n\nArguments\n\nT:  Continuous variable\nidx: T.data[idx] will be updated\n\n\n\n\n\n","category":"method"},{"location":"lib/montecarlo/#MCIntegration.shift!-Tuple{Discrete, Int64, Any}","page":"Monte Carlo","title":"MCIntegration.shift!","text":"shift!(d::Discrete, idx::Int, config)\n\nPropose to shift the old index in [1, size] to a new index\n\nArguments\n\noldIdx:  old index ∈ [1, size]\nnewIdx:  new index ∈ [1, size], will be modified!\nsize : up limit of the index\nrng=GLOBAL_RNG : random number generator\n\n\n\n\n\n","category":"method"},{"location":"lib/montecarlo/#MCIntegration.shift!-Tuple{MCIntegration.RadialFermiK, Int64, Any}","page":"Monte Carlo","title":"MCIntegration.shift!","text":"shift!(K::RadialFermiK, idx::Int, rng=GLOBAL_RNG)\n\nPropose to shift an existing k to a new k, both in [0, +inf), return proposal probability\n\nArguments\n\nK:  K variable\nidx: K.data[idx] will be updated\n\n\n\n\n\n","category":"method"},{"location":"lib/montecarlo/#MCIntegration.shift!-Tuple{TauPair, Int64, Any}","page":"Monte Carlo","title":"MCIntegration.shift!","text":"shift!(T::TauPair, idx::Int, rng=GLOBAL_RNG)\n\nPropose to shift an existing tau pair to a new tau pair, both in [0, β), return proposal probability\n\nArguments\n\nT:  Tau variable\nidx: T.t[idx] will be updated\n\n\n\n\n\n","category":"method"},{"location":"lib/montecarlo/#MCIntegration.shift!-Union{Tuple{D}, Tuple{FermiK{D}, Int64, Any}} where D","page":"Monte Carlo","title":"MCIntegration.shift!","text":"shiftK!(oldK, newK, step, rng=GLOBAL_RNG)\n\nPropose to shift oldK to newK. Work for generic momentum vector\n\n\n\n\n\n","category":"method"},{"location":"lib/montecarlo/#MCIntegration.summary","page":"Monte Carlo","title":"MCIntegration.summary","text":"function summary(result::Result, pick::Union{Function,AbstractVector}=obs -> real(first(obs)), name=nothing)\n\nprint the summary of the result.  It will first print the configuration from the last iteration, then print the weighted average and standard deviation of the picked observable from each iteration.\n\nArguments\n\nresult: Result object contains the history from each iteration\npick: The pick function is used to select one of the observable to be printed. The return value of pick function must be a Number.\nname: name of each picked observable. If name is not given, the index of the pick function will be used.\n\n\n\n\n\n","category":"function"},{"location":"lib/montecarlo/#MCIntegration.swap!-Tuple{Discrete, Int64, Int64, Any}","page":"Monte Carlo","title":"MCIntegration.swap!","text":"swap!(d::Discrete, idx1::Int, idx2::Int, config)\n\nSwap the variables idx1 and idx2\n\n\n\n\n\n","category":"method"},{"location":"lib/montecarlo/#MCIntegration.train!-Tuple{Continuous}","page":"Monte Carlo","title":"MCIntegration.train!","text":"Vegas adaptive map\n\n\n\n\n\n","category":"method"},{"location":"man/important_sampling/#Important-Sampling","page":"Important Sampling","title":"Important Sampling","text":"","category":"section"},{"location":"man/important_sampling/#Introduction","page":"Important Sampling","title":"Introduction","text":"","category":"section"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"This note compares two important sampling approaches for Monte Carlo integration. The first approach introduces a normalization sector and lets the Markov chain jumps between this additional sector and the integrand sector following a calibrated probability density for important sampling. One can infer the integration between the ratio of weights between two sectors. On the other hand, the second approach reweights the original integrand to make it as flat as possible, one then perform a random walk uniformly in the parameter space to calculate the integration. This is the conventional approach used in Vegas algorithm.","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"In general, the first approach is more robust than the second one, but less efficient. In many applications, for example, high order Feynman diagrams with a sign alternation, the important sampling probability can't represent the complicated integrand well. Then the first approach is as efficient as the second one, but tends to be much robust.","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"We next present a benchmark between two approaches. Consider the MC sampling of an one-dimensional functions f(x) (its sign may oscillate).","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"We want to design an efficient algorithm to calculate the integral int_a^b dx f(x). To do that, we normalize the integrand with an ansatz g(x)0 to reduce the variant. ","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"Our package supports two important sampling schemes. ","category":"page"},{"location":"man/important_sampling/#Approach-1:-Algorithm-with-a-Normalization-Sector","page":"Important Sampling","title":"Approach 1: Algorithm with a Normalization Sector","text":"","category":"section"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"In this approach, the configuration spaces consist of two sub-spaces: the physical sector with orders nge 1 and the normalization sector with the order n=0. The weight function of the latter, g(x), should be simple enough so that the integral G=int g(x) d x is explicitly known. In our algorithm we use a constant g(x) propto 1 for simplicity. In this setup, the physical sector weight, namely the integral F = int f(x) dx, can be calculated with the equation","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"    F=fracF_rm MCG_rm MC G","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"where the MC estimators F_rm MC and G_rm MC are measured with ","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"F_rm MC =frac1N left sum_i=1^N_f fracf(x_i)rho_f(x_i) + sum_i=1^N_g 0 right","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"and","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"G_rm MC =frac1N leftsum_i=1^N_f 0 + sum_i=1^N_g fracg(x_i)rho_g(x_i)  right","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"The probability density of a given configuration is proportional to rho_f(x)=f(x) and rho_g(x)=g(x), respectively. After N MC updates, the physical sector is sampled for N_f times, and the normalization sector is for N_g times. ","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"Now we estimate the statistic error. According to the propagation of uncertainty, the variance of F  is given by","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":" fracsigma^2_FF^2 =  fracsigma_F_rm MC^2F_MC^2 + fracsigma_G_rm MC^2G_MC^2 ","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"where sigma_F_rm MC and sigma_G_rm MC are variance of the MC integration F_rm MC and G_rm MC, respectively. In the Markov chain MC, the variance of F_rm MC can be written as ","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"sigma^2_F_rm MC = frac1N left sum_i^N_f left( fracf(x_i)rho_f(x_i)- fracFZright)^2 +sum_j^N_g left(0-fracFZ right)^2  right ","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"= int left( fracf(x)rho_f(x) - fracFZ right)^2 fracrho_f(x)Z rm dx + int left( fracFZ right)^2 fracrho_g(x)Z dx ","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"=  int fracf^2(x)rho_f(x) fracdxZ -fracF^2Z^2 ","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"Here Z=Z_f+Z_g and Z_fg=int rho_fg(x)dx are the partition sums of the corresponding configuration spaces. Due to the detailed balance, one has Z_fZ_g=N_fN_g.  ","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"Similarly, the variance of G_rm MC can be written as ","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"sigma^2_G_rm MC=  int fracg^2(x)rho_g(x) fracdxZ - fracG^2Z^2","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"By substituting rho_f(x)=f(x) and  rho_g(x)=g(x), the variances of F_rm MC and G_rm MC are given by","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"sigma^2_F_rm MC= frac1Z^2 left( Z Z_f - F^2 right)","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"sigma^2_G_rm MC= frac1Z^2 left( Z Z_g - G^2 right)","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"We derive the variance of F as","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"fracsigma^2_FF^2 = fracZ cdot Z_fF^2+fracZ cdot Z_gG^2 - 2 ","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"Note that g(x)0 indicates Z_g = G,  so that","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"fracsigma^2_FF^2 = fracZ_f^2F^2+fracGcdot Z_fF^2+fracZ_fG - 1","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"Interestingly, this variance is a function of G instead of a functional of g(x). It is then possible to normalized g(x) with a constant to minimize the variance. The optimal constant makes G to be,","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"fracd sigma^2_FdG=0","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"which makes G_best = F. The minimized the variance is given by,","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"fracsigma^2_FF^2= left(fracZ_fF+1right)^2 - 2ge 0","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"The equal sign is achieved when f(x)0 is positively defined.","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"It is very important that the above analysis is based on the assumption that the autocorrelation time negligible. The autocorrelation time related to the jump between the normalization and physical sectors is controlled by the deviation of the ratio f(x)g(x) from unity. The variance sigma_F^2 given above will be amplified to sim sigma_F^2 tau where tau is the autocorrelation time.","category":"page"},{"location":"man/important_sampling/#Approach-2:-Conventional-algorithm-(e.g.,-Vegas-algorithm)","page":"Important Sampling","title":"Approach 2: Conventional algorithm (e.g., Vegas algorithm)","text":"","category":"section"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"Important sampling is actually more straightforward than the above approach. One simply sample x with a distribution rho_g(x)=g(x)Z_g, then measure the observable f(x)g(x). Therefore, the mean estimation,","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"fracFZ=int dx fracf(x)g(x) rho_g(x)","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"the variance of F in this approach is given by,","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"sigma_F^2=Z_g^2int dx left( fracf(x)g(x)- fracFZ_gright)^2rho_g(x)","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"fracsigma_F^2F^2=fracZ_gF^2int dx fracf(x)^2g(x)- 1","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"The optimal g(x) that minimizes the variance is g(x) =f(x),","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"fracsigma_F^2F^2=fracZ_f^2F^2-1","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"The variance of the conventional approach is a functional of g(x), while that of the previous approach isn't. There are two interesting limit:\nIf the f(x)0, the optimal choice g(x)=f(x) leads to zero variance. In this limit, the conventional approach is clearly much better than the previous approach.\nOn the other hand, if g(x) is far from the optimal choice f(x), say simply setting g(x)=1, one naively expect that the the conventional approach may leads to much larger variance than the previous approach. However,  this statement may not be true. If g(x) is very different from f(x), the normalization and the physical sector in the previous approach mismatch, causing large autocorrelation time and large statistical error . In contrast, the conventional approach doesn't have this problem.","category":"page"},{"location":"man/important_sampling/#Benchmark","page":"Important Sampling","title":"Benchmark","text":"","category":"section"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"To benchmark, we sample the following integral up to 10^8 updates, ","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"int_0^beta e^-(x-beta2)^2delta^2dx approx sqrtpidelta","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"where beta gg delta.","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"g(x)=f(x)","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"Normalization Sector:  doesn't lead to exact result, the variance left(fracZ_fF+1right)^2 - 2=2 doesn't change with parameters","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"beta 10 100\nresult 0.1771(1) 0.1773(1)","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"Conventional: exact result","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"g(x)=sqrtpideltabeta1","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"beta 10 100\nNormalization 0.1772(4) 0.1767(17)\nConventional 0.1777(3) 0.1767(8)","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"g(x)=exp(-(x-beta2+s)^2delta^2) with beta=100","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"s delta 2delta 3delta 4delta 5delta\nNormalization 0.1775(8) 0.1767(25) 0.1770(60) 0.176(15) 183(143)\nConventional 0.1776(5) 0.1707(39) 0.1243(174) 0.0204 (64) ","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"The conventional algorithm is not ergodic anymore for s=4delta, the acceptance ratio to update x is about 015, while the normalization algorithm becomes non ergodic for s=5delta. So the latter is slightly more stable.","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"<!– The code are ![[test.jl]] for the normalization approach and ![[test2.jl]] for the conventional approach. –>","category":"page"},{"location":"man/important_sampling/","page":"Important Sampling","title":"Important Sampling","text":"Reference:  [1] Wang, B.Z., Hou, P.C., Deng, Y., Haule, K. and Chen, K., Fermionic sign structure of high-order Feynman diagrams in a many-fermion system. Physical Review B, 103, 115141 (2021).","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = MCIntegration","category":"page"},{"location":"#MCIntegration","page":"Home","title":"MCIntegration","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for MCIntegration.","category":"page"},{"location":"","page":"Home","title":"Home","text":"A Monte Carlo calculator for high dimension integration.","category":"page"},{"location":"#Manual-Outline","page":"Home","title":"Manual Outline","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Pages = [\n\"man/important_sampling.md\"\n]\nDepth = 1","category":"page"},{"location":"#Library-Outline","page":"Home","title":"Library Outline","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Pages = [\n    \"lib/montecarlo.md\",\n]\nDepth = 1","category":"page"}]
}
