# Important Sampling

## Introduction

This note compares two important sampling approaches for Monte Carlo integration. The first approach introduces a normalization sector and lets the Markov chain jumps between this additional sector and the integrand sector following a calibrated probability density for important sampling. One can infer the integration between the ratio of weights between two sectors. On the other hand, the second approach reweights the original integrand to make it as flat as possible, one then perform a random walk uniformly in the parameter space to calculate the integration. This is the conventional approach used in Vegas algorithm.

In general, the first approach is more robust than the second one, but less efficient. In many applications, for example, high order Feynman diagrams with a sign alternation, the important sampling probability can't represent the complicated integrand well. Then the first approach is as efficient as the second one, but tends to be much robust.

We next present a benchmark between two approaches. Consider the MC sampling of an one-dimensional functions $f(x)$ (its sign may oscillate).

We want to design an efficient algorithm to calculate the integral $\int_a^b dx f(x)$. To do that, we normalize the integrand with an ansatz $g(x)>0$ to reduce the variant. 

Our package supports two important sampling schemes. 

## Approach 1: Algorithm with a Normalization Sector

In this approach, the configuration spaces consist of two sub-spaces: the physical sector with orders $n\ge 1$ and the normalization sector with the order $n=0$. The weight function of the latter, $g({x})$, should be simple enough so that the integral $G=\int g({x}) d x$ is explicitly known. In our algorithm we use a constant $g(x) \propto 1$ for simplicity. In this setup, the physical sector weight, namely the integral $F = \int f(x) d{x}$, can be calculated with the equation
```math
    F=\frac{F_{\rm MC}}{G_{\rm MC}} G
```
where the MC estimators $F_{\rm MC}$ and $G_{\rm MC}$ are measured with 
```math
F_{\rm MC} =\frac{1}{N} \left[ \sum_{i=1}^{N_f} \frac{f(x_i)}{\rho_f(x_i)} + \sum_{i=1}^{N_g} 0 \right]
```
  and
```math
G_{\rm MC} =\frac{1}{N} \left[\sum_{i=1}^{N_f} 0 + \sum_{i=1}^{N_g} \frac{g(x_i)}{\rho_g(x_i)}  \right]
```

The probability density of a given configuration is proportional to $\rho_{f}(x)=|f(x)|$ and $\rho_{g}(x)=|g(x)|$, respectively. After $N$ MC updates, the physical sector is sampled for $N_f$ times, and the normalization sector is for $N_g$ times. 

Now we estimate the statistic error. According to the propagation of uncertainty, the variance of $F$  is given by
```math
 \frac{\sigma^2_F}{F^2} =  \frac{\sigma_{F_{\rm MC}}^2}{F_{MC}^2} + \frac{\sigma_{G_{\rm MC}}^2}{G_{MC}^2}, 
```
where $\sigma_{F_{\rm MC}}$ and $\sigma_{G_{\rm MC}}$ are variance of the MC integration $F_{\rm MC}$ and $G_{\rm MC}$, respectively. In the Markov chain MC, the variance of $F_{\rm MC}$ can be written as 
```math
\sigma^2_{F_{\rm MC}} = \frac{1}{N} \left[ \sum_{i}^{N_f} \left( \frac{f(x_i)}{\rho_f(x_i)}- \frac{F}{Z}\right)^2 +\sum_{j}^{N_g} \left(0-\frac{F}{Z} \right)^2  \right] 
```
```math
= \int \left( \frac{f(x)}{\rho_f(x)} - \frac{F}{Z} \right)^2 \frac{\rho_f(x)}{Z} {\rm d}x + \int \left( \frac{F}{Z} \right)^2 \frac{\rho_g(x)}{Z} dx 
```
```math
=  \int \frac{f^2(x)}{\rho_f(x)} \frac{dx}{Z} -\frac{F^2}{Z^2} 
```
Here $Z=Z_f+Z_g$ and $Z_{f/g}=\int \rho_{f/g}({x})d{x}$ are the partition sums of the corresponding configuration spaces. Due to the detailed balance, one has $Z_f/Z_g=N_f/N_g$.  

Similarly, the variance of $G_{\rm MC}$ can be written as 
```math
\sigma^2_{G_{\rm MC}}=  \int \frac{g^2(x)}{\rho_g(x)} \frac{dx}{Z} - \frac{G^2}{Z^2}
```

By substituting $\rho_{f}(x)=|f(x)|$ and  $\rho_{g}(x)=|g(x)|$, the variances of $F_{\rm MC}$ and $G_{\rm MC}$ are given by
```math
\sigma^2_{F_{\rm MC}}= \frac{1}{Z^2} \left( Z Z_f - F^2 \right)
```
```math
\sigma^2_{G_{\rm MC}}= \frac{1}{Z^2} \left( Z Z_g - G^2 \right)
```
We derive the variance of $F$ as
```math
\frac{\sigma^2_F}{F^2} = \frac{Z \cdot Z_f}{F^2}+\frac{Z \cdot Z_g}{G^2} - 2 
```
Note that $g(x)>0$ indicates $Z_g = G$,  so that
```math
\frac{\sigma^2_F}{F^2} = \frac{Z_f^2}{F^2}+\frac{G\cdot Z_f}{F^2}+\frac{Z_f}{G} - 1
```
Interestingly, this variance is a function of $G$ instead of a functional of $g(x)$. It is then possible to normalized $g(x)$ with a constant to minimize the variance. The optimal constant makes $G$ to be,
```math
\frac{d \sigma^2_F}{dG}=0,
```
which makes $G_{best} = |F|$. The minimized the variance is given by,
```math
\frac{\sigma^2_F}{F^2}= \left(\frac{Z_f}{F}+1\right)^2 - 2\ge 0.
```
The equal sign is achieved when $f(x)>0$ is positively defined.

**It is very important that the above analysis is based on the assumption that the autocorrelation time negligible. The autocorrelation time related to the jump between the normalization and physical sectors is controlled by the deviation of the ratio $|f(x)|/g(x)$ from unity. The variance $\sigma_F^2$ given above will be amplified to $\sim \sigma_F^2 \tau$ where $\tau$ is the autocorrelation time.**

## Approach 2: Conventional algorithm (e.g., Vegas algorithm)

Important sampling is actually more straightforward than the above approach. One simply sample $x$ with a distribution $\rho_g(x)=g(x)/Z_g$, then measure the observable $f(x)/g(x)$. Therefore, the mean estimation,
```math
\frac{F}{Z}=\int dx \frac{f(x)}{g(x)} \rho_g(x)
```

the variance of $F$ in this approach is given by,
```math
\sigma_F^2=Z_g^2\int dx \left( \frac{f(x)}{g(x)}- \frac{F}{Z_g}\right)^2\rho_g(x)
```
```math
\frac{\sigma_F^2}{F^2}=\frac{Z_g}{F^2}\int dx \frac{|f(x)|^2}{|g(x)|}- 1
```
The optimal $g(x)$ that minimizes the variance is $g(x) =|f(x)|$,
```math
\frac{\sigma_F^2}{F^2}=\frac{Z_f^2}{F^2}-1
```

- The variance of the conventional approach is a functional of $g(x)$, while that of the previous approach isn't. There are two interesting limit:
- If the $f(x)>0$, the optimal choice $g(x)=|f(x)|$ leads to zero variance. In this limit, the conventional approach is clearly much better than the previous approach.
- On the other hand, if $g(x)$ is far from the optimal choice $|f(x)|$, say simply setting $g(x)=1$, one naively expect that the the conventional approach may leads to much larger variance than the previous approach. **However,  this statement may not be true. If $g(x)$ is very different from $f(x)$, the normalization and the physical sector in the previous approach mismatch, causing large autocorrelation time and large statistical error . In contrast, the conventional approach doesn't have this problem.**

## Benchmark
- To benchmark, we sample the following integral up to $10^8$ updates, 
```math
\int_0^\beta e^{-(x-\beta/2)^2/\delta^2}dx \approx \sqrt{\pi}\delta
```
where $\beta \gg \delta$.
1. ``g(x)=|f(x)|``
- __Normalization Sector__:  __doesn't__ lead to exact result, the variance $\left(\frac{Z_f}{F}+1\right)^2 - 2=2$ doesn't change with parameters

| $\beta$ | 10        | 100       |
| ------- | --------- | --------- |
| result  | 0.1771(1) | 0.1773(1) |

- __Conventional__: exact result
2. ``g(x)=\sqrt{\pi}\delta/\beta1``

| $\beta$       | 10        | 100        |
| ------------- | --------- | ---------- |
| Normalization | 0.1772(4) | 0.1767(17) |
| Conventional  | 0.1777(3) | 0.1767(8)  |

3. ``g(x)=exp(-(x-\beta/2+s)^2/\delta^2)`` with ``\beta=100``

| $s$           | $\delta$  | $2\delta$  | $3\delta$   | $4\delta$   | $5\delta$ |
| ------------- | --------- | ---------- | ----------- | ----------- | --------- |
| Normalization | 0.1775(8) | 0.1767(25) | 0.1770(60)  | 0.176(15)   | 183(143)  |
| Conventional  | 0.1776(5) | 0.1707(39) | 0.1243(174) | 0.0204 (64) |

The conventional algorithm is not ergodic anymore for $s=4\delta$, the acceptance ratio to update $x$ is about $0.15%$, while the normalization algorithm becomes non ergodic for $s=5\delta$. So the latter is slightly more stable.

<!-- The code are ![[test.jl]] for the normalization approach and ![[test2.jl]] for the conventional approach. -->

**Reference**: 
[1] Wang, B.Z., Hou, P.C., Deng, Y., Haule, K. and Chen, K., Fermionic sign structure of high-order Feynman diagrams in a many-fermion system. Physical Review B, 103, 115141 (2021).

