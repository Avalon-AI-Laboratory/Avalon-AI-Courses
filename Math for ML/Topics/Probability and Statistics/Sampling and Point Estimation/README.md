# Sampling and Point Estimation

## Table Of Contents

- [Population and Sample](#population-and-sample)
- [Central Limit Theorem](#central-limit-theorem)
- [Point Estimation](#point-estimation)
- [Regularization](#regularization)
- [Bayesian Statistics](#bayesian-statistics)

## Population and Sample

A population is the entire set of elements that is studied. In most studies, it is impossible to study an entire population due to resource constraints. Researchers tend to take a subset of the population called a sample and infer or draw conclusions about the population with the selected sample.

![https://articles.outlier.org/_next/image?url=https:%2F%2Fimages.ctfassets.net%2Fkj4bmrik9d6o%2F5hIEOEewVartM64NEqdUSO%2F05818454e55feb389cc88b8d4bfaaa65%2FPopulation_vs._Sample_01.png&w=2048&q=75](https://articles.outlier.org/_next/image?url=https:%2F%2Fimages.ctfassets.net%2Fkj4bmrik9d6o%2F5hIEOEewVartM64NEqdUSO%2F05818454e55feb389cc88b8d4bfaaa65%2FPopulation_vs._Sample_01.png&w=2048&q=75)

There are several ways to obtain a sample from a population, which include:

- Random sampling
    
Randomly select samples from the entire population.
    
- Stratified sampling
    
Take the same amount of random samples from several stratums or classes in a population.
    

The name of measures in population and samples also vary.

$$
\begin{array}{c|c}
\text{Population} & \text{Sample} \\
\hline 
\mu & \bar X \\
\sigma^2 & s^2 \\
\sigma & s \\
P & \hat p
\end{array}
$$

## Central Limit Theorem

Central Limit Theorem states that even if the original data are not normally distributed, the average of many samples will form a normal distribution as the number of samples grows.

![](https://danielvartan.github.io/central-limit-theorem/index_files/figure-html/unnamed-chunk-9-1.gif)

Given the sample size $n$, the sample mean $\bar X$ and the population standard deviation $\sigma$, then we can obtain the 

$$
Z = \frac{\bar X - \mu}{ \frac{\sigma}{\sqrt n}}
$$

## Point Estimation

Point estimation is the process of estimating the best parameters from a sample. One way to do this is with the Maximum Likelihood Estimation (MLE) method.

1. Write a likelihood function
    
$$
L(\theta) = P(x | \theta)
$$
    
2. Take the log likelihood
    
$$
\log L(\theta) = \log P(x | \theta)
$$
    
3. Differentiate and maximize
    
$$
\frac{d}{d \theta} \log L(\theta) = 0
$$
    

## Regularization

Regularization helps us choose the best fit model by adding an extra penalty to handle overfitting models. This penalty takes account of model parameters.

$$
L_{\text{reg}}(\theta) = L(\theta) + \lambda R(\theta)
$$

where $R(\theta)$ is the regularization term with a scale of $\lambda$. There are many regularization methods, such as:

- L1
    
$$
R(\theta) = \sum | \theta_i |
$$
    
- L2
    
$$
R(\theta) = \sum \theta_i^2
$$
    

## Bayesian Statistics

Bayesian statistics is a branch of statistics that uses probability to represent uncertainty about the data and updates the probabilities as new data are observed.

| **Feature** | **Frequentist** | **Bayesian** |
| --- | --- | --- |
| **Definition of Probability** | Long term frequency of events | Degree of belief or certainty |
| **Inference** | Likelihood | Priors |
| **Goal** | Find the model that most likely generated the new data | Update prior beliefs based of the new data |
