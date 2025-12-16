# Confidence Interval and Hypothesis Testing

## Table Of Contents

- [Confidence Intervals](#confidence-intervals)
- [Calculating Sample Sizes](#calculating-sample-sizes)
- [Confidence Intervals for Population Parameters](#confidence-intervals-for-population-parameters)
    - [One-Sample](#one-sample)
        - [Mean (With $\sigma$)](#mean-with-)
        - [Mean (Without $\sigma$)](#mean-without-)
        - [Proportion](#proportion)
    - [Two-Sample](#two-sample)
        - [Difference Between Two Means (with $\sigma_1$ and $\sigma_2$)](#difference-between-two-means-with--and-)
        - [Difference Between Two Means (without $\sigma_1$ and $\sigma_2$ but equal)](#difference-between-two-means-without--and--but-equal)
        - [Difference Between Two Means (without $\sigma_1$ and $\sigma_2$ but unequal)](#difference-between-two-means-without--and--but-unequal)
        - [Difference Between Two Proportions](#difference-between-two-proportions)
- [Hypothesis Testing](#hypothesis-testing)
- [Type I and Type II Errors](#type-i-and-type-ii-errors)
- [p-value](#p-value)
- [A/B Testing](#ab-testing)

## Confidence Intervals

Confidence interval is an interval of values which contain the population parameter. It is used as an estimate of a population parameter given a sample.

![](https://www.inchcalculator.com/wp-content/uploads/2021/11/confidence-interval.png)

The width of the interval is determined by a confidence level (typically 95%), which states how many intervals from repeated samples would contain the true population parameter. This confidence levels is also determined by a significance level $\alpha$ by calculating $1 - \alpha$.

It’s important to note that the interpretation of a confidence level isn’t the probability of the true population parameter falling inside the confidence interval, but rather the amount of confidence intervals that would capture the true population parameter.

![](https://statisticseasily.com/wp-content/uploads/2023/04/interpretation-of-a-confidence-interval-1.jpg)

The general form of a confidence interval is defined as the following:

$$
\text{Confidence Interval} = \text{Sample Statistic} + \text{Margin of Error}
$$

There are many methods to calculate confidence intervals to estimate population parameters, but the general steps include:

1. Find the sample statistic.
2. Define a desired confidence interval $\alpha$.
3. Get the critical value.
4. Find the standard error.
5. Find the margin of error.
6. Add or subtract the margin of error with the sample parameter to obtain the upper and lower bounds of the confidence interval.

## Calculating Sample Sizes

Before conducting a study, it’s important to determine the required sample size $n$ to make our study precise. This calculation is derived from the margin of error $\text{MOE}$.

$$
\text{MOE} = \left \lceil z_{\alpha / 2} \left( \frac{\sigma}{\sqrt n} \right) \right \rceil
$$

where:

- $z_{\alpha / 2}$ is the critical value to the desired confidence interval.
- $\sigma$ is the population standard deviation.
- $n$ is the sample size.

If we rearrange the equation, then we can estimate the required sample size $n$.

$$
n \ge \left(z_{ \alpha / 2} \cfrac{\sigma}{\text {MOE}} \right) ^ 2
$$

## Confidence Intervals for Population Parameters

### One-Sample

#### Mean (With $\sigma$)

Assume that:

- The standard deviation of the population is $\sigma$.
- The population is normally distributed.

Then we can use the Normal distribution to get the confidence interval of the mean estimate.

The confidence interval is defined as the following:

$$
\bar x \pm z_{1 - \alpha / 2} \frac{\sigma}{\sqrt n}
$$

#### Mean (Without $\sigma$)

Assume that:

- The standard deviation of the population is unknown.
- The population is normally distributed.

Then we use the t-distribution and the sample’s standard deviation $s$ instead. 

The confidence interval is defined as the following:

$$
\bar x \pm t_{1 - \alpha / 2} \frac{s}{\sqrt n}
$$

#### Proportion

Assume that:

- There are two outcomes.
- The population follows the Binomial distribution (but we can use the Normal approach when $np \ge 5$ and $n ( 1 - p ) \ge 5$)

Then we can use the Normal distribution.

The confidence interval is defined as the following:

$$
\hat p \pm z_{1 - \alpha / 2} \sqrt \frac{\hat p (1 - \hat p)}{n}
$$

### Two-Sample

#### Difference Between Two Means (with $\sigma_1$ and $\sigma_2$)

Assume that:

- The standard deviation of both samples are known.
- The population is normally distributed.
- The sample spaces are independent.

Then we can use Normal distribution.

The confidence interval is defined as the following:

$$
(\bar{X_1} - \bar{X_2}) \pm z_{\alpha / 2} \sqrt { \frac{\sigma^2_1}{n_1}  + \frac{\sigma^2_2}{n_2} }
$$

#### Difference Between Two Means (without $\sigma_1$ and $\sigma_2$ but equal)

Assume that:

- The standard deviation of both samples are unknown but equal ($\sigma_1 = \sigma_2$).
- The population is normally distributed.
- The sample spaces are independent.

Then we can use the t-distribution ad the same standard deviation $s_p$.

The confidence interval is defined as the following:

$$
(\bar X_1 - \bar X_2) \pm t_{a / 2} s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}
$$

where:

$$
s_p = \sqrt \frac{(n_1 - 1) s^2_1 + (n_2 - 1) s^2_2}{n_1 + n_2 - 2}
$$

#### Difference Between Two Means (without $\sigma_1$ and $\sigma_2$ but unequal)

Assume that:

- The standard deviation of both samples are unknown and unequal ($\sigma_1 \ne \sigma_2$).
- The population is normally distributed.
- The sample spaces are independent.

Then we can use the t-distribution and separate standard deviations $s_1$ and $s_2$.

The confidence interval is defined as the following:

$$
(\bar X_1 - \bar X_2) \pm t_{a / 2} \sqrt{\frac{s^2_1}{n_1} + \frac{s^2_2}{n_2}}
$$

#### Difference Between Two Proportions

Assume that:

- There are two outcomes.
- The population follows the Binomial distribution (but we can use the Normal approach when $np \ge 5$ and $n ( 1 - p ) \ge 5$)

The confidence interval is defined as the following:

$$
(\hat p_1 - \hat p_2) \pm Z_{a/2} \sqrt{\frac{\hat p_1 \hat q_1}{n_1} + \frac{\hat p_2 \hat q_2}{n_2}}
$$

where $\hat q_1 = ( 1 - \hat p )$.

## Hypothesis Testing

Hypothesis testing is a formal statistical method to infer about a population using data from a sample. The key idea is that we start with two competing claims or hypotheses:

- Null Hypothesis ($H_0$)
    
    The “no difference” statement or the status quo, which is why this hypothesis always include “=”.
    
    We initially assume that this hypothesis is true unless the data show strong contradicting evidence.
    
- Alternative Hypothesis ($H_1$ or $H_a$)
    
    The hypothesis we want to prove that there is a difference.
    

The steps to conduct hypothesis testing include:

1. Define the hypotheses
2. Choose a significance level
3. Collect sample data
4. Compute the test statistic
5. Determine the critical value
6. Make a decision
7. State the conclusion

There are several types of hypothesis testing, which are:

- Left-tailed
    
$$
\begin{align*}
H_0 : \mu = 10 \\
H_1 : \mu < 10
\end{align*}
$$
    
- Right-tailed
    
$$
\begin{align*}
H_0 : \mu = 10 \\
H_1 : \mu > 10
\end{align*}
$$
    
- Two-tailed
    
$$
\begin{align*}
H_0 : \mu = 10 \\
H_1 : \mu \ne 10
\end{align*}
$$
    

![](https://media.istockphoto.com/id/1732976415/vector/difference-between-null-and-alternative-hypothesis.jpg?s=170667a&w=0&k=20&c=YRDyzA-PeixaNUaajeUYbnW5dhrP5qXJ4J5AotaqxFA=)

## Type I and Type II Errors

When we conduct a hypothesis test, we make a decision based of the sample data. We either reject $H_0$ or fail to reject $H_0$. However, inferring the population from the sample data does not guarantee the correct decision. Such mistakes are classified into two types of errors, which are:

- Type I error (False Positive)
    
    We reject $H_0$ when it is actually true.
    
- Type II error (False Negative)
    
    We fail to reject $H_0$ when it is actually false.
    

$$
\begin{array}{c|cc}
& H_0\ \text{true} & H_1\ \text{true} \\ 
\hline
\text{Fail to reject } H_0 & \text{Correct decision} & \text{Type II error } (\beta) \\
\text{Reject } H_0 & \text{Type I error } (\alpha) & \text{Correct decision} \\
\end{array}
$$

In most real-world cases, the type II error is more fatal than the type I error.

![](https://s3-us-west-2.amazonaws.com/courses-images/wp-content/uploads/sites/2789/2017/12/05040102/TypeITypeIIErrors1.jpg)

## p-value

A p-value is the probability, assuming the null hypothesis $H_0$ is true, that the test statistic takes on a value as extreme as or more extreme than the value observed. In other words, it’s the area outside of the confidence interval.

The decision rule is as follows:

- If $\text{P-value} \le \alpha$, then we can reject the null hypothesis $H_0$.
- If $\text{P-value} \gt \alpha$, then we fail to reject the null hypothesis $H_0$ since we don’t have enough evidence.

## A/B Testing

A/B testing is a method for comparing to variations, $A$ and $B$, of a variable to determine which performs better in achieving a specific goal. It’s essentially an application of two-sample hypothesis testing used to optimize business outcomes. In A/B testing, $A$ is considered the default version of the variable and $B$ is the modified version.

The process of A/B testing is as follows:

1. Propose variants (A/B)
2. Randomly split samples
3. Measure outcomes for each group and determine a metric
4. Conduct statistical analysis to make a decision
