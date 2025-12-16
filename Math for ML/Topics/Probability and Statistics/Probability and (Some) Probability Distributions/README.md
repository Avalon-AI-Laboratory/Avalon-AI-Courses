# Probability and (Some) Probability Distributions

## Table Of Contents

- [Probability](#probability)
- [Independence](#independence)
- [Conditional Probability](#conditional-probability)
- [Bayes' Theorem](#bayes-theorem)
    - [Naive Bayes](#naive-bayes)
- [Random Variables](#random-variables)
- [Probability Mass Functions](#probability-mass-functions)
- [Probability Density Functions](#probability-density-functions)
    - [(Some) Discrete Probability Distributions](#some-discrete-probability-distributions)
    - [(Some) Continuous Probability Distributions](#some-continuous-probability-distributions)
- [Joint Distributions](#joint-distributions)
- [Marginal Distributions](#marginal-distributions)
- [Conditional Distributions](#conditional-distributions)
- [Covariance](#covariance)
- [Correlation Coefficient](#correlation-coefficient)

## Probability

Probability is the measure of how likely an event is to occur. It usually denoted as the following:

$$
P(A) = \frac{n(A)}{n(S)}
$$

where $A$ is an event that happened in the sample space $S$. Obviously, the probability of all events occurring in the sample space is 100% or 1.

$$
P(S) = 1
$$

There’s also complementary probability which is the probability of any other event happening other than the given event. The complementary probability of event $A$ is defined as:

$$
P(A') = 1 - P(A)
$$

In most cases, many events can happen in the same sample space and we want to find probabilities of each event to happen with one another. Such case is called an intersection, which is denoted as the following:

$$
P(A \cap B)
$$

Another case is when we want to find out the probability of at least one of the events to happen. This is called a union, which is denoted as the following:

$$
P(A \cup B)
$$

To find the union of events, we have to know the type of events. Events can be classified into two groups, which are:

- Disjoint Events
    
The events are unrelated withone another, such that:
    
$$
P(A \cap B) = 0
$$
    
which means:
    
$$
P(A \cup B) = P(A) + P(B)
$$
    
- Joint Events
    
The events are related withone another, such that:

$$
P(A \cap B) \ne 0
$$

which means:

$$
P(A \cup B) = P(A) + P(B) - (A \cap B)
$$
    

## Independence

Events can also be considered independent if the occurrence of an event does not affect the probability of another event. Two events $A$ and $B$ are considered independent if:

$$
P(A \cap B) = P(A) P(B)
$$

## Conditional Probability

Conditional probability talks about how probabilities change when another event occurs. The new probability of event $A$ occurring after event $B$ occurs is:

$$
P(A | B) = \frac{P(A \cap B)}{P(B)}, \space P(B) \ne 0
$$

In the case of independent events, then the probability won’t change.

$$
P(A|B) = P(A)
$$

## Bayes’ Theorem

If we were substitute conditional probabilities which each other:

$$
\begin{align*}
P(A | B) = \frac{P(A \cap B)}{P(B)} \\
P(B | A) = \frac{P(A \cap B)}{P(A)}
\end{align*}
$$

then we can find the cause of each event given its effect:

$$
\begin{align*}
P(A \cap B) = P(B|A)P(A) \\
P(A | B) = \frac{P(B|A) P(A)}{P(B)}, \space P(B) \ne 0
\end{align*}
$$

The expression above yields Bayes’ theorem.

A more formal definition of Bayes’ theorem in a sample space with a set of events $E_1, E_2, \dots, E_n$ is defined as the following:

$$
P(E_i | A) = \frac{P(E_i) P(A | E_i)}{ \sum P(E_j) P(A | E_j)}
$$

$P(E_i)$ in this case is called a prior probability, which are initial probabilities of the events. $P(E_i | A)$ is called a posterior probability, which are updated probabilities after considering new information.

### Naive Bayes

Bayes’ theorem can be used for many applications. For example, we can predict a label $y$ given a set of features $x$.

$$
\begin{align*}
P(y | x) &= \frac{P(x | y) P(y)}{P(x)}
\end{align*}
$$

where $P(x) = P(x_1 \cap x_2 \cap \dots \cap x_n)$.

However, in most cases, we might find features that were never found in the prior probabilities, such that $P(x) = 0$, thus the model becomes invalid.

To handle this problem, we can “naively” assume that all the features are independent.

$$
P(x) = \prod P(x_i)
$$

Thus, we can fix the model.

$$
P(y | x) = \frac{P(y) \prod P(x_i | y)}{\prod P(x_i)}
$$

And since $P(x)$ is constant for all labels, then we can ignore it for easier computation.

$$
P(y | x) \propto P(y) \prod P(x_i | y)
$$

To classify with the model, we find the label with the largest probability.

$$
\hat y = \arg \max_y P(y | x)
$$

This is essentially the Naive Bayes model. It’s a popular classification machine learning model which doesn’t require a large dataset yet still produces accurate results. However, the independence assumption might not hold for all datasets.

## Random Variables

Random variables are variables that can take many values, like the temperature of a room or number of heads after 10 coin tosses. They are usually associated with uncertain outcomes.

There are two types of random variables:

- Discrete
- Continuous

## Probability Mass Functions

Probability mass functions (PMF) describe the probability of a discrete variable takes on each possible value. There are several rules that define a PMF, which are:

- Non-Negativity Rule
    
$$
0 \le P(x) \le 1 \space\space \text{for all }x
$$
    
- Total Probability Rule
    
$$
\sum P(x) = 1
$$
    

![](https://i.ytimg.com/vi/Ns1Uixo3_3U/maxresdefault.jpg)

## Probability Density Functions

Probability density functions (PDF) are used to describe the likelihood of continuous variables. It can be seen as a continuous version of PMFs. 

![https://tse1.mm.bing.net/th/id/OIP.0ZCGAR7il0dzDRqiedU8qAHaFj?rs=1&pid=ImgDetMain&o=7&rm=3](https://tse1.mm.bing.net/th/id/OIP.0ZCGAR7il0dzDRqiedU8qAHaFj?rs=1&pid=ImgDetMain&o=7&rm=3)

But unlike discrete probabilities, PDFs returns densities as probabilities, which are integrations over an interval to get an actual probability.

$$
P(a \le X \le b) = \int^b_a f(x) dx
$$

Another difference is that the probability of $x$ being an exact number is impossible, or $0$.

$$
P(X=x) = 0
$$

## Cumulative Density Functions

Cumulative density functions (CDF) are essentially accumulations of PDFs. They sum probabilities up to a given point.

$$
\begin{align*}
F_X(x) &= P(X \le x) \\
&= \begin{cases}
\sum_{-\infty}^x f(x) & \text{if discrete} \\
\int^{x}_{-\infty} f(x) dx & \text{if continuous}
\end{cases}
\end{align*}
$$

## Describing Probability Distributions

When we describe a probability distribution, there are two measures, which are:

### Measures of Central Tendency
    
  This measure describes where    the central position of the distribution.
    
- Mean / Expected Value
        
    The average of the distribution. It can also be seen as the weighted average of the random variables.
        
$$
\mu = \mathbb E [X] = 
\begin{cases} 
\sum x f(x) & \text{if discrete} \\
\int x f(x) dx & \text{icontinuous}
\end{cases}
$$
        
- Median
        
    The median is the middle value of the distribution when ordered. If there are two middle values, then take the average.
        
- Mode

    The mode is the most frequent value that appears in the distribution.
        
    
![](https://media.geeksforgeeks.org/wp-content/uploads/20250501122658765639/mean_mod_median.webp)
    
### Measures of Spread
    
This measure describes how spread out the distribution is. 
    
- Variance
        
    Variance explains how spread out the data is.
        
$$
\begin{align*}
\sigma ^ 2 = \text{Var}(X) &= \mathbb E [ ( X - \mathbb E [ X ] ) ^ 2 ]\\
&= \mathbb E [ X ^ 2 ] - \mathbb E [ X ] ^ 2 \\
\end{align*}
$$

- Standard deviation
        
    Standard deviation is similar to variance, but in the same unit as the data.
        
$$
\sigma = \text{std}(X) = \sqrt{ \text{Var} (X) }
$$
        
- Skewness
        
    This measure determines how skewed the distribution is.
        
$$
\text{Skewness} = \mathbb E \left[ \left( \frac{X - \mu}{\sigma} \right) ^ 3 \right]
$$
        
There are several types of skewness, which are:
        
- Positive skew / right-skewed
            
    This type is caused by $\text{Skewness} > 0$
            
    The right tail is longer or fatter than the left. Thus most of the data are concentrated on the left with unusually large values pulling the mean to the right. 
            
    It can be interpreted as Mean > Median > Mode.
            
- Negative skew / left-skewed
            
    This type is caused by $\text{Skewness} < 0$

    The left tail is longer or fatter than the right. Thus most of the data are concentrated on the right with unusually large values pulling the mean to the left. 
            
    It can be interpreted as Mean < Median < Mode.
            
- Zero skew
            
    This type is caused by $\text{Skewness} = 0$
            
    Both sides are equal. This can be interpreted as Mean = Median = Mode.
    
![](https://av-eks-blogoptimized.s3.amazonaws.com/sk1.png)
            
- Kurtosis
        
    This measure explains how much probability lies in the tails versus the mean.
        
$$
\text{Kurtosis} = \mathbb{E} \left[ \left( \frac{X - \mu}{\sigma} \right) ^ 4 \right]
$$
        
There are also types of kurtosis, which include:
        
- Mesokurtic
            
    This is caused by $\text{Kurtosis} = 3$
            
    This is characterized with  a normal bell shaped curved.
            
    It means that the distribution is overall normal.
            
- Leptokurtic
            
    This is caused by $\text{Kurtosis} > 3$
            
    This is characterized with heavier tails and sharper peak.
            
    It means that there are outliers in the distribution.
            
- Platykurtic
            
    This is caused by $\text{Kurtosis} < 3$
            
    This is characterized with  lighter tails and flatter peak.
            
    It means the probabilities are evenly spread out.
    
![](https://cdn-images-1.medium.com/max/1600/1*Nqu07THa7APRTOF7kaVr5Q.jpeg)

One way to describe these measures is by visualizing the data. There are several methods of visualizing data, such as:

- Boxplots
    
    ![](https://builtin.com/sites/www.builtin.com/files/styles/ckeditor_optimize/public/inline-images/1_boxplots.jpg)
    
- Kernel density estimation
    
    ![](https://miro.medium.com/v2/resize:fit:578/1*VtGO4058FxKN5vbtnfTGmw.png)
    
- Violin plots
    
    ![](https://images.ctfassets.net/fi0zmnwlsnja/sdfgtcRp16wTNOcRceGQm/5bfcb73d2261d49ff20dd7857e0152b1/Screen_Shot_2019-03-01_at_11.36.10_AM.png)
    
- QQ plots
    
    ![https://cdn.buttercms.com/3Av1YayDSkGEttHUTuRf](https://cdn.buttercms.com/3Av1YayDSkGEttHUTuRf)
    

## (Some) Discrete Probability Distributions

### Bernoulli Distribution

The Bernoulli distribution describes the probability distribution of a variable with two possible outcomes in a single trial. The PDF is defined as the following:

$$
\text{Bernoulli}(x; p) = p ^ x (1 - p) ^ x
$$

where $p$ is the probability of success.

![](https://www.wallstreetmojo.com/wp-content/uploads/2022/08/Bernoulli-Distribution-Graph.png)

### Binomial Distribution

The Binomial distribution can be seen as an extension of the Bernoulli distribution with multiple trials. The PDF is defined as the following:

$$
\text{Binomial}(x; n, p) = \dbinom{n}{x} p ^ x (1 - p) ^ {n - x}
$$

where $n$ is the number of trials and $p$ is the probability of success.

![](https://statisticsglobe.com/wp-content/uploads/2019/08/rbinom-histogram-random-numbers-in-r-featured.png)

## (Some) Continuous Probability Distributions

### (Continuous) Uniform Distribution

The (Continuous) Uniform distribution describe the likelihood of continuous variables in a certain interval that are homogenous.

$$
\text{Uniform}(x; a, b) = 
\begin{cases}
\cfrac{1}{b - a} & a \le x \le b \\
0 & \text{otherwise}
\end{cases}
$$

![](https://miro.medium.com/v2/resize:fit:564/0*CxkBLGQ2CuvoLkGU.png)

### Normal / Gaussian Distribution

The Normal distribution describe how many real-world quantities naturally vary around a central average value $\mu$ and deviate with a certain spread measured by $\sigma$.

$$
\text{Normal}(x; \mu, \sigma) = \frac{1}{\sqrt{2 \pi} \sigma} \exp \left[ {- \frac{(x - \mu) ^ 2}{2 \sigma ^ 2} } \right]
$$

![https://tse2.mm.bing.net/th/id/OIP.a4nb3j_Y-VZdJqCzECKNSQHaDt?rs=1&pid=ImgDetMain&o=7&rm=3](https://tse2.mm.bing.net/th/id/OIP.a4nb3j_Y-VZdJqCzECKNSQHaDt?rs=1&pid=ImgDetMain&o=7&rm=3)

### Student’s t-Distribution

It is a distribution similar to the Normal distribution but with fatter tails since it reflects extra uncertainty when we have a small sample size ($n \le 30$) or an unknown $\sigma$.

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_lossy/https://substack-post-media.s3.amazonaws.com/public/images/068a72bb-5d57-4cde-96c2-2115318f7f0b_979x653.gif)

The shape of the interval depends of degrees of freedom $v$, which is defined as the following:

$$
v = n - 1
$$

## Joint Distributions

Joint distributions take account of multiple random variables instead of just one. It is denoted as:

$$
P_{XY}(x, y) = P(X = x \cap Y = y)
$$

![](https://images.deepai.org/glossary-terms/375b2234f023403994cd0b6a612009b9/jointdist.png)

## Marginal Distributions

Marginal distributions are distributions of one variable while completely ignoring other variables.

$$
P_X(x) = 
\begin{cases}
\sum P_{XY}(x, y_i) & \text{if discrete} \\
\int P_{XY}(x, y) dy & \text{if continuous}
\end{cases} \\
P_Y(y) = 
\begin{cases}
\sum P_{XY}(x_i, y) & \text{if discrete} \\
\int P_{XY}(x, y) dx & \text{if continuous}
\end{cases}
$$

## Conditional Distributions

Just like probabilities, there are conditional distributions which describes the new distribution given some information.

$$
P_{X | Y = y}(x) = \frac{P_{XY}(X = x, Y = y)}{P_X(Y = y)}
$$

![](https://quantifyinghealth.com/wp-content/uploads/2022/09/conditional-distribution-plot-of-height.png)

## Covariance

Covariance explains how correlated a variable is with another variable.

$$
\begin{align*}
\text{Cov}(X, Y) &= \begin{cases}
\cfrac{\sum (x_i - \mu_x)(y_i - \mu_y)}{N} & \text{if discrete} \\
\cfrac{\int\int(x - \mu_x)(y - \mu_y) dx dy}{N - 1} & \text{if continuous}
\end{cases} \\
&= \mathbb E [XY] - \mathbb E[X] \mathbb E[Y]
\end{align*}
$$

The covariance is interpreted as the following:

- If the covariance reaches $+\infty$, then both variables are positively correlated
- If the covariance reaches $-\infty$, then both variables are negatively correlated
- If the covariance reaches 0, then both variables have no correlation.

## Correlation Coefficient

Correlation coefficient is similar to covariance, but in the range of $[-1, 1]$. This leads to better interpretation.

$$
\text{Correlation} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}
$$

The correlation coefficient is interpreted as the following:

- If the correlation coefficient reaches 1, then both variables are positively correlated
- If the correlation coefficient reaches -1, then both variables are negatively correlated
- If the correlation coefficient reaches 0, then both variables have no correlation.
