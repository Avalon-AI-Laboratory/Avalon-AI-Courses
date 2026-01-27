# Understanding Euler’s Number ($e$)

## An Introduction Using Bacteria Growth

### 1. Why Do We Even Talk About $e$?

In many areas of science and machine learning, we often encounter expressions like $e^x$. At first glance, this feels strange: Why this specific number? Why not 2, 3, or 10? To understand this, we must look at continuous growth. We shouldn't look at money or complex formulas yet—instead, let's look at a simple biological process.

### 2. A Simple Thought Experiment: Bacteria in One Hour
Imagine a very simple scenario:

- At time $t = 0$: We have 1 bacterium.
- After exactly 1 hour: The population becomes 2 bacteria.

In this condition, the population doubles within one hour. The key question is not what happens after 2 or 3 hours, but how the growth happens inside that single hour

### 3. Growth: All at Once vs. Gradually

<b>Case 1: Sudden Growth (Discrete) </b>

The simplest model assumes nothing happens during the hour, and then snap the bacterium splits at the very last second.

- Start of hour: 1
- End of hour: 2

This is easy to imagine, but it is not how nature actually works.

<b>Case 2: Growth in Two Steps </b>

Now suppose the bacterium grows twice during the hour (once at 30 minutes and once at 60 minutes). To still end up with 2 bacteria, each step must multiply the population by $\sqrt{2}$.

$$\sqrt{2} \times \sqrt{2} = 2$$

The final result is still 2, but the process is slightly smoother.

<b>Case 3: Growth in Many Small Steps </b>

Let’s push this further. Suppose growth happens:

- 10 times per hour or 100 times per hour or 1,000,000 times per hour

Mathematically, this looks like:

$$\left(1 + \frac{1}{n}\right)^n$$

Where $n$ is the number of tiny growth steps in one hour.

4. What Happens When Growth Becomes Continuous?

As $n$ gets larger, each growth step becomes smaller and the growth becomes smoother. Surprisingly, the final value doesn't grow to infinity; it stops increasing significantly and hits a limit.

In the limit:

$$\lim_{n \to \infty} \left(1 + \frac{1}{n}\right)^n = e$$

Numerically:

$$e \approx 2.71828$$

This means if growth happens continuously and naturally, the growth factor is $e$, not 2. 

### 5. The Key Insight

Notice something subtle but profound:

1. We are still only observing one hour.

2. We are not adding more time.

3. We are only changing how finely the growth is divided.

At first, you might expect more frequent growth to make the population explode. Instead, the increase becomes smaller and smaller as it approaches a natural limit. This limit is exactly Euler’s number, $e$.