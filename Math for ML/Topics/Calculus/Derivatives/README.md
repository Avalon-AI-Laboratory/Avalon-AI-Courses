# Derivatives

## Table of Contents

- [Derivatives](#derivatives)
- [The Exponential](#the-exponential)
- [Second Derivatives](#second-derivatives)
- [Differential Equations](#differential-equations)
- [Partial Derivatives](#partial-derivatives)
- [Hessian](#hessian)

## Derivatives

The slope of a function can be used to predict the next output given the current input. To do so, we can take two points on a function and calculate the slope with this equation:

$$
\begin{align*}
f'(x) &= \frac{y_2 - y_1}{x_2 - x_1} \\
&= \frac{\Delta y}{\Delta x}
\end{align*}
$$

This slope can get even more accurate as $\Delta x$ reaches a very small number ( $\Delta x \rightarrow 0$ ). This is what derivatives do.

In simple terms, derivatives can be viewed as a rate of change of some variable with respect to another variable. It is often written as $\frac{dy}{dx}$ (or $f'(x)$), which can be interpreted as the rate of change of the variable $y$ with respect to $x$.

The derivative is mathematically describes as the following:

$$
f'(x) = \lim_{\Delta x \rightarrow 0}\frac{f(x + \Delta x) - f(x)} {\Delta x }
$$

The derivative has several rules. Some of which include:

1. Power rule
    
$$
\frac{d}{dx} x^n = n x ^ {n - 1}
$$
    
2. Sum and difference rule
    
$$
\frac{d}{dx} \left[ u(x) \pm v(x) \right] = u'(x) \pm v'(x)
$$
    
3. Product rule
    
$$
\frac{d}{dx} \left[ u(x) v(x) \right] = u'(x) v(x) + u(x) v'(x)
$$
    
4. Quotient rule
    
$$
\frac{d}{dx} \left[ \frac{u(x)}{v(x)} \right] = \frac{ u'(x)v(x) - u(x)v'(x) }{ [ v(x) ] ^ 2 }
$$
    
5. Chain rule
    
$$
\frac{d}{dx} u(v(x)) = u'(v(x)) \cdot v'(x)
$$
    

The value of the derivative, let’s say $\frac{dy}{dx}$, can be interpreted as the following:

- If the derivative is positive, then $y$ increases as $x$ increases.
- If the derivative is negative, then $y$ decreases as $x$ increases.
- If the derivative is $0$, then $y$ does not change as $x$ increases.

The inverse of a function can have a derivative, which is:

$$
\left( f ^ {-1} \right)' (x) = \frac{1}{f ' \left[ f ^ {-1} (x) \right]}
$$

Trigonometric functions also have their own derivatives.

$$
\begin{array}{c|c}
\text{Function} & \text{Derivative} \\
\hline
\sin (x) & \cos (x) \\
\cos (x) & - \sin (x) \\
\tan (x) & \sec ^ 2 (x) \\
\csc (x) & - \cot (x) \csc (x) \\
\sec (x) & \tan (x) \sec (x) \\
\cot (x) & - \csc ^ 2 (x)
\end{array}
$$

The logarithmic function also has their own derivative. If we were to use the derivative of the inverse function, then we can prove that:

$$
\begin{align*}
\frac{d}{dx} \log (x) &= \frac{1}{10 ^ {\text{log} (x)}} \\
& = \frac{1}{x}
\end{align*}
$$

## The Exponential

The Euler number is a constant which is defined as the following:

$$
e = \lim_{ n \rightarrow \inf } \left( 1 + \frac{1}{n} \right)^n
$$

The number itself has a unique property, which is:

$$
\frac{d}{dx} e^x = e^x
$$

which means its rate of change is equal to its current value. 

## Second Derivatives

Derivatives can also have derivatives, which describe the rate of change of the rate of change (I’m not joking). One real world example of this property is acceleration, which is the rate of change of speed with respect to time and speed is the rate of change of position with respect to time as well.

$$
v = \frac{d x}{dt}
a = \frac{d v}{dt} = \frac{d^2x}{ dt^2}
$$

## Differential Equations

A differential equation is a mathematical equation that relates an unknown function with one or more of its derivatives.

Since the derivative describes the splote of a function, a differential equations provides a rule for the slope at any point. Solving the differential equation means finding the original unknown function that satisfies the relationship.

![](https://d18l82el6cdm1i.cloudfront.net/uploads/Gf8vVhA3Fe-tangent_function_animation.gif)

There are several types of differential equations, which are:

- Ordinary Differential Equations (ODEs)

    It involves only one independent variable and their ordinary derivatives.

- Partial Differential Equations (PDEs)

    It involves multiple independent variables and their partial derivatives.

One example use case of differential equations is the Euler's method (which we'll learn in [optimizations](../Optimization/))

## Partial Derivatives

In a function that takes account of more than one variable, partial derivatives are used as a way to find a derivative of one variable while other variables are treated as constants.

$$
\begin{align*}
f(x, y) &= x^2 + y^2 \\
\frac{\partial f}{\partial x} &= 2x \\
\frac{\partial f}{\partial y} &= 2y
\end{align*}
$$

![](https://tse3.mm.bing.net/th/id/OIP.JK6Twpc8oWEYM39K1tLh3gAAAA?rs=1&pid=ImgDetMain&o=7&rm=3)

## Hessian

The Hessian matrix stores information regarding second partial derivatives.

$$
H = \nabla ^2 f = \begin{bmatrix}
\cfrac{\partial f}{\partial x^2} & \cfrac{\partial f}{\partial xy} \\
\cfrac{\partial f}{\partial yx} &
\cfrac{\partial f}{\partial y^2}
\end{bmatrix}
$$

The Hessian matrix can be used to determine whether a point in a function is a minima, maxima, or saddle point. To do so, we search for its eigenvalues and compare each eigenvalues.

- If all eigenvalues are positive, then the point is the local minima.
- If all eigenvalues are negative, then the point is the or the local maxima.
- If the eigenvalues are positive and negative, or even zero, then the point is a saddle point.

![](https://iclr-blogposts.github.io/2024/assets/img/2024-05-07-bench-hvp/hess_eig.png)