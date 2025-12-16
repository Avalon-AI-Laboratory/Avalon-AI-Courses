# Optimization

## Table of Contents

- [Definition](#definition)
- [Gradients](#gradients)
- [Euler's Method](#eulers-method)
- [Newton's Method](#newtons-method)

## Definition

Optimization is the process of finding an optimal solution for a given problem. It is usually used in deep learning for training models.

Assume that we have a model $f(x;\theta)$ which starts with random parameters of $\theta$ and a loss function $J$ that returns an error of our model. The lower the error, then the better the model will become. Our goal in optimization is to find the best state, $\theta$, for the model.

![](https://static.wixstatic.com/media/3eee0b_155ce2aa05d5419698f844ab29062d70~mv2.gif)

## Gradients

Gradients are vectors of partial derivatives with respect to all the variables in the function.

$$
f(x, y) = x^2 + y^2
$$
$$
\begin{aligned}
\nabla f &= \begin{bmatrix}
\cfrac{\partial f}{\partial x} \\
\cfrac{\partial f}{\partial y}
\end{bmatrix} \\
&= \begin{bmatrix}
2x \\
2y
\end{bmatrix}
\end{aligned}
$$

These are useful when we want to find the minima or maxima of a function and are widely used in machine learning and optimization.

## Euler's Method

Euler's method is the simplest numerical method for approximating the solution to a first-order ordinary equation. The method relies on the idea of tangent line approximation and takes small, successive steps to "follow" the solution curve. The method is as follows:

1. Start with some point $x_0$
2. Update the point with this equation.

$$
y_{t+1} = y_t + h \cdot f(x_t, y_t)
$$
$$
x_{t+1} = x_t + h
$$

3. Repeat step 2 until $x$ reaches a desired value.

## Newton’s Method

Newton’s method is an method that can finds the point $x$ that returns $f(x) = 0$. The method is as follows:

1. Start with some point $x_0$.
2. Update the point with this equation:
    
$$
x_{t+1} = x_t - \frac{f(x)}{f'(x)}
$$
    
3. Repeat step 2 until $f(x)$ returns 0.

This method can be used for optimization as the minima can be determined by $f'(x) = 0$. Thus, we simply replace the function with its derivative to get the optimization method.

$$
x_{t+1} = x_t - \frac{f'(x_t)}{f''(x_t)}
$$

For multivariate models, we can use the Hessian matrix to compute the second derivative.

$$
\begin{bmatrix}
x_{t+1} \\
y_{t+1}
\end{bmatrix} = \begin{bmatrix}
x_t \\
y_t
\end{bmatrix} - H^{-1} \nabla f
$$

And all we have to do is to just use a loss function as $f$.

$$
\begin{bmatrix}
x_{t+1} \\
y_{t+1}
\end{bmatrix} = \begin{bmatrix}
x_t \\
y_t
\end{bmatrix} - H^{-1} \nabla L
$$
