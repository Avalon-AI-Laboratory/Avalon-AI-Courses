# System of Linear Equations

## Table of Contents

- [Definition](#definition)
- [System Solutions](#system-solutions)
- [Linear Dependence](#linear-dependence)
- [Matrix Rank](#matrix-rank)
- [Solving Linear Systems](#solving-linear-systems)

## Definition

An equation is a statement that asserts the equality of two expressions. It can be written like the following:

$$
x + 2 y = 5
$$

The goal of an equation is to find the values of each variables, or also known as **unknowns**, of the equation. However, one equation might not be enough to find the value of all unknowns, so multiple equations are needed to give enough information to find such values. 

Such multiple equations that involve the same set of unknowns can form a system of equations. If the equations are linear (no variables are multiplied together, no exponents greater than 1, etc.), then it can be called a system of linear equations. It can be written like the following:

$$
\begin{align*}
x + 2 y &= 5 \\
3 x + 6 y &= 10 \\
2 x + y &= 2 \\
\vdots
\end{align*}
$$

Mathematically, if there are $n$ unknowns and $m$ equations, then the system might look like:

$$
\begin{align*}
a_{11} x_1 + a_{12} x_2 + \cdots + a_{1n} x_n & = b_1 \\
a_{21} x_1 + a_{22} x_2 + \cdots + a_{2n} x_n & = b_2 \\
\vdots \\
a_{m1} x_1 + a_{m2} x_2 + \cdots + a_{mn} x_n & = b_m
\end{align*}
$$

The system can be written in matrix form:

$$
\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{bmatrix} = 
\begin{bmatrix}
b_1 \\
b_2 \\
\vdots \\
b_n
\end{bmatrix}
$$

And for the sake of simplicity, it can be viewed as an augmented matrix:

$$
\begin{bmatrix}
\begin{array}{cccc|c}
a_{11} & a_{12} & \cdots & a_{1n} & b_1 \\
a_{21} & a_{22} & \cdots & a_{2n} & b_2 \\
\vdots & \vdots & \ddots & \vdots & \vdots\\
a_{m1} & a_{m2} & \cdots & a_{mn} & b_n
\end{array}
\end{bmatrix}
$$

## System Solutions

The solution of a system is the set of values of unknowns that makes all the equations in the system true. There are several types of solutions, which are:

- Unique
    
There is exactly one solution to the equation. For example:
    
$$
\begin{align*}
x + 2y &= 5 \\
2x + y &= 4
\end{align*}
$$

This system will exactly have one solution, which is $x = 1$ and $y = 2$.
    
- No Solution / Inconsistent
    
There is no solution to the system as there are contradictory information in the system. For example:
    
$$
\begin{align*}
x + 2y &= 5 \\
x + 2y &= 4
\end{align*}
$$
    
Since there are contradictory information, the system does not have any valid solution.
    
- Infinitely Many Solutions
    
There are more than one solution to the system. This can happen because of redundant information. For example:
    
$$
\begin{align*}
x + 2y &= 5 \\
-x - 2y &= -5
\end{align*}
$$
    
The system can have an infinite amount of solutions. If we assume that $x = 5 - 2y$, then any value of $y$ will have a valid value for $x$.
    

![](https://www.onlinemathlearning.com/image-files/consistent-inconsistent-system.png)

Another way to determine the solution of a system is by singularity.

- Singular
    
Singular systems contain redundant or contradictory information.
    
- Non-Singular
    
Non-singular contain enough information for a valid solution.
    
These views can mapped into the following table.

$$
\begin{array}{c|c|c|c}
& \text{Unique} & \text{Inconsistent} & \text{Infinite} \\
\hline
\text{Singular} & & \checkmark & \checkmark \\
\text{Non-Singular} & \checkmark & &
\end{array}
$$

## Linear Dependence

Linear dependence determines whether any vector can be expressed as a linear combination of other vectors in the system.

A more formal definition of linear dependence can be stated as the following. Given some vectors $x_1, x_2, \dots, x_n$ and scalars, some vectors are said to be linearly dependent if there exists $a_1, a_2, \dots, a_n$ such that:

$$
a_1 x_1 + a_2 x_2 + \dots + a_n x_n = 0
$$

and at least one the scalars is different from zero.

For instance, if we have two vectors:

$$
x_1 = \begin{bmatrix}
1 \\ 
2
\end{bmatrix} \, 
x_2 = \begin{bmatrix} 
2 \\ 
4
\end{bmatrix}
$$

then they are linearly dependent since:

$$
-2 x_1 + x_2 = 0
$$

Linear dependence can be used to ensure that a set of vectors does not contain any redundant vectors.

Linear dependence can be determined by the determinant of the system. There are several ways to compute the determinant. For a $2 \times 2$ matrix, it can be computed as:

$$
\begin{align*}
\det ( A ) & = 
\begin{vmatrix}
a & b \\
c & d
\end{vmatrix} \\
& = a d - b c
\end{align*}
$$

For larger matrices, it can be computed with the Laplace / Cofactor Expansion. The method is as follows:

1. Pick any row or column
    
It is recommended to pick the row or column with the most zeros for easier computation.
    
2. Obtain the determinant with the following formula

$$
\det (A) = \sum^n_{j=1} (-1)^{i + j} a_{ij} \det (M_{ij})
$$
    
where $M_{ij}$ is the minor matrix formed by deleting the $i$-th row and $j$-th column.
    

For example, if the matrix is:

$$
A = \begin{bmatrix}
a & b & c \\
d & e & f \\
g & h & i
\end{bmatrix}
$$

then with Laplace / Cofactor Expansion on the first row, the determinant would be:

$$
\det (A) = a 
\begin{vmatrix}
e & f \\
h & i
\end{vmatrix} - b 
\begin{vmatrix}
d & f \\
g & i
\end{vmatrix} + c 
\begin{vmatrix}
d & e \\
g & h
\end{vmatrix}
$$

The determinant can be interpreted into linear dependence like the following:

- If $\det ( A ) = 0$, then the system is linearly dependent. There are contradictory or redundant information in the system.
- If $\det ( A ) \ne 0$, then the system is linearly independent. There is enough information to provide a unique solution.

## Matrix Rank

The rank of a matrix determines how many independent rows / columns are available in a matrix. It can be used as a measure of how much information is provided to solve the system.

$$
\begin{cases}
A = \begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix} & \text{rank}(A) = 2 \\
B = \begin{bmatrix}
1 & 1 \\
0 & 0
\end{bmatrix} & \text{rank}(B) = 1 \\
C = \begin{bmatrix}
0 & 0 \\
0 & 0
\end{bmatrix} & \text{rank}(C) = 0
\end{cases}
$$

## Solving Linear Systems

We can solve a system by manipulating the equations in such a way that we get the value of the unknowns.

$$
\begin{aligned}
x + 2 y & = 5 \\
2 x + y & = 4
\end{aligned}
 \Rightarrow 
\begin{aligned}
x & = 1 \\
y & = 2
\end{aligned}
$$

Such manipulations are called elementary row operations. The available operations include:

- Swapping equations
    
$$
\begin{aligned}
x + 2 y & = 5 \\
2 x + y & = 4
\end{aligned}
 \Rightarrow 
\begin{aligned}
2 x + y & = 4 \\
x + 2 y & = 5
\end{aligned}
$$
    
- Adding equations.
    
$$
\begin{aligned}
x + 2 y & = 5 \\
- x - y & = - 3
\end{aligned}
 \Rightarrow 
\begin{aligned}
x + 2 y & = 5 \\
y & = 2
\end{aligned}
$$
    
- Multiplying equations.
    
$$
\begin{aligned}
x + 2 y & = 5 \\
2 x + y & = 4
\end{aligned}
 \Rightarrow 
\begin{aligned}
2 x + 4 y & = 10 \\
2 x + y & = 4
\end{aligned}
$$
    

The result of our manipulations can also indicate whether the system has a unique, inconsistent or infinite solutions.

- If the values of all unknowns are found, then the system has a unique solution.
- If the system contains $0 = 0$, then the system has infinite solutions.
- If the system contains contradictory information (for instance $0 = 2$), then the system has no solution.

To solve a system, we can use the elimination method where we eliminate an unknown from other equations iteratively until we find the value of each unknown.

$$
\begin{bmatrix}
\begin{array}{cc|c}
2 & 3 & 8 \\
3 & 1 & 5
\end{array}
\end{bmatrix} 
 \Rightarrow 
\cdots
 \Rightarrow 
\begin{bmatrix}
\begin{array}{cc|c}
1 & 0 & 1 \\
0 & 1 & 2
\end{array}
\end{bmatrix} 
$$

Despite reducing our matrix, the determinant will not change, hence the singularity is still preserved.

This method can also be done in matrix form and it is called matrix row reduction, where we try to convert the matrix into an upper diagonal. This reduced form is also called row echelon form.

$$
\begin{bmatrix}
2 & 3 & 1 \\
5 & 2 & 4 \\
3 & 1 & 2
\end{bmatrix}
 \Rightarrow 
\begin{bmatrix}
1 & * & * \\
0 & 1 & * \\
0 & 0 & 1
\end{bmatrix}
$$

There’s an even more reduced form called the reduced row echelon form, where we obtain a diagonal of the matrix instead of the upper diagonal.

$$
\begin{bmatrix}
2 & 3 & 1 \\
5 & 2 & 4 \\
3 & 1 & 2
\end{bmatrix}
\Rightarrow
\begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

To obtain both forms, we can use the Gaussian elimination algorithm. The algorithm is as follows:

1. Create the augmented matrix.
2. Forward elimination.
    1. Set an equation as the pivot.
    2. Make the pivot element into 1 by dividing the pivot equation with the pivot element itself.
    3. Eliminate all entries below the pivot by subtracting multiples of the pivot row from lower rows.
    4. Move to the next pivot equation.
    5. Repeat step 2 until the matrix is in row echelon form.
3. Back substitution (for reduced row form)
    1. Solve the last equation for its variable.
    2. Substitute the variable into the row above to solve for the next variable.
    3. Repeat upwards until the matrix is in reduced row echelon form.

![](https://i.ytimg.com/vi/5mD_CnbC7Zk/hqdefault.jpg)
