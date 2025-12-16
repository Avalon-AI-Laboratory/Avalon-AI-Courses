# Vectors and Linear Transformations

## Table Of Contents

- [Vectors](#vectors)
- [Dot Product](#dot-product)
- [Inner Product](#inner-product)
- [Outer Product](#outer-product)
- [Linear Transformations](#linear-transformations)
- [Matrix Inverse](#matrix-inverse)

## Vectors

Unlike scalars, vectors have magnitude and direction properties. It can be represented as a tuple of values.

$$
\mathbf{v} = 
\begin{bmatrix}
v_1 \\
v_2 \\
\vdots \\
v_n
\end{bmatrix} \text{or} 
\begin{bmatrix}
v_1 & v_2 & \cdots & v_n
\end{bmatrix}
$$

Vectors also have norms, which are like the length of the vector itself. The most common vector norms are:

- L1
    
$$
\| \mathbf{v} \|_1 = \sum | v_i |
$$
    
- L2
    
$$
\| \mathbf{v} \|_2 = \sqrt{ \sum v_i^2 }
$$
    

For addition and subtraction, it is done by adding / subtracting each component from each vector.

$$
\mathbf u \pm \mathbf v = 
\begin{bmatrix}
u_1 \pm v_1 \\
u_2 \pm v_2 \\
\vdots \\
u_n \pm v_n
\end{bmatrix}
$$

Vectors can also be multiplied by scalars, where each component of the vector is multiplied by the same scalar.

$$
a \mathbf u = 
\begin{bmatrix}
a u_1 \\
a u_2 \\
\vdots \\
a u_n 
\end{bmatrix}
$$

## Dot Product

The dot product is an operation that takes two vectors and returns a scalar. The value of the dot product can be interpreted as how aligned or how similar the two vectors are.

![](https://physicsgirl.in/wp-content/uploads/2024/05/Dot-Product-Of-Two-Vectors.jpeg)

It is defined as the following:

$$
\begin{align*}
\mathbf{u} \cdot \mathbf{v} & = \mathbf{u} ^ T \mathbf{v} \\ 
& = \sum u_i v_i \\
& = | \mathbf{u} | | \mathbf{v} | \cos \theta
\end{align*}
$$

## Inner Product

The inner product is similar to the dot products, but it can work in any abstract vector space.

$$
\begin{align*}
\langle \mathbf{u}, \mathbf{v} \rangle & = \mathbf{u} ^ T \mathbf{v} \\ & = \sum u_i v_i
\end{align*}
$$

However, it is also has several properties that it must satisfy. In a vector space $V$ over the field $F$, for all vectors $\mathbf{x}, \mathbf{y}, \mathbf{z} \in V$ and all scalars $\mathbf{a}, \mathbf{b} \in F$, it must satify the following properties:

- Conjugate Symmetry

$$
\langle \mathbf{x}, \mathbf{y} \rangle = \overline{ \langle \mathbf{y}, \mathbf{x} \rangle }
$$

- Linearity

$$
\langle a \mathbf{x} + b \mathbf{y}, \mathbf{z} \rangle = a \langle \mathbf{x}, \mathbf{z} \rangle + b \langle \mathbf{y}, \mathbf{z} \rangle
$$

- Positive-definiteness

$$
\langle \mathbf{x}, \mathbf{x} \rangle > 0
$$

## Outer Product

The outer product is an operation that takes two vectors and returns a matrix instead. Outer products are matrices whose entries are all products of an element in the first vector with an element in the second vector. 

$$
\begin{align*}
\mathbf{u} \otimes \mathbf{v} & = \mathbf{u} \mathbf{v} ^ T \\ & = u_i v_j
\end{align*}
$$

It can be used to capture all possible pairwise interactions between components of the two input vectors.

## Linear Transformations

Linear transformations are functions that transforms one point into another point in a very structured way. These transformations are usually described in the form of matrix multiplications. 

$$
T \mathbf u = \mathbf v
$$

![](https://tse4.mm.bing.net/th/id/OIP.pJCpz8JzQtXfoVembE-RtgHaFj?rs=1&pid=ImgDetMain&o=7&rm=3)

Moreover, multiple transformations can be combined.

$$
T_n T_{n - 1} \cdots T_1 \mathbf u = \mathbf v
$$

However, the order of the transformations affects the result.

$$
A = 
\begin{bmatrix}
1 & - 1 \\
-1 & 1
\end{bmatrix} \ B = \begin{bmatrix}
-3 & 5 \\
2 & 1
\end{bmatrix} 
$$
$$
A B = \begin{bmatrix}
-5 & 4 \\
5 & - 4
\end{bmatrix} \ B A = \begin{bmatrix}
-8 & 8 \\
1 & - 1
\end{bmatrix}
$$
$$
A B \ne B A
$$

Linear transformations also have ranks and singularity. It is all determined by the number of dimensions at the result.

![](https://advaitha.github.io/myDataScienceJourney/Images/rank_and_dimensions_linear_transform.png)

## Matrix Inverse

Matrices can also have an inverse just like ordinary numbers. It is usually used as an “undo” for matrix transformations.

$$
\begin{cases}
2 \times \frac{1}{2} = \frac{1}{2} \times 2 = 1 & \text{Ordinary numbers} \\
A A ^{-1} = A ^ {-1} A = I & \text{Matrices}
\end{cases}
$$

The inverse of a matrix is:

$$
A ^ {- 1} = \frac{ 1 }{ \det ( A ) } 
\begin{bmatrix}
d & -b \\
-c & a
\end{bmatrix}
$$

For larger matrices, we can use the Gaussian Elimination method, but instead of the usual augmented matrix $[ A | B ]$, we use $[ A | I ]$ . The goal is to transform the matrix in such a way so that it becomes $[ I | A ^ {-1} ]$.

$$
\begin{bmatrix}
\begin{array}{c c c | c c c}
1 & 2 & 3 & 1 & 0 & 0 \\
0 & 1 & 4 & 0 & 1 & 0 \\
5 & 6 & 0 & 0 & 0 & 1
\end{array}
\end{bmatrix}
\Rightarrow 
\begin{bmatrix}
\begin{array}{c c c | c}
1 & 0 & 0 & -24 & 18 & 5 \\
0 & 1 & 0 & 20 & -15 & -4 \\
0 & 0 & 1 & -5 & 4 & 1
\end{array}
\end{bmatrix}
$$

It’s important to note that not all matrices have an inverse since $\det ( A )$ can be $0$.
