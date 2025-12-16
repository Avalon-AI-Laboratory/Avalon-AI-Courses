# Determinants and Eigenvectors

## Table Of Contents

- [Determinant as Areas](#determinant-as-areas)
- [Determinant of Inverses](#determinant-of-inverses)
- [Basis & Spans](#basis--spans)
- [Eigenbases, Eigenvalues and Eigenvectors](#eigenbases-eigenvalues-and-eigenvectors)
- [Dimensionality Reduction](#dimensionality-reduction)

## Determinant as Areas

Determinant can also be seed as an area or volume formed by all order of basis vector linear combinations. The area of the determinant can also be negative due to vector orientations.

![](https://advaitha.github.io/myDataScienceJourney/Images/determinant_as_area.png)

## Determinant of Inverses

Coincidentally, the determinant of the inverse of a matrix is:

$$
\det (A^{-1}) = \frac{1}{\det (A)}
$$

## Basis & Spans

A basis is a set of linearly independent vectors that can form any other vector in the same space. Those other vectors can be obtained by linear combinations of the basis.

![](https://th.bing.com/th/id/R.138b419c70a6e98c62841a9ea7d9065b?rik=YuZJ9Vr3ghRuDQ&riu=http%3a%2f%2fupload.wikimedia.org%2fwikipedia%2fcommons%2fthumb%2ff%2ff4%2f3d_two_bases_same_vector.svg%2f130px-3d_two_bases_same_vector.svg.png&ehk=qN6HfP6UTRpuets0fI9nOWJ%2f6VIMRsyrNFo8LjCVLPc%3d&risl=&pid=ImgRaw&r=0)

For example, in $\mathbb R ^2$, the vectors:

$$
\mathbf e_1 = \begin{bmatrix}
1 \\ 
0
\end{bmatrix}
, \,
\mathbf e_2 = \begin{bmatrix}
0 \\
1
\end{bmatrix}
$$

form a basis.

The set of all linear combination of the basis is called a span. For example, the vector:

$$
\mathbf v = \begin{bmatrix}
2 \\ 
3
\end{bmatrix}
$$

is a span as it can be defined as a linear combination of:

$$
\mathbf v = 2 \mathbf e_1 + 3 \mathbf e_2
$$

## Eigenbases, Eigenvalues and Eigenvectors

When we transform vectors, there is chance that we rotate the vector. However, some vectors only scale and don’t rotate (or rotate $180 ^ \circ$ due to negative scaling). 

![](https://bookdown.org/marktrede/linalg/gifs/eigenvectors.gif)

These special vectors are called eigenvectors and those scales are called eigenvalues. Like basis, the set of these eigenvectors form an eigenbasis.

We can find eigenvectors, we first have to find the eigenvalues with this equation:

$$
\det ( A - \lambda I ) = 0
$$

And after obtaining the eigenvalues, we can find the eigenvectors.

$$
A \mathbf v = \lambda \mathbf v
$$

## Dimensionality Reduction

Dimensionality reduction is the process of reducing the number of features used to represent the data while preserving as much important information as possible. It is usually used to compress data and visualize patterns in lower dimensions.

One way to do is by using Principal Component Analysis (PCA). PCA finds new basis vectors that are perpendicular to each other yet capture the maximum variance in the data. The data will then be reprojected to the new basis.

![](https://builtin.com/sites/www.builtin.com/files/inline-images/national/Principal%2520Component%2520Analysis%2520second%2520principal.gif)

To do so, the PCA algorithm includes the following steps:

1. Center the data.
    
$$
\bar X = X - \mu
$$
    
2. Compute the covariance matrix.
    
    Note: The covariance matrix will be discussed in the Probability and Statistics chapter.
    
$$
C = \text{Cov} (\bar X, \bar X) = \frac{1}{N} \bar X ^T \bar X
$$
    
3. Compute eigenvalues and eigenvectors.
    
$$
C \mathbf u = \lambda \mathbf u
$$
    
4. Sort eigenvectors based of their eigenvalues in descending order.
    
$$
\lambda_1 > \lambda_2 > \dots > \lambda_n
$$
    
5. Select the principal components.
    
$$
U_k = \begin{bmatrix}
\mathbf u_1 & \mathbf u_2 & \cdots & \mathbf u_k
\end{bmatrix}
$$
    
6. Project the data.
    
$$
Z = \bar X U_k
$$
    
7. (Optional) Reconstruct the data.
    
$$
\hat X = ZU_k^T + \mu
$$