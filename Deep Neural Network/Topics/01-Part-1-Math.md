# Part 1: Math Foundations (The "Why")

Before we build, we need to understand the tools. Deep learning is built on a few key mathematical concepts.

### 1.1. Derivatives
* **What it is:** A derivative measures the "slope" or rate of change of a function.
* **Why it matters:** It tells us how to "nudge" a parameter (like a weight) to make our model's error go down. If the derivative (gradient) is positive, we decrease the weight; if it's negative, we increase it. This is the core of **gradient descent**.

### 1.2. Partial Derivatives
* **What it is:** A derivative of a function that has *multiple* input variables. A partial derivative measures the slope with respect to just *one* of those variables, while holding all others constant.
* **Why it matters:** A neural network has thousands of parameters (weights and biases). To update a single weight, we need to find the partial derivative of the cost function with respect to that *one* weight.

### 1.3. The Gradient
* **What it is:** The gradient is a vector (or a list) containing *all* the partial derivatives of a function.
* **Why it matters:** It represents the "direction of steepest ascent" or the direction to change all parameters to *increase* the cost the fastest. To *decrease* the cost, we just move in the exact opposite direction. This is **Gradient Descent**.

### 1.4. The Chain Rule
* **What it is:** A rule for finding the derivative of a composite function (a function inside another function, e.g., `f(g(x))`).
* **Why it matters:** A neural network is a *massive* composite function. The output depends on Layer 3, which depends on Layer 2, which depends on Layer 1. The **Chain Rule** is the fundamental mechanism that allows us to calculate the gradients for all layers, starting from the output and moving backward. This process is called **Backpropagation**.

### 1.5. Jacobian & Matrix Calculus
* **What it is:** The Jacobian is a matrix containing all the partial derivatives of a *vector-valued* function (a function that outputs multiple values).
* **Why it matters:** In our network, the cost is a single number, but the gradients for a layer (`dW`, `db`) are matrices/vectors. Matrix calculus (using Jacobians/Gradients) is simply the notation and set of rules we use to apply the Chain Rule efficiently to matrices instead of calculating one derivative at a time.
