# Part 3: Practical Aspects of Deep Learning

This section covers the foundational setup of any machine learning project: structuring your data, diagnosing problems, and applying standard regularization techniques.

### 3.1. Train / Dev / Test Sets

Structuring your data correctly is the first step to building a successful model.

* **Training Set:** The largest set, used to train your model's parameters.
* **Development (Dev) Set:** Also called the "hold-out cross-validation set." You use this set to tune your hyperparameters (like learning rate, network size, etc.).
* **Test Set:** This set is used only *once* at the very end to get an unbiased estimate of your model's final performance.

**Splitting Ratios:**
* **Traditional (e.g., < 1,000,000 examples):** 60% Train / 20% Dev / 20% Test.
* **Big Data Era (e.g., > 1,000,000 examples):** 98% Train / 1% Dev / 1% Test. Your dev and test sets just need to be large enough to give statistical confidence.

**Golden Rule:** Make sure your **Dev and Test sets come from the same distribution**. They should reflect the data you expect to see in the real world.

### 3.2. Bias / Variance

Diagnosing your model's errors is crucial.

* **Bias (Underfitting):** The model is too simple and doesn't even perform well on the training data.
    * **Symptom:** High Training Error. (e.g., Training Error: 15%, Dev Error: 16%).
* **Variance (Overfitting):** The model is too complex and "memorizes" the training data, but fails to generalize to new, unseen data (the dev set).
    * **Symptom:** Low Training Error, but High Dev Error. (e.g., Training Error: 1%, Dev Error: 11%).

### 3.3. The Basic Recipe for Machine Learning

Based on your diagnosis, follow this recipe:

1.  **Does your model have High Bias?**
    * If yes, try:
        * Bigger network (more layers/neurons).
        * Train longer.
        * Different optimization algorithm (e.g., Adam).
2.  **Does your model have High Variance?**
    * If yes, try:
        * Get more data.
        * **Regularization** (L2, Dropout).
        * Different network architecture.

You repeat this process until both bias and variance are low.

### 3.4. Regularization to Reduce Variance

Regularization techniques help prevent overfitting.

#### L2 Regularization
* **What it is:** Adds a penalty term to the cost function based on the "Frobenius norm" (squared sum) of the weights.
* **Cost Function:** `J_reg = J_original + (λ / 2m) * Σ||W[l]||^2`.
* **Why it works (Intuition):** It penalizes large weights, pushing them closer to zero. This makes the model "simpler" and less prone to fitting the noise in the training data, resulting in a smoother decision boundary. This effect is also called **"weight decay"**.

#### Dropout Regularization
* **What it is:** At each training iteration, randomly "drops" (sets to zero) a fraction of neurons in a layer. The fraction is defined by a `keep_prob` hyperparameter.
* **Why it works (Intuition):** A neuron cannot rely on any single feature from the previous layer, so it is forced to learn more robust features by "spreading out the weights". It's like training a smaller, different network at every iteration.
* **Implementation:** Use **Inverted Dropout**. This technique scales the activations of the remaining neurons by `1 / keep_prob` during training. This ensures that the expected value of the activations remains the same, so no scaling is needed at test time.
* **Important:** Only apply Dropout during **training**. Never use it during **testing**.

#### Other Regularization Methods
1.  **Data Augmentation:** Create "new" training data by applying realistic transformations to your existing data (e.g., flipping, random cropping, rotating, or color shifting images).
2.  **Early Stopping:** Plot your Dev Set error against the number of training iterations. Stop training at the point where the Dev Set error begins to rise, as this indicates overfitting has started.

### 3.5. Optimizing the Training Process

#### Input Normalization
* **What it is:** Standardizing your input features to have a mean of 0 and a variance of 1.
* **Steps:**
    1.  Subtract the mean (`μ`) from the data: `X = X - μ`.
    2.  Divide by the variance (`σ^2`): `X = X / σ^2`.
* **Why it works:** It speeds up training significantly. If features are on different scales, the cost function becomes "elongated," and gradient descent will oscillate. Normalization makes the cost function more symmetrical, allowing for a larger learning rate and faster convergence.

#### Vanishing / Exploding Gradients
* **What it is:** In very deep networks, gradients can become extremely small (vanish) or extremely large (explode) as they are backpropagated. This makes training slow or unstable.
* **Solution: Weight Initialization.** The choice of initial weights is critical.
    * **Rule of thumb:**
        * For **ReLU** activation: `W[l] = np.random.randn(shape) * np.sqrt(2 / n[l-1])` (**He Initialization**).
        * For **Tanh** activation: `W[l] = np.random.randn(shape) * np.sqrt(1 / n[l-1])` (**Xavier Initialization**).

### 3.6. Efficient Backpropagation: The Chain Rule & Vectorization

Understanding *how* we calculate gradients efficiently is key to training deep networks.

#### The Chain Rule
* **What it is:** A formula for computing the derivative of the composition of two or more functions.
* **In Deep Learning:** A Neural Network is essentially a massive composite function: `Output = Layer3(Layer2(Layer1(Input)))`.
* **How it works:** To find how a weight in the first layer affects the final error (Cost), we multiply the derivatives (slopes) of every layer in between.
    * `dCost/dWeight1 = (dCost/dOutput) * (dOutput/dLayer2) * (dLayer2/dLayer1) * ...`
* **Backpropagation:** This is simply the Chain Rule applied sequentially from the last layer backward to the first layer.

#### Vectorization
* **The Problem:** Calculating gradients for 1,000 training examples using a `for-loop` is incredibly slow.
* **The Solution:** **Vectorization**. Instead of looping, we stack all training examples into big matrices (e.g., Input `X` has shape `(features, m_examples)`).
* **How it works:** We use Linear Algebra (Matrix Multiplication).
    * Instead of `w * x` (scalar), we do `np.dot(W, X)` (matrix).
    * This allows the computer (CPU/GPU) to calculate the gradients for **all examples at the exact same time** in parallel.
* **Key Takeaway:** Always use vectorized operations (like in NumPy or TensorFlow) instead of explicit loops. This is the single most important factor for training speed.
