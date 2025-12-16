# Part 4: Optimization Algorithms

These algorithms help your model converge to the minimum of the cost function faster and more reliably than standard gradient descent.

### 4.1. Mini-Batch Gradient Descent
* **Batch Gradient Descent:** Processes the *entire* training set in one go to take one step. Very slow on large datasets.
* **Stochastic Gradient Descent (SGD):** Processes *one* training example at a time. Very fast per step, but very "noisy" and may never converge.
* **Mini-Batch Gradient Descent:** The best of both worlds. Splits the training set into small "mini-batches" (e.g., 64, 128, 512 samples). It takes a gradient descent step for each mini-batch.
    * **Advantage:** Allows for vectorization (fast computation) and makes faster, more stable progress.

### 4.2. Gradient Descent with Momentum
* **What it is:** Computes an "exponentially weighted average" of the past gradients and uses that average to update the weights.
* **Intuition:** Imagine a ball rolling down a hill. It builds up momentum, allowing it to move faster in the correct direction and dampen oscillations in other directions.
* **Hyperparameter:** `β` (beta), typically set to **0.9**.

### 4.3. RMSprop (Root Mean Square Prop)
* **What it is:** Also uses an exponentially weighted average, but on the *square* of the gradients.
* **Intuition:** It slows down learning in "steep" directions (where the gradient is large) and speeds it up in "flat" directions (where the gradient is small). This is very effective in elongated cost functions.
* **Hyperparameter:** `β2` (beta 2), typically set to **0.999**.

### 4.4. Adam Optimization Algorithm
* **What it is:** **Adam (Adaptive Moment Estimation)** is the most popular optimizer. It essentially **combines Momentum and RMSprop**.
* It uses exponentially weighted averages for both the gradient itself (like Momentum) and the square of the gradient (like RMSprop).
* **Hyperparameters:** `β1` (0.9), `β2` (0.999), `ε` (10e-8) are the standard defaults and work well.

### 4.5. Learning Rate Decay
* **What it is:** Gradually reducing the learning rate (`α`) as training progresses.
* **Why it works:** When starting, a large `α` helps you move quickly towards the minimum. As you get closer, a smaller `α` allows for finer, more precise steps to converge at the true minimum instead of oscillating around it.
