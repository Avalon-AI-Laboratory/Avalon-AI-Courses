# Part 5: Hyperparameter Tuning & Batch Normalization

### 5.1. The Tuning Process
* **Hyperparameters:** These are the parameters you set *before* training, like learning rate (`α`), momentum (`β`), mini-batch size, number of layers, etc.
* **Priority:**
    1.  **Most Important:** Learning Rate (`α`).
    2.  **High Importance:** Momentum (`β`), mini-batch size, number of hidden units.
    3.  **Medium Importance:** Number of layers, learning rate decay.
* **Best Practice:**
    * **Don't use a grid search.** It's inefficient.
    * **Use Random Search.** Pick values randomly within a valid range. This is much more effective at finding good combinations.
    * **Use an Appropriate Scale:** For `α`, search on a log scale (e.g., 0.0001, 0.001, 0.01, 0.1) instead of a linear scale.

### 5.2. Batch Normalization (Batch Norm)
* **What it is:** Batch Norm is a powerful technique that normalizes the *activations* (the `Z[l]` values) inside the network, right before the activation function.
* **How it works:**
    1.  For a mini-batch, it calculates the mean and variance of `Z[l]`.
    2.  It normalizes `Z[l]` using this mean and variance.
    3.  It then scales and shifts the result using two new *learnable* parameters, `γ` (gamma) and `β` (beta), allowing the network to decide the new mean and variance.
* **Why it works:**
    1.  **Speeds up training:** By normalizing activations, it makes the cost function "easier" to optimize, similar to input normalization.
    2.  **Reduces "Internal Covariate Shift":** Makes each layer more independent.
    3.  **Regularization Effect:** Adds a slight amount of noise (since the mean/variance are per-batch), which has a small regularization effect.
* **Important:** At test time, you use an "exponentially weighted average" of the means/variances calculated during training.

### 5.3. Softmax Regression
* **What it is:** A generalization of logistic regression used for **multi-class classification** (when you have more than two classes, e.g., cat, dog, or bird).
* It outputs a probability distribution across all `C` classes, ensuring all probabilities sum to 1.
