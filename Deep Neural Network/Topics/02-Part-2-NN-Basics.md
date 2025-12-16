# Part 2: Neural Network Basics (The "What")

Now we use the math to build our first model.

### 2.1. The Perceptron (Logistic Regression)
* **What it is:** The simplest form of a neural network, consisting of a single "neuron."
* **How it works:**
    1.  It takes all inputs ($x$).
    2.  Multiplies each input by a weight ($w$).
    3.  Sums them up and adds a bias ($b$) to get `z = w*x + b`.
    4.  Passes the result `z` through an **activation function** (like Sigmoid) to produce a prediction `a`.

### 2.2. Network Representation
* **What it is:** A full neural network is just many perceptrons stacked into "layers."
    * **Input Layer:** Your data.
    * **Hidden Layers:** The layers in the middle that learn complex patterns.
    * **Output Layer:** Produces the final prediction.
* **Parameters:** The "knobs" the network learns are all the weights (`W`) and biases (`b`) in each layer.

### 2.3. Forward Propagation
* **What it is:** The process of making a prediction.
* **How it works:** We pass the input data "forward" through the network, layer by layer.
    1.  `Z[1] = W[1] * X + b[1]`
    2.  `A[1] = g(Z[1])` (where `g` is an activation function like ReLU)
    3.  `Z[2] = W[2] * A[1] + b[2]`
    4.  `A[2] = g(Z[2])`
    5.  ...until we get the final output `A[L]`.

### 2.4. The Cost Function
* **What it is:** A function that measures how "wrong" the model's prediction (`A[L]`) is compared to the true answer (`Y`).
* **Why it matters:** The goal of training is to find the weights and biases that *minimize* this cost function. A common example is **Cross-Entropy Loss**.

### 2.5. Backpropagation (The Algorithm)
* **What it is:** The "from scratch" algorithm that trains the network. It is the practical application of the Chain Rule.
* **How it works:**
    1.  Do **Forward Propagation** to make a prediction and calculate the cost.
    2.  Start at the end: Calculate the derivative of the cost with respect to the output `A[L]`.
    3.  Use the **Chain Rule** to move "backward" one layer: Calculate the gradients for `dW[L]`, `db[L]`, and `dA[L-1]`.
    4.  Repeat this process, passing the gradient `dA[L-1]` backward to Layer `L-2`, and so on, until you have all the gradients (`dW`, `db`) for every layer.
    5.  **Update Parameters:** Adjust all weights and biases using the gradients (this is Gradient Descent).
    6.  Repeat this entire loop thousands of times.
