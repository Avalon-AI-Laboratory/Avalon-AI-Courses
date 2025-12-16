# Lab 1: Neural Network from Scratch (NumPy)

In this lab, we will build a **2-layer neural network** from scratch to classify a simple **"moons" dataset"**. This exercise will strengthen your understanding of **Forward Propagation** and **Backward Propagation**, without using deep learning frameworks like TensorFlow or PyTorch.

---

## 🚀 Step 0: Setup and Data Generation

We start by importing libraries and generating a dataset using `sklearn`. The *moons dataset* is a classic non-linear dataset—perfect for testing neural networks.

```python
import numpy as np
import sklearn
import sklearn.datasets
import matplotlib.pyplot as plt

# Generate a "moons" dataset
X, Y = sklearn.datasets.make_moons(n_samples=200, noise=0.20)
# Reshape Y to be a row vector (1, m)
Y = Y.reshape(1, Y.shape[0])

# Visualize the data
plt.scatter(X[:,0], X[:,1], c=Y[0], s=40, cmap=plt.cm.Spectral);
plt.title("Our Moons Dataset")
plt.show()

# Get the shapes of our data
shape_X = X.shape
shape_Y = Y.shape
m = shape_X[0]  # number of training examples

print(f"X shape: {shape_X}")
print(f"Y shape: {shape_Y}")
print(f"Number of examples (m): {m}")
```

---

## 🧩 Step 1: Define Layer Sizes & Initialize Parameters

We will build a simple 2-layer network:

* **Input Layer:** 2 neurons (x, y)
* **Hidden Layer:** 4 neurons
* **Output Layer:** 1 neuron (binary output)

```python
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)
    
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters

# Example usage
n_x = shape_X[1]
n_h = 4
n_y = shape_Y[0]
```

---

## 🔢 Step 2: Activation Function

We will use:

* **tanh** for hidden layer
* **sigmoid** for output layer

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

---

## 🔮 Step 3: Forward Propagation

Compute activations and store intermediate values.

```python
def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    Z1 = np.dot(W1, X.T) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache
```

---

## 🔁 Step 4: Backpropagation

Compute gradients using the chain rule.

```python
def backward_propagation(parameters, cache, X, Y):
    m = X.shape[0]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * (1 - np.power(A1, 2))
    dW1 = (1/m) * np.dot(dZ1, X)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return grads
```

---

## 🏋️ Step 5: Training Loop (Gradient Descent)

Combine everything into the main training function.

```python
def update_parameters(parameters, grads, learning_rate=1.2):
    W1 = parameters["W1"] - learning_rate * grads["dW1"]
    b1 = parameters["b1"] - learning_rate * grads["db1"]
    W2 = parameters["W2"] - learning_rate * grads["dW2"]
    b2 = parameters["b2"] - learning_rate * grads["db2"]
    
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    np.random.seed(3)
    n_x = X.shape[1]
    n_y = Y.shape[0]
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    for i in range(num_iterations):
        A2, cache = forward_propagation(X, parameters)
        
        logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), (1 - Y))
        cost = - (1 / m) * np.sum(logprobs)
        
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads)
        
        if print_cost and i % 1000 == 0:
            print(f"Cost after iteration {i}: {cost}")
            
    return parameters

# Train the model
print("Training Model...")
parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)
```

---

## 🎯 Step 6: Visualize Predictions

Plot decision boundary.

```python
def predict(parameters, X):
    A2, _ = forward_propagation(X, parameters)
    return (A2 > 0.5)

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()

# Plot
plot_decision_boundary(lambda x: predict(parameters, x), X, Y[0])
```

---