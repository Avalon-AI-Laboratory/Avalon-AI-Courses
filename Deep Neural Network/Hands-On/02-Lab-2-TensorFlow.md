# Lab 2: Applied Optimization (TensorFlow)

In this lab, we will move from "building from scratch" to using a professional deep learning framework: **TensorFlow (Keras)**.

We will apply the advanced techniques discussed in Module 2, Regularization, Optimization algorithms, and Batch Normalization to improve the performance of a Convolutional Neural Network (CNN) on the **CIFAR-10** dataset.

---

### Step 0: Setup and Data Loading

First, import the necessary libraries and prepare the data. We will use **Input Normalization** by dividing pixel values by 255.0 to scale them between 0 and 1. This speeds up training.

```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

# 1. Load the CIFAR-10 dataset
# CIFAR-10 contains 60,000 32x32 color images in 10 classes
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# 2. Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Check shape
print(f"Training data shape: {train_images.shape}")
```

### Step 1: Baseline Model (High Variance)

We start by building a standard CNN without any regularization or advanced optimization. We use Stochastic Gradient Descent (SGD).

Hypothesis: This model will likely suffer from High Variance (Overfitting)—it will learn the training data well but fail to generalize to the test data.

```python
def build_baseline_model():
    model = models.Sequential()
    # Block 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Block 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Block 3
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    # Dense Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10)) # 10 Output classes
    return model

# Build and Compile
baseline_model = build_baseline_model()
baseline_model.compile(optimizer='sgd',
                       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                       metrics=['accuracy'])

# Train
print("--- Training Baseline Model ---")
baseline_history = baseline_model.fit(train_images, train_labels, epochs=15,
                                      validation_data=(test_images, test_labels),
                                      verbose=1)
```

### Step 2: Adding Regularization (L2 + Dropout)

To fix the High Variance, we add regularization:

* **L2 Regularization**: Added to the Convolutional kernels (`kernel_regularizer=l2(0.001)`).
* **Dropout**: Added after pooling layers (`layers.Dropout(0.25)`).

```python
def build_regularized_model():
    model = models.Sequential()
    
    # Block 1 with L2
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            kernel_regularizer=l2(0.001), input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25)) # Dropout added
    
    # Block 2 with L2
    model.add(layers.Conv2D(64, (3, 3), activation='relu',
                            kernel_regularizer=l2(0.001)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25)) # Dropout added
    
    # Block 3
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    # Dense Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    return model

# Build and Compile (Still using SGD)
reg_model = build_regularized_model()
reg_model.compile(optimizer='sgd',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

# Train
print("\n--- Training Regularized Model ---")
reg_history = reg_model.fit(train_images, train_labels, epochs=15,
                            validation_data=(test_images, test_labels),
                            verbose=1)
```

### Step 3: Adding a Better Optimizer (Adam)

SGD can be slow. We will switch to **Adam**, which combines Momentum and RMSprop to converge much faster.

```python
# Re-use the regularized architecture
adam_model = build_regularized_model()

# Compile with Adam
adam_model.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])

# Train
print("\n--- Training with Adam Optimizer ---")
adam_history = adam_model.fit(train_images, train_labels, epochs=15,
                            validation_data=(test_images, test_labels),
                            verbose=1)
```

### Step 4: Adding Batch Normalization

Finally, we add **Batch Normalization**. This normalizes the inputs to the activation functions, stabilizing and speeding up training.

**Best Practice:** Place BatchNormalization after the convolution/dense layer but **before** the activation function.

```python
def build_batchnorm_model():
    model = models.Sequential()
    
    # Block 1
    model.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3))) # No activation here
    model.add(layers.BatchNormalization()) # Batch Norm
    model.add(layers.Activation('relu'))   # Activation
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Block 2
    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Block 3
    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    # Dense Block
    model.add(layers.Flatten())
    model.add(layers.Dense(64))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(10)) # Output
    
    return model

# Build and Compile
bn_model = build_batchnorm_model()
bn_model.compile(optimizer='adam',
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])

# Train
print("\n--- Training with Batch Norm Model ---")
bn_history = bn_model.fit(train_images, train_labels, epochs=15,
                          validation_data=(test_images, test_labels),
                          verbose=1)
```

### Step 5: Conclusion & Analysis

Let's plot the Validation Accuracy of all four models to compare their performance.

* **Baseline (Blue):** Likely overfits (Training acc high, Validation acc stalls).
* **Regularized (Orange):** Reduces overfitting (Gap between Train/Val is smaller), but learns slowly due to SGD.
* **Adam (Green):** Learns much faster than SGD.
* **Batch Norm (Red):** Should show the fastest convergence and potentially the best stability.

```python
plt.figure(figsize=(12, 8))
plt.plot(baseline_history.history['val_accuracy'], label='1. Baseline (SGD)')
plt.plot(reg_history.history['val_accuracy'], label='2. Regularized (SGD)')
plt.plot(adam_history.history['val_accuracy'], label='3. Regularized (Adam)')
plt.plot(bn_history.history['val_accuracy'], label='4. Full Model (BN + Adam)')

plt.title('Model Validation Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

```
