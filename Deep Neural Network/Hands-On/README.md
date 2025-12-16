# Hands-On Labs: Optimization & Regularization

Welcome to the practical section of Module 2!

In this section, we will move from theory to practice. We have designed two specific labs to help you solidify your understanding of how Deep Neural Networks actually learn and how to make them perform better using the techniques discussed in the material.

## Available Labs

### 1. [Lab 1: Neural Network from Scratch (NumPy)](01-Lab-1-From-Scratch.md)
* **Goal:** Demystify the "Black Box".
* **What you will do:** Build a 2-layer neural network without using any deep learning frameworks (like TensorFlow or PyTorch). You will implement **Forward Propagation** and **Backpropagation** manually using Python and NumPy.
* **Why it matters:** Understanding the calculus and algebra behind backpropagation is the single most important step to becoming a true Deep Learning practitioner, not just a library user.

### 2. [Lab 2: Applied Optimization (TensorFlow)](02-Lab-2-TensorFlow.md)
* **Goal:** Master the Modern Toolkit.
* **What you will do:** Use **TensorFlow/Keras** to build a Convolutional Neural Network (CNN) for image classification (CIFAR-10).
* **Key Techniques:** You will apply the C2 concepts:
    * **Regularization:** L2 and Dropout to stop overfitting.
    * **Optimization:** Switch from SGD to **Adam** for speed.
    * **Normalization:** Implement **Batch Normalization** for stability.

---

## How to Use These Labs

1.  **Read the Guide:** Click on the lab links above to read the step-by-step walkthrough directly here in GitBook.
2.  **Run the Code:**
    * You can copy-paste the code blocks into a **Google Colab** notebook.
    * Alternatively, if you have cloned this repository, you can find the ready-to-run Jupyter Notebooks (`.ipynb`) in the `notebooks/` folder.

**Ready? Let's code! Start with [Lab 1](01-Lab-1-From-Scratch.md).**