# 1. Introduction from LeCun's paper 
# "Gradient Based Learning Applied to Document Recognition"
# (Why we should use learning instead of manual rules)

## Learning from Data (The Goal)
one of the most successful approaches for automatic machine learning often called gradient-based learning. When learning machine computes a function 

$$Y^p = F(Z^p, W)$$

where $Z^p$ is an input (e.g., an image of the digit "7" represented as pixel matrix), $W$ as a "adjustable parameters" (this is the "looking for"). and of course $Y^p$ is the output (or the recognized class label of pattern $Z^p$). A loss
function 

$$ E_p = D(D^p, F(W, Z^p)) $$

the D()'s part meant to measures the discrepancy/distance between $D^p$ (the actual label/ground truth of $Z^p$),and the output produced by the system. the main goal of this algorithm is to minimaze $E_{\text{train}}$ which is the average of the errors $E^p$ over the training dataset ${(Z_1, D_1), ..., (Z_P, D_P)}$
. in more simple way to say, the learning is trying in finding the value of $W$ that minimize $E_{\text{train}}(W)$. In practice, performance is estimated by measuring accuracy on both the training set and the test set, many experiments has shown that the gap between the test error and training error $E_{\text{test}} - E_{\text{train}}$ decreases as the number of training samples ($P$) increases, following the approximation:

$$E_{test} - E_{train} \approx k(h/P)^\alpha$$

where $P$ is the number of training samples, here $h$ represents the "effective capacity" or complexity of the machine (reffering to the model's flexibility in defining a decision boundary),the parameter $\alpha$ typically ranges number between 0.5 and 1.0, and k is a constant. 

using an overkill (too complex) machine can actually backfire leading to not very good performance. as the value of $h$ increase, the machine tends to just memorize the data instead of learning from it (overfitting). And the trade-off even when $h$ increase, the $E_{train}$ will decrease, but distance/gap of error from train and test will increase. thats why from  the equation above, explain the only way to decrease the gap is to add more of the datas ($P$) or we control the capacity of our machine ($h$) 

So by that, Most learning algorithms aim to minimize $E_{train}$ while simultaneously narrowing the generalization gap. or the polite way of this is called "Structural Risk Minimization" (SRM), SRM is implemented by defining a basic models then increasing it capacity. where each complex model's parameter space is a superset of its simpler predecessor. so for implementing this, LeCun suggests us by minimizing the cost function $E_{train} + \beta H(W)$, where the $H(W)$ called a regularization function, and $\beta$ is a penalty constant.  The function $H(W)$ penalizes high-capacity parameter configurations. By minimizing this term, we effectively restrict the accessible parameter space, thereby controlling the trade-off between achieving a low training error and minimizing the expected gap between training and test performance.

## Gradient-Based Learning (The "How")
The problem however, minimizing a function with respect to a set of parameters is hard. from Gradient Based Learning, we know that it is generally more easyier to minimize a reasonably smooth, contious function than a discrete function. 

The loss function can be minimized by estimating the impact of small change of the parameter values on the loss function. measured by the gradient of the loss function with respect to the parameters. we can call the algorithm is an efficient when the gradien's vector can be computed analytically (the opposite of numerically perturbations)

```
wait, lets talk about this 2 gradient's Analytic vs Numeric

- Numeric (perturbations): we try to change W by a little, we look the result,
and try it again. which very slow for a millions++ parameter.

- Analitic: analytic gradient gives you the exact direction (slope) instantly using math, rather than guessing bit by bit
```

This became the basis of numerous gradient-based learning with continuous-valued parameters. LeCun said in this article "the set parameters W is a vector using a real-valued, so that $E(W)$ function is continuous, and differentiable almost in every domain. the simpliest minimization procedure called "gradient descent algorithm", where $W$ adjusted in iteratively with this equation:'

$$W_k = W_{k-1} - \epsilon \frac{\partial E(W)}{\partial W}$$

the $\epsilon$ is a scalar constant (learning rate), in another procedur $\epsilon$ is substituted with a diagonal matrix, or with an estimate of the inverse hessian matrix, like how in Newton or quasi-Newton methods, at the end, the use of second-order methods (Hessian) LeCun said "it was very limited".

stochastic gradient algorithm, sometimes called the on-line update (because how it weight updated in every iteration). It consists in updating the parameter vector using a noisy or approximated version, of the average gradient. so the point is $W$ is updated on the basis of a single sample:

$$W_k = W_{k-1} - \epsilon \frac{\partial E^{p_k}(W)}{\partial W}$$

the main difference with the first equation (Batch Gradient Descent) is $E(W)$ means an average loss (meaning the gradient is computed over all of the training dataset). while the second equation (Stochastic Gradient Descent (SGD)) with $E^{p_k}(W)$ we only updates the parameters using the gradient of a single training example, $p_k$ meant randomly sampled training example or so a mini-batch at iteration $k$. Although the stchastic gradient is noisy, it is computationally efficient and often leads to better generalization, making it more suitable for large-scale neural networks. 
**at the end, noise is a feature, not a bug**

SGD's gradient:

$$\nabla E^{p_k}(W) \neq \nabla E(W)$$

but:

$$\mathbb{E}[\nabla E^{p_k}(W)] = \nabla E(W)$$

This means that, when averaged over many iterations, the SGD gradient converges to the same gradient computed by Batch Gradient Descent, which uses the entire training dataset.

## Gradient BackPropagation
Gradient-Based Learning is a good way to a machine to learn and it widely used already, but in early era, the used are limited to linear systems (the $y = W_x + b$'s functions). based on what LeCun said, the importances of gradient descent was not very clear before the three events occured. which
<ol>
<li> While theoretically feared, it was discovered that local minima are not a major obstacle in practice. Success in early models like Boltzmann machines proved that gradient-based learning could navigate complex loss landscapes effectively.

<li> The Rise of Backpropagation: The popularization of the backpropagation algorithm provided a mathematically simple and computationally efficient way to calculate gradients across multiple layers, replacing more cumbersome methods like "virtual targets."

<li> Proven Success on Complex Tasks: The actual demonstration that multilayer networks (using sigmoidal units) could solve difficult, real-world problems proved that the combination of Gradient Descent and Backpropagation was the right path forward.
</ol>

The core idea of back-propagation is "gradients can be computed efficiently by propagation from output back to the input". But back then people seem doesnt used it, they dont use backprog but rather something called "virtual targets" for units in intermediate layers. But when LeCun wrote this paper, backprog is already the most widely used neural-network learning algorithm 

## Learning in Real Handwriting Recognition Systems
⚠️ This part is mostly about talking a well known problem, but if you want to read it, sure. because later LeCun will test his CNN architecture (LeNet-5), with this problem too anyway.

the popular "Optical Character Recognition" (OCR) task which task given to a machine to recognising an Isolated handwritten character has been studied in many literature. but neural network trained with Gradient-Based Learning perform the best. and the best neural networks is "Convolutional Network", that are designed to learn to extract relevant features directly from pixel images. Well because of this old paper, the problems are not only just recognise individual characters, but also we have to separate our character from other (when we talk about a word/sentences) and this what weve known with "segmentation". but we also have an approach for this kind of problem known as "Heuristic Over-Segmentation". It will generating a large number of potential cuts between characters with the use of heuristic image processing techniques. and later selecting the best combination of cuts based on the scores that given by the recognizer.

The downsides are the accuracy depends on the cuts and the ability of the recognizer to distinguish correctly. and training the recognizer alone already a hard part of the process. Heuristic over-segmentation produces many incorrect and ambiguous character fragments, making it extremely difficult to create a large, consistent, and well-labeled training dataset for the recognizer. As a result, the recognizer struggles to learn reliable distinctions between valid characters and incorrectly segmented fragments. So whats the proposed solutions?

    1. The first solution, basically training the machine at the whole strings of characters (not just at character level). back again, the Gradient-Based Learning can be used for this purpose. yada yada again system trained to minimize the overall loss function which measures the probability of an erroneous answer. later on the paper's Section V we will explores various ways to ensure that the loss function is differentiable, so that will automically lends to the use of Gradient-Based Learning methods so basically the machine is making many possible intrepatation (means the important now is just the output) which later will be corrected by loss function thats why the use of gradient. and later on the Graph Transformer Network (GTN) will be used. GTN have a form of node (position) and an edge (possible character + score). then the loss will counted from the end of a graph (Global Optimisation), and yeah basically how backprog works, it will flows all the way to all the part of a graph (end-to-end for OCR). all we need just an image and the label (the text inside of the image to machine to learn) 

    2. We eliminate the segmentation entirely. forget about making a possible cuts, we know slides the "recognizer" to all part of the image. this part can imagine like a sliding windows, and in every position machine try to spot a character (character spotting). later recognizer have to recognising a character or just ignore if no character there. the result will be a possible word in every position. after that we pass it to the GTN again, which GTN will consider the most sense word kinda works like "Hidden Markov Model" (HMM). this tecnique would be expensive, but thats why LeCun gonna use Convolutional Neural Networks (CNN). to help with the computational cost. 

    LeCun’s key insight was to avoid explicit segmentation by training the system end-to-end at the sequence level using gradient-based learning and graph-based representations.

## Globally Trainable Systems (The Breakthrough)
as was already told before, pattern recognition are "using" many other techniques. like the field locator (for extracting the Region Of Interest (ROI), for finds the text), the field segmenter (for cutting the input image into many possibillities of candidate character, for cuts letters), the recognizer (for the classifier and giving scores for each candidate of possibillities, for identifies letters), and the contextual post-processor (for checks spelling), based on a stochastic grammar, so the grammar that got pick, is the best grammatically correct based by the highest scores generated by the recognizer.

Basically, this how they work
![OldSchool](Asset/OldSchool.png)

And this not so good to do it, so the better alternative would be "why dont we train the whole system", Instead of training each component separately, a better approach is to train the entire system end-to-end. The goal is to minimize a global error measure, like as character misclassification errors at the document level  (e.g, confusing l with i). honestly, in practice all what we want is to find a set of parameter $W$ that will minimizes a global loss function ($E$) defined over the whole system, if this loss function is differentiable with respect to the parameters $W$, we can compute gradients and use gradient-based learning methods (what we already discuss above) to iteratively update the parameters and reach a local minimum of the loss function.

for the global loss function $E^p (Z^p, W)$ is differentiable, the rules are we must built the system is feed-forward network composed of differentiable modules. each module implements a function that must be continuous and differentiable almost everywhere with respect to its internal parameters (like Weights ($W$) and biases ($B$)) and with respect to its inputs. this condition ensures that gradient of the loss function with respect to all parameters can be computed using back-propagation

Back to the equation we already discuss (for the sake of strengthen the understanding of the topic). because under this formulation, the system can be seen as cascade (chain/step by step) of modules, where each module implements a function

$$X_n = F_n(W_n, X_{n-1})$$

where again, $X_n$ represents the output vector of the module, $W_n$ is the vector of tunable (trainable) parameters of the module, meaning paramaters whose values are updated during training, and it is a subset of the global parameter set $W$ (means we dont use all of the parameters), and $X_{n-1}$ is the input of the vector of the module, which is also the output of the previous module, and so the first input ($X_0$), got the input from the data it self ($Z^p$, $Z$ means a raw data, and $p$ means the index of data sample)

if we have known the partial derivative of loss function $E^p$ with respect to the module output $X_n$, then we also can compute the gradients of the loss function with respect to the module parameters $W_n$ and the module input $X_{n-1}$ as

$$ \frac{\partial E^p}{\partial W_n} = \frac{\partial F}{\partial W}(W_n, X_{n-1}) \frac{\partial E^p}{\partial X_n} $$

$$ \frac{\partial E^p}{\partial X_{n-1}} = \frac{\partial F}{\partial X}(W_n, X_{n-1}) \frac{\partial E^p}{\partial X_n} $$

Here, $\frac{\partial F}{\partial W}(W_n, X_{n-1})$ and $\frac{\partial F}{\partial X}(W_n, X_{n-1})$ denote the jacobians of the module function with respect to the parameters and inputs. The jacobian of a vector-valued function is a matrix containing the partial derivatives of all outputs with respect to all inputs

the first equation computes part of the gradient of the loss with respect to the model parameters, while the second equation propagates gradient backward through the system, forming a backward reccurence equivalent to the standard back-propagation procedure used in neural networks. in practice, the Jacobian matrices do not need to be computed explicitly, as only their product with a gradient vector is required. Finally, by averaging gradients over training samples, the full gradient of the loss function can be obtained.


# Convolutional Neural Networks for Isolated Character Recognition

## The Problem with old system (Fully Connected)
To understand the important with CNN, we first have to understand the flaw with the old method, as we already know the tradisional model, for extract the relevant information and throw away irrelevant variabilities from input (raw image) there will be a "hand-designed feature extractor" (containing edge detection, counting stroke (what is this? oh, is just counting the stroke, like how 1 have 1 stroke, 4 have 4's and so on), projection profiles (getting the black pixel per-raw, per-column), zoning (for? oh is an old method to gain the information of the location by split image into a smaller square, let say 3x3 and we count how many black pixel in every square), and skeletonization (for? again "oh", its just making every character only have 1 width of pixel, an effort for standardization)), and this "feature extractor" not get optimisation by end-to-end, and often task-spesific, and the output will a feature vector $[0.12, 0.87, 0.01, ...]$. Then there will be a trainable classifier that will categorizes the feature vector from the output into a classess (the guess label). the trainable classifier they use is fully-connected multi-layer networks (or we known it as Fully Connected (FC) or Multi Layer Perceptron (MLP)). but there was another way/idea LeCun said, it was to rely on the feature extractor itself as much as possible (the feature extractor getting train too), so then the network could be give with raw input image (or not really raw, because we normalized the size). but again, there are some obstacle with this idea.

First problem which will affect the hardware site. image are large (in most cases), let say a hundred variables (pixels). and we use fully connected layer, the first layer with 100 hidden units (the neuron, the one with function $y=f(wx+b)$), because of every hidden units have their own weight, the first layer would already contain atleast 10,000 of weights. this will result in increasing the capacity of the system. beside that, the main problem with old-style networks is their lack of invariance to translations or distortions, so the image has to get some preprocessing, and not like a modern style. as example handwriting usually got size-normalized at the word level, so if we have "I eat an apple" and every word got cut and we resize so it fit inside the input field, so the problem occur when (and of course) the word have a different size, the short one will be big and the long one would be very small, this problem and a fact that every human often have a different writing style. But of course it still can works but the problem is it will inneficient, because of every pixel connects with every neuron but every neuron has a different weight, when given a data that always show the number in the middle, it makes only a certain neuron to have the idea about the how line look alike, and if in test we move (translations) the number a little bit off the right, now every neuron that never see the pixel to appear has no idea, so that we have to "teach" every neuron using a very large number of training to cover the possible variations. This results in massive redundancy, where thousands of different neurons end up learning the exact same pattern just to cover different locations.

The second problem is LeCun said, the old systems ignored topology, basically saying because the fact that every neuron connect with every pixel, if we in case have an image of just a number of "7", and this matrix flatten into a vector, and we messing with the order (like say pixel in index 1 got switch with pixel in index 250) the output will be the same. why is this a problem? because it means the old system treats pixels as independent variables, ignoring the fact that a pixel's meaning depends entirely on its neighbors (local correlation), but of course not. logically by considering the position's of every pixel we can get more information, for a line can be represented as a black pixel and the neighbour have a lighter black and when reach white pixel it can be say the edge of the line. 

## So what is CNN?
so here comes the solution, Convolutional Networks. Instead of a neuron staring at the whole image and getting overwhelmed, we restrict its Receptive Field. We force the model to look at Local Features first. In modern terms, we use Kernels (those small matrices that stride across the image). This way, the network learns "concepts" (like edges or corners) that work anywhere in the image, making the system much more efficient and brain-like. Because the kernel slides everywhere, it doesnt matter if the "7" is in the corner or the middle—the same kernel will catch it

CNN is a network that having some layer like this 

### The Principle of Local Receptive Fields
Convolutional Network basically combining "3 architectural ideas" for dealing with invariance which was a big challenge for older systems. 

First of all, "local receptive fields" this time we called it as "Convolutional Layer". so why do we have to look all part of the image. now we use a small filter (kernel). which this part will only look a small amount of the image at one time, and it will stride along the image from row to row, column to column till the last pixel of the image. 

Second of all a "Shared Weights" this part i want to make a point, that old system with the new one (CNN) have a different meaning of "Weights", when the old one as we know every neuron of it have a different weight (because the fact that it receive 1 pixel for 1 neuron). but CNN use the filter/kernel in form of matrix (ussually 3x3) as it Weight. basically this 3x3 kernel, or 5x5 kernel it is the weight right now. and because of the same kernel used in every part of the image. it what meant by "Shared Weight".

after convolution layer, we will have as many feature maps as we want depends how many weight we use (okay hold on, what is feature map? feature map is a matrix made from dot product between a certain part of the image and the kernel. so remember $y=f(Wx + b)$? the $Wx$ it doing the dot product, and so the $Wx$ part will result of many of scalar, where the result will have a position just like when they get in, resulting it in a matrix and each of the values inside the matrix will get add by the bias. this whole process will result in $y$ and this what we called a "feature map"

### Activation Functions

and btw for the $f$ part, it is the "Activation Functions, and back then leCun use the word 'squashing functions' and using Sigmoid or Tanh, which is smart, because of the purpose of this function is to "squash" the unlimited possibility of result (from $\infty$ until $-\infty$) to a range of finite number, why? lets say we dont use this squash function, later after many layer of multiplication, the number in our network would explode and become huge, not just it inneficient, but the computer itself couldnt handle the math anymore. More Importantly, this adds non-linearity to the networks, without it (if we only use $y=wx + b$) our AI just be a simple model that only understand straight line. by giving the $f$, we make sure it can understand complex, curvy patterns

as an example, the sigmoid funcction ($\sigma(x) = \frac{1}{1 + e^{-x}}$) just making sure the positive large number are more to 1, and large negative number are more to 0). 

<p align="center">
  <img src="Asset/sigmoidFunction.jpeg" alt="an image visualize the sigmoid activation function" width="400">
</p>

⚠️ i personally dont want to dive deep into the topic of activation, the point is you got the idea how does it works. but for this particular function, theres a well-known problem called "Vanishing Gradient". what is that? so as we can see from the graph above,as the value approaches zero, whether it positive or negative, the line becomes steeper but the more extremely large the value, the flatter the line becomes. And remember after the AI learn the loss, it will do the backpropagation, and use gradien to change the value for weight and bias (tell me if there are other parameter that getting updated ya). the correction does by doing matrix multiplication in every layers

### Spatial Subsampling (Pooling)

now the feature map then will goes into a new layer (pooling layer). Similar to the convolutional layer, pooling uses a filter (often 2x2) that strides across the map. The goal of this layer is Spatial Subsampling meaning it will reducing the dimensions of the data while keeping the most important information, ideally the stride we use for pooling is based on the size of the filter, because it have to keep the most important information. so we want to make every part of the feature maps are "recorded" (so we dont want to make the stride > the filter size) and we dont want to have many of redundant information (so we dont want to make the stride = 1).

In a common approach like Max Pooling, the filter looks at a specific area and only "spits out" the highest value from that section. As the filter strides across the entire feature map, it produces a new, condensed matrix. This makes the representation smaller and more manageable, ensuring the network focuses on the most prominent features while becoming less sensitive to the exact position of those features.

### Output Layer/Fully Connected (dense layer)
after we receive the feature maps that have been pooled, we now have to flatten the feature maps, dont think it will have complex math, it literally just change form, from matrix to vector, so if we have 2 feature maps with size of 7x7 now we have a vector with size of $1 \times 98$, and each scalar will connect to some of neuron, let say we want to predict 3 class, so each scalar inside the vector will connect to each neuron so the size will be $98 \times 3$ and after we multiply we will have a vector with a size of $1 \times 3$ this vector is our raw score (logits). logits is just a raw score result from linear transformation before we put it in a probility function (like softmax) so it normal if we got high score because it havent be "normalise" yet

$$Logits = (x \cdot W) + b$$

say the scores result we got for each class are [2.5, -1.2, 0.3], and after we put it in softmax function, for a vector $z = [z_1, z_2, ..., z_k]$ and every 

$$softmax(z_i​) = \frac{e^{z_i}}{\sum_{j=1}^{k} e^{z_j}}$$
so:

first we have to compute the exponential of each element

$$
\begin{aligned}
e^{2.5} &\approx 12.182 \\
e^{-1.2} &\approx 0.301 \\
e^{0.3} &\approx 1.350
\end{aligned}
$$

then we Compute the sum of exponentials

$$
\sum_{j=1}^{3} e^{z_j}
= 12.182 + 0.301 + 1.350
= 13.833
$$

we normalize each value by the total

$$
\begin{aligned}
\text{softmax}(2.5) &= \frac{12.182}{13.833} \approx 0.881 \\
\text{softmax}(-1.2) &= \frac{0.301}{13.833} \approx 0.022 \\
\text{softmax}(0.3) &= \frac{1.350}{13.833} \approx 0.098
\end{aligned}
$$

then yeay we turned the logits value into a probability

$$
\text{softmax}(\mathbf{z})
=
[0.881,\;0.022,\;0.098]
$$

this mean, class[0] are the highest prediction given by our CNN




