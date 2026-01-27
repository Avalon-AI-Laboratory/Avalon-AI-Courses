# This part we are gonna talk about Gradient Descent and Backpropagation
and for this particular topics, iam gonna use **Grant Anderson (A.K.A 3Blue1Brown)**'s video as a reference so shoutout to the videos <a href="https://www.youtube.com/watch?v=IHZwWFHWa-w">Gradient Descent</a> and <a href="https://www.youtube.com/watch?v=tIeHLnjs5U8"> BackPropagation</a>

Spoiler alert the neural network used in the videos, are fully connected layer. and for backpropagation process (because gradient used in backprop). it actually does the same thing to CNN.

## What is Gradient Descent?
So to the machine we train to learning ("this machine is learning"), we cant just throw a train set of data to the networks, and rely all the accuracy to Weight and Bias. thats why we added a backpropagation process, where the network will updated the parameter by respect to the loss, using Gradien Descent. for the purpose of finding the global minima (a place where the parameters resulting in a very small loss) from the function. 

**👾 You guys can skip this part if you want, just me trying to make something clear**

as i already state before, 3Blue1Brown used the fully connected layer as the network, and i just want to make sure to all of us, this will not affect the topics because the difference between old style (used by the 3Blue1Brown) and CNN (used by us) its just because of every pixel connect to all of the neuron, the input first have to be flatten into a 1x1 vector (so if the image size 28x28 pixel, after flatten it would be a one line with 784 number in it (a.k.a vector), and so because every pixel have a connection to the weight, weight will be a matrix a size of 100x784. where $100$ represented a 100 neuron we have in layer 1, and $784$ because of every neuron have a connection with every pixel. But in CNN we use layer convolution as the weight so it is a matrix and the $x$ is also a matrix of part in the image that overlap with the convlayer and thats it)

In the old style, the calculation is $f(w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b)$ where $n$ is the entire image. In CNN, the formula is actually the same, BUT only for a small window at a time. Instead of $n$ being the whole image, $n$ is just the size of our kernel (e.g., $3 \times 3 = 9$ pixels). So, CNN is basically doing the same 'dot product' math, but locally and repeatedly, which makes it much more efficient at catching patterns. Ok head back to the main topics
 
**okay back to the main topic 🤗**

### Cost Function
So after we rely on the weights and biases, and normally the output layer will give a very wrong prediction, because the system is not learning enough and it just a bunch of scrap inside the output layer. and to fix this why we use "Cost Function", what is that? Cost function is used for us to tell the machine that the prediction they give is a wrong one (wrong class/label), the mathematical way of doing this is just we add up the squares of the differences between probability given by the output layer and the actual label. 

Because output layer give a probability in every class, and the highest will be picked up as the prediction, we take all of the probablity in every class, let say we have 3 class (odd, even, zero). and the probability is (0.27, 0.99, 0.01) but the real class is "odd" so it should be (1, 0, 0), all left to do to get the cost function is just $(0.27 - 1)^2 + (0.99 - 0)^2 + (0.01 - 1)^2$ and the result is: "$hmm...$". and by this we realise that the cost would be small if the prediction is more accurate (the real class got big probabilities) and would big if the prediction is miss. and so we doing this operation by all of the training data and at the end all of the cost function will be average, the average will show us how bad our machine is. 


### Gradient Descent

But having the value of the cost doesnot solve the problem. By telling the machine "you have a problem" doesn't explain how to fix it. We need to change the weights and biases. To understand this, let’s simplify. Instead of imagining a cost function with 172,800 weights ($C(w_1, w_2, \dots, w_n)$), imagine a simple function with just one input and one output ($C(w)$ (i guess 3Blue1Brown just use $w$ because he said "1 input and 1 output")). The Goal is to find the input $w$ that minimizes the cost. Since we don't know where the "bottom" is, we start with a random $w$. We then use the First-Order Taylor Expansion (a way to estimate a value from a complited function by using their derivative) to approximate the function:

$$f(x) \approx f(a) + f'(a)(x-a)$$


🦦 why we cut taylor's equation right after $f(x) \approx f(a) + f'(a)(x-a)$, because compute the second derivative $f''$ are just expensive for computation 

and we have to make a decision which "step" the machine have to take to lower the output. so if we can figure out the slope of the function are "standing" now. the machine can move to the left (lower the value in weight) if the slope is positive, and move to the right (add more value in weight) if the slope is negative. by doing this repeatedly at each point we check the new slope and make sure the step not too short or too far, we are going to approach the local minima. but of course there are many local minima (or if you can, the "global minima") that might you have ended up, because it depends on what input you start at. and because of the slope, the closer we are to the minimum which mean the area will be more plane then the slope will reach closer to 0, so the step also get smaller and smaller

now lets add up 1 more input, now, we have two inputs ($C(x,y)$), the cost function is graphed as a surface above an xy-plane. We dont just ask for a single slope anymore, we ask for the gradient ($\nabla{C(x, y)}$). The gradient points in the direction of the steepest ascent (the direction that increases the cost most quickly). By taking the negative of that gradient, it shows the machine the direction to step that decreases the cost most quickly. furthermore, the magnitude (the length) of this gradient vector indicates the steepness of the slope. If the magnitude is large, the slope is very steep (meaning we are likely far from the goal), so the machine takes a bigger step. If the magnitude is small, the slope is almost flat (meaning we are near the bottom), so the machine takes a smaller, more precise step.

### BackPropagation
Now, how does that "step" actually reach the weights? We use the Chain Rule.

1. The Error Signal: It starts at the Loss ($E$). We calculate the derivative of the loss with respect to the output

2. The Activation Gate: The signal moves backward through the activation function (like Sigmoid or Tanh). We multiply the error by the derivative of the activation function. This "filters" the signal; if the activation was in a very flat region, the gradient becomes very small (this is known as the Vanishing Gradient problem (we already talk this topic at the main topic))

3. The Update: Once the signal passes the activation, it finally reaches the weights and biases. For Biases, The bias update is the sum of the gradient signals across the entire feature map. Since one bias is shared, it collects all the "complaints" from that layer. And for Weights (Kernels): The gradient for each weight is the error signal multiplied by the Input that was used during the forward pass. Finally, we apply the update using the Learning Rate ($\epsilon$):

$$W_{new} = W_{old} - \epsilon \cdot \text{Gradient}$$

## A quick summary, without any out of topic talk
### The Chain Rule: Connecting the Dots
We have the Cost Function, and we have the Gradient. But how does a mistake at the output layer tell a specific weight in the first layer to change? We use the Chain Rule.Think of it as a chain of command. If the final product is bad (High Cost), we look at the last person in line (Output Layer), then the person before them (Pooling), then the one before that (Activation), and finally the person who made the first move (Kernel Weights). Mathematically, to find how much the Cost ($C$) changes when we change a Weight ($w$), we multiply the sensitivities along the path

$$\frac{\partial C}{\partial w} = \frac{\partial C}{\partial \text{out}} \times \frac{\partial \text{out}}{\partial \text{net}} \times \frac{\partial \text{net}}{\partial w}$$

### Backpropagation through Pooling (Subsampling)
This is the part where CNNs differ from the 3Blue1Brown model. Since Pooling doesnt have weights, it doesn't "learn." However, it acts as a router for the gradient:

- Max Pooling: During the forward pass, we only kept the maximum value. During backpropagation, the gradient only flows back to that specific position where the maximum value came from. The other pixels get a gradient of 0 because they didn't contribute to the output.

- Average Pooling: The gradient is distributed equally among all pixels in the pooling window.

### Backpropagation through Activation
When the error signal travels back, it will eventually hits the Activation Function gate. As we discussed, the values in the feature map were changed by functions like Sigmoid or Tanh during the forward pass. When going backward, we must multiply our error signal by the derivative of the activation function.

- If the derivative is large, the signal passes through strongly.

- If the derivative is near zero (which happens if the input was very high or very low in Sigmoid), the signal "dies out." This is why choosing the right activation function is crucial for the gradient to actually reach your weights.

### Updating the Bias 
we might wonder about the Bias. In CNNs, we use Shared Bias. This means one single number (bias) is added to every pixel in a Feature Map.Because it is shared, the update is simple: we sum up all the gradient signals across that entire Feature Map to calculate the "Total Bias Gradient." We then update it just like the weights

$$b_{new} = b_{old} - \epsilon \cdot \sum (\text{Gradients in Feature Map})$$


## Now we dive into example 

### A Concrete Example: From Error to Update

To make sure we really get the idea about whats really going on here, let’s trace a single "signal" from the moment it fails at the output until it fixes a weight in a kernel. In this section, we will trace how a mistake made at the output layer is translated, purely through derivatives, into a small change in a convolution kernel weight. For clarity, we will focus on a single weight, even though the same process happens simultaneously for all weights in the network.



