---
layout: post
title: Weight Decay is Not L₂ Regularization
math: true
author: john
---
When it comes to training neural networks, understanding the nuances of optimization techniques is crucial for achieving the best performance. One common area of confusion is the difference between **weight decay** and **L₂ regularization**. While they may seem similar, they serve different purposes and behave differently, especially when using adaptive optimizers like Adam or RMSprop.

We once treated **weight decay** and **L₂ regularization** as interchangeable terms in machine learning. In fact, this is exactly what I was taught over 10 years ago. While it made sense at the time, this conflation caused numerous issues, especially with the rise of adaptive optimizers like Adam. Unfortunately, this misunderstanding still persists today with Google's AI search results saying "Weight decay, also known as L₂ regularization...", and blog posts using the wrong version of PyTorch's Adam optimizer. Here I'll explain why weight decay and L₂ regularization are not the same, especially when using adaptive optimizers like RMSprop or Adam, and how to properly use weight decay in PyTorch.

## Neural Network Optimization Basics

Before we get into weight decay and L₂ regularization, let's briefly review how neural networks are trained and how weights get updated.

At its core, training a neural network is all about adjusting the model's weights (or parameters, which I'll use interchangeably) to minimize a loss function. The loss function tells us how badly our model is performing on the training data. We use an optimizer, like stochastic gradient descent (SGD), to find the weights that result in the lowest loss. In its simplest form, the weight update rule for SGD at each time step `t` looks like this:

$$
\Theta_{t+1} = \Theta_t - \alpha \cdot \nabla L(\Theta_t)
$$

Here, $\Theta$ is a vector representing the weights of our model. At each step, we calculate the gradient of the loss function with respect to the weights, $\nabla L(\Theta_t)$, and take a small step in the opposite direction. The scalar $\alpha$ (learning rate) controls how big of a step we take. This process is repeated thousands or millions of times until the model converges.

## Understanding L₂ Regularization

With that context, let's talk about L₂ regularization. L₂ regularization is a technique to prevent overfitting, which is when a model learns the training data too well and fails to generalize to new, unseen data. It works by adding a penalty term to the loss function. This new term is the sum of the squares of all the weights, multiplied by a small scalar constant $\lambda$:

$$
L_{total}(\Theta) = L_{data}(\Theta) + \frac{\lambda}{2} \|\Theta\|_2^2
$$

The $\lVert\Theta\rVert_2^2$ term is the squared L₂ norm of the weight vector, $\Theta$, which is simply the sum of the squares of all its elements. The $\lambda$ parameter controls the strength of the regularization. A larger $\lambda$ means a stronger penalty for large weights, which encourages the model to keep its weights small.
The intuition behind L₂ regularization is that by keeping model weights small, no individual weight can exert disproportionate influence on the output, thereby reducing the model’s tendency to overfit.

## What is Weight Decay?

Weight decay, on the other hand, seeks to reduce the risk of overfitting by directly nudging weights toward zero during the update step. It does this by applying a small multiplicative shrinkage factor to the weights at each update:

$$
\Theta_{t+1} = (1 - \alpha \lambda)\Theta_t - \alpha \cdot \nabla L_{data}(\Theta_t)
$$

Again we have a $\lambda$ term, in this case to control the strength of the weight decay. Larger values of $\lambda$ cause the weights to shrink more aggressively. The $( 1 - \alpha \lambda)$ term is generally a value a little less than 1 which causes the weights to "decay" exponentially over time, hence the name.

## Why Were They Considered the Same?

So, why did we use L₂ regularization and weight decay interchangeably in the past? Let's see how L₂ regularization impacts the weight update step when using SGD. First, we need to find the gradient of our new total loss with respect to the weights:

$$
\begin{align*}
L_{total}(\Theta) &= L_{data}(\Theta) + \frac{\lambda}{2} \|\Theta\|_2^2 \\
\nabla L_{total}(\Theta) &= \nabla L_{data}(\Theta) + \lambda \Theta
\end{align*}
$$

Now, let's plug this into our SGD update rule:

$$
\begin{align*}
\Theta_{t+1} & = \Theta_t - \alpha \cdot (\nabla L_{data}(\Theta_t) + \lambda \Theta_t) \\
 & = \Theta_t - \alpha \cdot \nabla L_{data}(\Theta_t) - \alpha \lambda \Theta_t \\
& = (1 - \alpha \lambda) \Theta_t - \alpha \cdot \nabla L_{data}(\Theta_t)
\end{align*}
$$

Look at that, the update steps are identical! This is why L₂ regularization and weight decay were historically treated as the same thing, especially in the context of plain stochastic gradient descent (SGD). In fact, many deep learning libraries implemented weight decay as L₂ regularization in their optimizers, which further reinforced this misconception.

If the math works out the same, then what's the problem? Issues arise when we introduce adaptive optimizers like RMSprop and Adam. These optimizers adjust the learning rate for each parameter based on its historical gradients. This is where the equivalence breaks down. The paper [Decoupled Weight Decay Regularization (Loshchilov & Hutter, 2017)](https://arxiv.org/abs/1711.05101) explains this in detail. They focus on Adam, but I'll use RMSprop as an example since it's a bit simpler to grasp.

In SGD, weights are updated by subtracting the gradient of the loss function scaled by a fixed learning rate. However, gradients can be noisy, and using the same learning rate for all parameters may lead to inefficient or unstable updates. RMSprop improves upon this by maintaining an exponentially decaying average of the squared gradients for each parameter. This moving average is used to scale the gradient, effectively adapting the learning rate per parameter. As a result, RMSprop helps stabilize training and improve convergence, especially in settings with noisy or sparse gradients. The update step looks something like this:

$$
\begin{align*}
V_t &= \beta V_{t-1} + (1 - \beta) (\nabla L(\Theta_t))^2 \\
\Theta_{t+1} &= \Theta_t - \frac{\alpha}{\sqrt{V_t} + \epsilon} \nabla L(\Theta_t)
\end{align*}
$$

The vector $V_t$ keeps a moving average of the squared gradients, where $\nabla L(\Theta_t)$ is the gradient at the current step `t`. The scalars $\beta$ and $\epsilon$ are hyperparameters, with $\epsilon$ being a small number to prevent division by zero. The key part is that the learning rate $\alpha$ is divided by the square root of $V_t$, effectively giving each parameter its own learning rate.

Now let's look at what happens if we add L₂ regularization to our cost function and use RMSprop:

$$
\begin{align*}
L_{total}(\Theta_t) & = L_{data}(\Theta_t) + \frac{\lambda}{2} \|\Theta_t\|_2^2 \\
\nabla L_{total}(\Theta_t) & = \nabla L_{data}(\Theta_t) + \lambda \Theta_t
\end{align*}
$$

When we plug this into the RMSprop update rule, we get:

$$
\begin{align*}
V_t &= \beta V_{t-1} + (1 - \beta) (\nabla L_{total}(\Theta_t))^2 \\
\Theta_{t+1} &= \Theta_t - \frac{\alpha}{\sqrt{V_t} + \epsilon} (\nabla L_{data}(\Theta_t) + \lambda \Theta_t) \\
& = \Theta_t - \frac{\alpha}{\sqrt{V_t} + \epsilon} \nabla L_{data}(\Theta_t) - \frac{\alpha }{\sqrt{V_t} + \epsilon}\lambda \Theta_t \\
& = (1 - \frac{\alpha \lambda}{\sqrt{V_t} + \epsilon}) \Theta_t - \frac{\alpha}{\sqrt{V_t} + \epsilon} \nabla L_{data}(\Theta_t)
\end{align*}
$$

Notice the problem? What would be our weight decay term $(1 - \alpha \lambda) \Theta_t$, now divides $\alpha \lambda$ by $\sqrt{V_t}$. This means the effective weight decay is different for each parameter and changes over time, which is not what we want from true weight decay. High-variance parameters (with large historical gradients) will get less decay, and low-variance parameters will get more. This is not the uniform shrinkage we expect from weight decay, and it turns out to not be very helpful either.

## The Solution: Decoupled Weight Decay

The solution proposed by Loshchilov and Hutter is pretty straightforward. Instead of using L₂ regularization in the loss function to get weight decay, just apply the weight decay directly to the weights in the update step. This way, we ensure that the weight decay is applied uniformly across all parameters, regardless of the optimizer used. The update step for RMSprop with decoupled weight decay looks like this:

$$
\begin{align*}
V_t &= \beta V_{t-1} + (1 - \beta) (\nabla L_{data}(\Theta_t))^2 \\
\Theta_{t+1} &= (1 - \alpha \lambda) \Theta_t - \frac{\alpha}{\sqrt{V_t} + \epsilon} \nabla L_{data}(\Theta_t)
\end{align*}
$$

## Proper Weight Decay in PyTorch

Great! Now that we know weight decay and L₂ regularization are not the same thing, and weight decay is probably the thing we actually want, how do we use it in PyTorch? You might think that using the `weight_decay` parameter in PyTorch's optimizers like Adam or RMSprop would do the trick, but that's not always the case. Even though this issue was thoroughly worked out back in 2017, PyTorch still preserves this bug in many instances for posterity, so that future generations may know our suffering. Luckily, the docs for each optimizer will tell you how the `weight_decay` parameter is implemented. For example, in the [Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html) optimizer, `weight_decay` is implemented as L₂ regularization (probably not what you want). If you want to use decoupled weight decay, you need to use the [AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html) optimizer or pass the `decoupled_weight_decay` argument to the `Adam` optimizer:

```python
import torch
import torch.optim as optim

# Using AdamW with decoupled weight decay
model = ...  # Your model here
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01
)

# Alternatively, using Adam with decoupled weight decay
model = ...  # Your model here
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01,
    decoupled_weight_decay=True
)
```

As far as I'm aware, PyTorch does not come with a built-in RMSprop optimizer that supports decoupled weight decay, but if you have your heart set on using RMSprop, you could just use `AdamW` and set the momentum coefficient, the first element of the `betas` parameter tuple, to 0, which will essentially give you RMSprop with decoupled weight decay:

```python
import torch
import torch.optim as optim

# Using AdamW with momentum set to 0 for RMSprop-like behavior
model = ...  # Your model here
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01,
    betas=(0, 0.99)
)
```

## Conclusion: Getting Weight Decay Right Matters

In the world of deep learning, subtle implementation details can make a significant difference in model performance. The difference between weight decay and L₂ regularization is one such detail that's often overlooked but can be crucial when using adaptive optimizers.

To summarize the key points:

1. With standard SGD, weight decay and L₂ regularization are mathematically equivalent.
2. With adaptive optimizers like Adam and RMSprop, they diverge significantly.
3. True weight decay applies a uniform shrinkage to all parameters, while L₂ regularization with adaptive optimizers leads to parameter-specific decay that varies with gradient history (see [Decoupled Weight Decay Regularization (Loshchilov & Hutter, 2017)](https://arxiv.org/abs/1711.05101)).
4. In PyTorch, use AdamW or set `decoupled_weight_decay=True` with Adam to implement proper weight decay.

The next time you're training a model with an adaptive optimizer like Adam, remember to use proper decoupled weight decay instead of L₂ regularization. Your model's performance might thank you for it!

Have you experienced differences in model performance when switching from L₂ regularization to proper weight decay? I'd love to hear about your experiences in the comments.
