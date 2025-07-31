---
layout: post
title: ArcFace Margin Loss
math: true
---

Introduction

We will start by looking at how we might approach this problem using standard softmax and where that breaks down, and then look at how ArcFace solves the problem. For demonstrative purposes, we will use embeddings of only 2 dimensions (as this makes drawing much easier), and assume a training data set of 5 identities: Tom Hanks, Cate Blanchett, Morgan Freeman, Meryl Streep, and Harrison Ford.


## Explain embeddings and dot product

When building a face recognition model using SoftMax, we can break down the network into three components:

1. **Training Data**: This is a collection of face images along with their respective identities as one-hot vectors.
2. **Embedding Network**: This network takes an image of a face and maps it to an embedding, a real valued vector representing the face's identity. This embedding should have a meaningful spatial relationship with other embeddings of images of faces with the same identity. This network might leverage an existing off-the-shelf model architecture such as ResNet, ViT, etc.
3. **Classifier Network**: The classifier network maps embeddings to probability distributions over the available classes. It effectively tells us the likely class of a given embedding, and consequently the face image from which the embedding was derived.

For demonstrative purposes, lets assert the following:
- Our training data consists of face images for 5 identities: Tom Hanks, Cate Blanchett, Morgan Freeman, Meryl Streep, and Harrison Ford
- Our embedding network outputs 2D embeddings (for the sake of making plots easier)

The classifier network will be our main focus. 



Using an off-the-shelf computer vision model as the base, we can build a network that produces embeddings representing face identities as the output. So it takes an image of a face as an input and outputs an embedding representing the face's identity as an output. Lets call this network the embedding network. The embeddings will be vectors of 2 dimensions (to make visualization easier to draw and understand). I'll denote these output embeddings using $x$.

Unfortunately, we don't have embeddings in our training data, just pictures of people for each class (the classes in this case being the actors mentioned above). To bridge this gap, we'll need a classifier network. The classifier network takes embeddings as input and outputs a probability distribution over the classes in the training data, indicating how likely a given embedding is to belong to each class. So so far we have this:

image --> embedding-network --> embedding --> classifier --> probability distribution

The classifier contains two pieces, the classifier weights $W$, and the SoftMax function. $W$ is a matrix that really just a collection of vectors representing each class. Each of these vectors will be represented with $w$ and a subscript indicating the class. In this case, we'll have $w_tom$, $w_cate$, $w_morgan$, $w_meryl$, and $w_harrison$. The matrix might look something like this:

TODO: a Matrix W with some weights and an indicator as to the class vectors.

Our embedding $x$ is multiplied by the matrix $W$ to produce logits, $z$. To explain what the logits are, lets dig deeper into what happens when we multiply $x$ and $W$:

$$
z = x \dot W
$$

When we take the dot product between two matrices (or a row vector and a matrix in this case), what we are really doing is taking the dot product between each row vector on the left with each column vector on the right. This yields a new matrix, or in this case row vector since we only have one embedding here, like this:

$$
z = x \dot W = [ x \dot w_tom, x \dot w_cate, x \dot w_morgan, x \dot w_meryl, x \dot w_harrison]
$$

The dot product of two vectors is the product of the magnitudes of each vector scaled by the cosine of the angle in between them:

$$
x \dot w = |x||v| cos \theta
$$

So if two vectors point in the same direction (have $\theta close to 0), and each have a large magnitude, then their dot product will also be large. If two vectors have a $\theta$ of 90 degrees, then the dot product will be 0 irrespective of the magnitudes of the vectors (since the cosine of 90 degrees is 0). Ideally, we want the dot product for x to be largest with the vector $w$ representing the correct class and small for all other classes. We will also use a set of biases, $b$, as is common with classifiers, for each class which is simply added on to our dot product. After training, these biases will often reflect the frequency of each class in the training data. So if the training data contains more pictures of Harrison Ford than anyone else, the $b_harrison$ will be the largest bias. This gives us an equation for the logits as follows:

$$
z = x \dot W + b = [ x \dot w_tom + b_tom, x \dot w_cate + b_cate, x \dot w_morgan + b_morgan, x \dot w_meryl + b_meryl, x \dot w_harrison + b_harrison]
$$

With the logits alone, we can rank the categories for an embedding $x$ from most likely to least likely, but we still need to cajole these logits into a probability distribution. Lets consider the following case:

TODO: example where some of the logits end up being negative

One thing you might think to try to turn the above logits into a probability distribtuion is to simply add all the logits together and then divide each logit by the sum. Unfortunately, some of the numbers are negative, so this won't work. One neat transformation we could do is 10 to the power of each logit, like so:

$$
z = [1, -1, 0.5, -0.25, 3]

10^z = [10^1, 10^-1, 10^0.5, 10^-0.25, 10^3] = ...
$$

Notice how this operation does change the relative ordering of the logits, however it does make the logits all positive numbers. Now we can sum them together and divide each logit by the sum to get our probability distribution. This is exactly what the softmax function does:

$$
softmax(v) = u where u_i = \frac{e^{v_i}}{sum e^{v_j}}
$$

This does effectively the same thing, except using euler's number, instead of 10:

$$
z = [1, -1, 0.5, -0.25, 3]

e^z = [e^1, e^-1, e^0.5, e^-0.25, e^3] = ...

sum e^z = ....

softmax(z) = e^z / (sum e^z) = [....]
$$

We now have our probability distribution. This gets us from an image of a face, to an embedding the encodes the face's identity, to a probability distribution among our 5 possible classes. Let's run through of how this all works using a picture of Harrison Ford:

harrison ford --> embedding network --> embeddings --> classifier weights --> logits --> probability distribution





## Explain how SoftMax scales logits into a probability distribution

## Weight vectors establish class boundaries

## Inter-class and Intra-class distances with normal SoftMax

## Remove bias when calculating logits

## Normalize the embeddings and weight matrix

## More ways to think about dot products (product of magnitudes scaled by cosine of the angle)

## Adding a margin to the correct category for each sample