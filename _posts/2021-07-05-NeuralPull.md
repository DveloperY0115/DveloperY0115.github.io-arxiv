---
layout: post
title: "Summary of 'Neural-Pull: Learning Signed Distance Functions from Point Clouds by Learning to Pull Space onto Surfaces'"
# use_math: true
background: '/assets/post-images/NeuralPull/NeuralPull_fig1.png'
---

# Motivation

- Reconstructing continuous surfaces from 3D point clouds is a fundamental operation in 3D geometry processing.
- Recent approach using neural network to learn SDF gives promising results → Adopt this idea for 3D point cloud to surface reconstruction.

# Key Contributions

- Neural-Pull, a simple, new approach for learning SDFs directly from raw 3D point clouds ***without*** ground truth signed distance values.
- A network that pulls query 3D point to its closest point on the surface using the predicted SDF values and the gradient at the query point. → Done by network end-to-end
- Effectively learn SDFs by updating the predicted signed distance values and the gradient simultaneously in order to pull surrounding 3D space onto the surface.
- Significant accuracy improvement in surface reconstruction and single image reconstruction.

---

# Method

## Problem Statement

A neural network for learning SDFs that represent 3D shapes.

An SDF $\boldsymbol{\mathcal{f}}$ predicts a signed distance value $s \in \mathbb{R}$ for a query 3D location $\textbf{q} = [x, y, z]$.

Optionally, one can provide an additional condition $\mathcal{c}$ as input, such that $f(\textbf{c}, \textbf{q}) = s$. Unlike previous approahes *(Park et al., 2019; Michalkiewicz et al., 2019)* which provided known SDF values during training, **this work aims to learn SDF $f$ in a 3D space directly from 3D point cloud** $\textbf{P} = \\{ \textbf{p}_j, j \in [1, J]\\}$.

## Overview

<img class="img-fluid" src="/assets/post-images/NeuralPull/NeuralPull_fig1.png">
<span class="caption text-muted">Figure 1. <b>Demonstration of pulling surrounding 2D space onto a surface</b>.</span>

*Neural-Pull* is a neural network that learns how to pull a 3D space onto the surface represented by the point cloud $\textbf{P}$, and eventually learns to represent a SDF $f$. 

It uses the given point cloud $\textbf{P}$ and the gradient within the network itself to represent 3D shapes. It tries to learn to pull a query location $q_i$ randomly sampled around the surface to its nearest neighbor $\textbf{t}_i$ on the surface, where the query locations form a set $\textbf{Q} = \\{ \textbf{q}_i, i \in [1, I] \\} $.

The pulling operation pulls the query location $q_i$ with a stride of signed distance $s_i$, along or against (since the query point can either be inside or outside of the surface) the direction of the gradient $\textbf{g}_i$ at $\textbf{q}_i$, obtained within the network.

## Pulling Query Points

A 3D query location $\textbf{q}_i$ is pulled to its nearest neighbor $\textbf{t}_i$ on the surface using the predicted signed distance $s_i$ and the gradient $\textbf{g}_i$ at $\textbf{q}_i$ within the network. The gradient $\textbf{g}_i$ is a vector whose components are the partial derivatives of $f$ at $\textbf{q}_i$, such that,

$$\textbf{g}_i = 
\begin{bmatrix} 
\frac{\partial f (\textbf{c}, \textbf{q}_i)}{\partial x} \\
\frac{\partial f (\textbf{c}, \textbf{q}_i)}{\partial y} \\
\frac{\partial f (\textbf{c}, \textbf{q}_i)}{\partial z}
\end{bmatrix}$$

where $\textbf{q}_i = [x, y, z]$. Note that it's often denoted as $\nabla f (\textbf{c}, \textbf{q}_i)$, where $\textbf{c}$ is a condition. Mathematically, this is the direction along which the signed distance change is the largest in 3D space. Thus, it's natural to choose the gradient as the direction for pulling the nearby query point onto the closest point on the surface. Note that the graident $\nabla f$ also serves as a normal vector at a given point due to the characteristic of SDF.

That being said, we can use such property to define 'pulling' operation, which pulls a query point $\textbf{q}_i$ onto a point $\textbf{t}_i$, along or against the direction of gradient $\textbf{g}_i$. And the equation is as follows:

$$ \begin{gather} \textbf{t}_i^{\prime} = \textbf{q}_i - f(\textbf{c}, \textbf{q}_i) \times \frac{\nabla f (\textbf{c}, \textbf{q}_i)}{\left\lVert \nabla f (\textbf{c}, \textbf{q}_i) \right\rVert\_{2}} \end{gather} $$

where $ \textbf{t}\_i^{\prime} $ is the pulled query point $ \textbf{q}_i $ after pulling, $\textbf{c}$ is the condition to represent ground truth point cloud $\textbf{P}$, and $ \nabla f (\textbf{c}, \textbf{q}_i) / \lVert \nabla f (\textbf{c}, \textbf{q}_i) \rVert\_{2} $ is the unit vector representing the directional component of the gradient $\nabla f (\textbf{c}, \textbf{q}_i)$. Since $f$ is a continuously differentiable function, $\nabla f (\textbf{c}, \textbf{q}_i)$ can be easily obtained via back-propagation.

As shown in the figure 1, there are two possible cases when applying this operation on the query point:

1. $ \textbf{q}\_i $ inside of the shape $\textbf{P}$: $s_i < 0$, then the operation will pull $ \textbf{q}\_i $ *along* the direction of gradient such that $ \textbf{t}_i^{\prime} = \textbf{q}_i + \vert f (\textbf{c}, \textbf{q}_i)\vert \times \nabla f (\textbf{c}, \textbf{q}_i) / \lVert \nabla f (\textbf{c}, \textbf{q}_i) \rVert\_{2} $.
2. $ \textbf{q}\_i $ outside of the shape $ \textbf{P} $: $s_i > 0$, then the operation will pull $\textbf{q}\_i$ *against* the direction of gradient such that  $ \textbf{t}\_{i}^{\prime} = \textbf{q}_i - \vert f (\textbf{c}, \textbf{q}_i)\vert \times \nabla f (\textbf{c}, \textbf{q}_i) / \lVert \nabla f (\textbf{c}, \textbf{q}_i) \rVert\_{2} $.

## Query Locations Sampling

The query locations are sampled randomly around each point $ \textbf{p}\_j $ of the ground truth point cloud $\textbf{P}$. Specifically, we construct an isotropic Gaussian distribution $\mathcal{N} (\textbf{p}_{j}, \sigma^{2})$ for each point $\textbf{p}_j \in \textbf{P}$. Then we sample 25 points from the distribution. Here, the variance $\sigma^{2}$ determines how widely the sample points are spreaded according to the mean point $\textbf{p}_j$.

Particularly, this work used an adaptive way to set $\sigma^{2}$ as the square distance between $\textbf{p}_j$ and its 50-th nearest neighbor. → reflects location density around $\textbf{p}_j$

This adaptive sampling improves the learning accuracy, since it's hard to predict both SDF value and the corresponding gradient accurately with a point far from the ground truth (which we need to predict) surface.

## Loss Function

The goal of the network is to train a network so that it can pull a query location $\textbf{q}_i$ to its nearest neighbor $\textbf{t}_i$ on the point cloud $\textbf{P}$. Thus, one of the effective way to guide the network to learn this behavior is to constrain it on the square error (Euclidean distance between the predicted and ground truth),

$$ \begin{gather} d(\\{ \textbf{t}\_{i}^{\prime}\\}, \\{ \textbf{t}\_{i}\\}) = \frac{1}{I} \sum_{i \in [1, I]} \lVert \textbf{t}\_{i}^{\prime} - \textbf{t}\_{i} \rVert_{2}^{2} \end{gather} $$

here, $ \textbf{t}\_{i}^{\prime} $ is the pulled query point obtained by the pulling operation described previously, and $ \textbf{t}\_{i} $ is the nearest neighbor of $ \textbf{t}\_{i}^{\prime} $, which is one of $\textbf{p}\_{j}$s in the set of points in point cloud $\textbf{P}$.

## Convergence to SDF

While the idea is fancy and elegant, one important question still remains unanswered.

**"Why the learned function $f$ can converge to a *signed* distance function representing a shape?"**

Obviously, the equation of pulling operation is also valid in the case of unsigned distance function. However, there's a crucial difference between these two in terms of gradient.

<img class="img-fluid" src="/assets/post-images/NeuralPull/NeuralPull_fig2.png">
<span class="caption text-muted">Figure 2. <b>The illustration of the difference between signed distance field and unsigned distance field</b>.</span>

In the figure 2, one can easily observe that the gradient of signed distance field has constant value as a query point $\textbf{q}$ moves along a particular direction. Meanwhile, for the case of unsigned distance field, there's a discontinuity when the $\textbf{q}$ is actually on the surface. Thus, from this observation, we can derive that a continuous function approximated by MLP can automatically converge to an SDF (not uSDF) using the proposed loss.

**Theorem 1.**
A continuous function $f$ implemented by MLP which is trained to minimize $\ell_2$ loss $ d(\{ \textbf{t}\_{i}^{\prime}\}, \{ \textbf{t}\_{i}\}) = \frac{1}{I} \sum_{i \in [1, I]} \lVert \textbf{t}\_{i}^{\prime} - \textbf{t}\_{i} \rVert_{2}^{2} $, can converge to a signed distance function if the equation $f(\textbf{p} - \textbf{N} \Delta t) = - f (\textbf{p} + \textbf{N} \Delta t)$ is satisfied at any point $\textbf{p}$ on the surface $(f(\textbf{p})=0)$, where $\textbf{N}$ is the normal at $\textbf{p}$, $\lVert \Delta t\rVert < \mu$ and $\mu$ indicates a small number.

**Proof of Theorem 1.**
Since $f$ is a continuous function representing SDF, if $\nabla f(\textbf{p}) \neq \textbf{0}$, the normal at $\textbf{p}$ becomes $\textbf{N} = \nabla f (\textbf{p}) / \lVert \nabla f (\textbf{p}) \rVert_{2}$. Assuming $\Delta \textbf{p} = \textbf{N} \Delta t$ and from the definition of gradient, we have,

$$\lim_{\Delta \textbf{p} \to \textbf{0}} (f (\textbf{p} + \Delta \textbf{p}) - f(\textbf{p})) / \Delta \textbf{p} = \textbf{N} \times \lVert \nabla f (\textbf{p}) \rVert_{2}$$

Then the above can be rewritten by removing $\lim$ and instead introducing infinitesimal $\alpha$,

$$(f (\textbf{p} + \Delta \textbf{p}) - f(\textbf{p})) / \Delta \textbf{p} = \textbf{N} \times \lVert \nabla f (\textbf{p}) \rVert_{2} + \alpha$$

And this can further be modified by multiplying $\Delta \textbf{p} \neq \textbf{0}$ on both sides, obtaining:

$$f(\textbf{p} + \Delta \textbf{p}) - f (\textbf{p}) = (\textbf{N} \times \lVert \nabla f (\textbf{p}) \rVert_{2} + \alpha) \times \Delta \textbf{p}$$

Similarily, we can derive,

$$f(\textbf{p} - \Delta \textbf{p}) - f (\textbf{p}) = -(\textbf{N} \times \lVert \nabla f (\textbf{p}) \rVert_{2} + \alpha) \times \Delta \textbf{p}$$

Since $f(\textbf{p}) = 0$, we finally have:

$$f(\textbf{p} - \Delta \textbf{p}) = - f (\textbf{p} + \Delta \textbf{p})$$

by replacing $\Delta \textbf{p}$ to $\textbf{N} \Delta t$, we can prove the theorem.

🤔 **NOTE: The paper also proves the effectiveness of the loss function that penalizes** $\nabla f (\textbf{p}) = \textbf{0}$. **For details, please refer to the paper.** 🤔

## Optimization Visualization

<img class="img-fluid" src="/assets/post-images/NeuralPull/NeuralPull_fig3.png">
<span class="caption text-muted">Figure 3. <b>Optimization visualization on a 2D case</b>.</span>

Here's the description for each of these figure:

- **(a)**: A circle $\textbf{P}$ that Neural-Pull will learn.
- **(b)**: Regions where query points $\textbf{q}_i$s are sampled. Each quadrant is colored to track the pulled points after pulling.
- **(c)**: Result of pulling. One can see that Neural-Pull successfully translated query points in the relevant region onto the surface $\textbf{P}$.
- **(d)**: Unsigned distance field obtained from the learned signed distance field (taking absolute value will suffice).
- **(e)**: Sign of learned signed distance field.

The result justifies the effectiveness of the proposed method.

## Training

For ground truth, the authors randomly sampled $J = 2 \times 10^{4}$ points $\textbf{p}_{j}$ from point clouds formed by $1 \times 10^{5}$ points released by OccNet.

As mentioned previously, 25 query points $ \textbf{q}\_i $ are sampled (with adaptive sampling strategy) for each point $\textbf{p}_{j}$ forming the corresponding query location set $\textbf{Q}$, such that $i \in [1, I]$ and $I = 25 \times J = 5 \times 10^{5}$.

During training, 5000 query points are randomly chosen from $\textbf{Q}$ as a batch to train the network. Two different sampling strategies were examined during the experiment:

1. Random sampling directly from $\textbf{Q}$.
2. First uniformly sample 5000 points from $\textbf{P}$, and then select one query point per each sampled point on the ground truth surface.

While the second method is expected to perform well since the uniform sampling over the target surface can ensure wider coverage, the experimental results show that both of two ways perform well.

The authors used a neural network similar to OccNet to learn the signed distance function.

Adam optimizer with an initial learning rate of 0.0001 was used and the network was trained for 2500 epochs. Furthermore, the network parameters were initialized using the geometric network initialization (GNI) to approximate the signed distance function of a sphere.

# Experiments and Analysis

🤔 **NOTE: This post only covers a small portion of experimental result (mostly qualitative). For details, please refer to the original paper.** 🤔

## Surface Reconstruction from Point Clouds

Neural-Pull was used to reconstruct 3D surfaces from point clouds. Given a point cloud $\textbf{P}$, and in this case the condition $\textbf{c}$ was not used, the neural network was overfitted to the shape. After training, Neural-Pull becomes a neural representation for a shape in 3D, and it's visualized by applying marching cubes to reconstruct the mesh surface. Below figures show the reconstruction results & comparisons on different datasets.

<img class="img-fluid" src="/assets/post-images/NeuralPull/NeuralPull_fig4.png">
<span class="caption text-muted">Figure 4. <b>Comparison under FAMOUS in surface reconstruction</b>.</span>

<img class="img-fluid" src="/assets/post-images/NeuralPull/NeuralPull_fig5.png">
<span class="caption text-muted">Figure 5. <b>Comparison under ABC in surface reconstruction</b>.</span>

<img class="img-fluid" src="/assets/post-images/NeuralPull/NeuralPull_fig6.png">
<span class="caption text-muted">Figure 6. <b>Comparison under ShapeNet in surface reconstruction</b>.</span>

## Single Image Reconstruction

This time, Neural-Pull was used to reconstruct 3D shapes from 2D images. Here, the 2D image is considered as a condition $\textbf{c}$, which corresponds to a 3D shape represented as a point cloud $\textbf{P}$. **→ How can we train the network when we don't have any *ground truth points*?**

During training, a condition and a set of query points $\textbf{Q}$ are used to minimize the loss. And at test time, a 3D shape is reconstructed from an input image with a given condition. For encoding 2D images into features, 2D encoder used by SoftRas (Liu et al., 2019) was used. The result are as follows:

<img class="img-fluid" src="/assets/post-images/NeuralPull/NeuralPull_fig7.png">
<span class="caption text-muted">Figure 7. <b>Single image reconstruction comparison under ShapeNet subset</b>.</span>

<img class="img-fluid" src="/assets/post-images/NeuralPull/NeuralPull_fig8.png">
<span class="caption text-muted">Figure 8. <b>Single image reconstruction results using real world data</b>.</span>

# Conclusion

- The paper introduces Neural-Pull, a novel, effective way to learn signed distance functions from 3D point clouds by learning to *pull* 3D space onto the target surface.
- The network can learn an SDF **without ground truth SDF** by pulling sampled query points to their nearest neighbor on the surface during training.
- Experimental results show that Neural-Pull outperforms state-of-the-art methods in (1) surface reconstruction from 3D point clouds, and (2) single image reconstruction.