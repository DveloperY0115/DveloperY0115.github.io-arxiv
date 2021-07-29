---
layout: post
title: "Summary of 'Robust solving of optical motion capture data by denoising'"
subtitle: "Robust solving of optical motion capture data by denoising (SIGGRAPH 2018)"
background: '/assets/post-images/RobustSolvingofOpticalMotionCaptureDataByDenoising/background.png'
---

# Motivations

- Raw optical motion capture data often includes errors such as occluded markers, mislabeled markers, and high frequency noise of jitter.
- These errors are mostly fixed by hand, thus extremely time-consuming and tedious.
- We need a tool which can **directly infer joint transforms from raw marker data** and **robust** **to noises** at the same time.

# Key Contributions

- Introduces a robust neural network-based framework which can predict (global) joint transforms from raw marker data.

---

# TL;DR

This work proposes a set of preprocessing methods and a ResNet-like neural network to predict joint transforms from raw marker data. Unlike traditional solvers for inverse-kinematics, this work suggests several steps to build a framework for such task in data-driven manner.   

# Methods

The method starts with two inputs:

1. A large database of skeletal motion capture data (CMU motion capture database)
2. A set of marker configurations from a variety of different capture subjects

## Preprocessing

### Scaling

All motion data is scaled so that the figure (character) has a uniform height. By doing so, we eliminate a need to deal with characters of different heights, then **focus on dealing with characters with different proportions**.

An appropriate scaling factor can be computed by:

1. Computing from T-pose using the average length of the character's joints
2. Extracted directly from the motion capture software

Other than this, there is no assumption about the capture subject. The system is designed to operate on variety of subjects of different body shapes and proportions.

### Representation

Given a character of $j$ joints, and a dataset of $n$ poses (i.e. there are typically $n$ frames representing a single motion), we represent animation data using the joints' global homogeneous transformation (Affine) matrices $\textbf{Y} \in \mathbb{R}^{n \times j \times 3 \times 4}$.

Given $m$ markers, the local marker configurations for each of the $n$ different poses in the database are represented using the local offset from each marker to each joint $\textbf{Z} \in \mathbb{R}^{n \times m \times j \times 3}$.

And the skinning weights associated with these marker offsets (how much a marker contributes to each joint in the skeleton) are represented as $\textbf{w} \in \mathbb{R}^{m \times j}$. For joints which are not assigned to a given marker the skinning weight and offset are set to zero.

Note that these local offsets almost always remain constant across motions of the same actor (relative position between markers and joints rarely change during the process of motion capture). However, this framework supports time-varying offsets as well.

### Linear Blend Skinning

Using the data represented in the form described above, one can compute a set of *global* marker positions $\textbf{X} \in \mathbb{R}^{n \times m \times 3}$ using the linear blend skinning function $\textbf{X} = \text{LBS}(\textbf{Y}, \textbf{Z})$ whose definition is as follows:

$$\text{LBS} (\textbf{Y}, \textbf{Z}) = \sum\_{i=0}^{j} \textbf{w}\_{i} \odot (\textbf{Y}\_{i} \otimes \textbf{Z}\_{i}).$$

The global position of marker is determined as **a weighted sum of transformed local offsets**.

More specifically, the symbol $\otimes$ stands for the homogeneous transformation matrix multiplication of each of the $n$ marker offsets in $\textbf{Z}\_{i}$ by the $n$ transformation matrices $\textbf{Y}\_{i}$, computed for each of the $m$ markers.

And the symbol $\odot$ represents a component-wise multiplication, weighting the contribution of each of the $j$ joints in the resulting transformed marker positions, computed for each of the $n$ poses.

While alternative skinning methods are available, this work is done using LBS. This implies that using better skinning algorithm may improve results (but not certain).

### Local Reference Frame

<center>
    <img class="img-fluid" src="/assets/post-images/RobustSolvingofOpticalMotionCaptureDataByDenoising/fig1.png">
</center>
<span class="caption text-muted">Figure 1. <b>Overview of rigid body fitting procedure</b>.</span>

One important point when solving a problem with data-driven approach is that all the data in the dataset must share a consistent schema. In the case of motion data, **defining a proper reference frame is crucial** for the performance of the framework.

We want to find a local reference frame which well describes the data, but at the same time, doesn't require knowing the joint transforms in advance. To this end, this work used rigid body alignment (Besl and McKay, 1992).

More precisely, we first choose a subset of markers around the torso and then compute the mean relative location of these markers with respect to a joint of our choice (usually one of the spine joints). This computation is done for all $n$ given poses, yielding a set of $n$ reference frames $\textbf{F} \in \mathbb{R}^{n \times 3 \times 4}$ where the data can be described locally.

### Statistics

Before jumping into the actual training, there are few statistics we need to calculated that will be used throughout training.

First, we compute **the mean and standard deviation of the joint transformations** $\textbf{y}^{\mu} \in \mathbb{R}^{j \times 3 \times 4}$, $\textbf{y}^{\sigma} \in \mathbb{R}^{j \times 3 \times 4}$.

Second, **the mean and covariance of the marker configurations** $\textbf{z}^{\mu} \in \mathbb{R}^{m \times j \times 3}$, $\textbf{z}^{\Sigma} \in \mathbb{R}^{(m \times j \times 3) \times (m \times j \times 3)}$ are computed.

Third, **the mean and standard deviation of marker locations** (which is derived by applying LBS on $\textbf{Y}$ and $\textbf{Z}$) $\textbf{x}^{\mu} \in \mathbb{R}^{m \times 3}$, $\textbf{x}^{\sigma} \in \mathbb{R}^{m \times 3}$ are computed.

Finally, the data called ***pre-weighted* local offsets** $\hat{\textbf{Z}} \in \mathbb{R}^{n \times m \times 3}$ is calculated following the definition $\hat{\textbf{Z}} = \sum\_{i=0}^{j} \textbf{w}\_{i} \odot \textbf{Z}\_{i}$ as well as their **mean and standard deviation** $\hat{\textbf{z}}^{\mu} \in \mathbb{R}^{m \times 3}$, $\hat{\textbf{z}}^{\sigma} \in \mathbb{R}^{m \times 3}$ (looks like there's a mistake in the paper).

This pre-weighted local offsets are used as an additional input to the neural network which will be introduced later. The reasons for using pre-weighted values instead of the entire set of offsets are:

- This helps the neural network distinguish between characters with different body proportions or marker placements.
- Providing the network with the full set of local offsets $\textbf{Z}$ is inefficient since it requires $m \times j \times 3$ values.

For marker skinned to a single joint (and this is the case for most of markers), the pre-weighted offset is identical to the offset from the marker to the joint. And for markers skinned to multiple joints, it will be a weighted sum of the different offsets. Thus, we can say that this pre-weighted offset can serve as a good representitive for the entire offsets. Moreover, this reduces the size of data while retaining most of information.

## Training

<center>
    <img class="img-fluid" src="/assets/post-images/RobustSolvingofOpticalMotionCaptureDataByDenoising/fig2.png">
</center>
<span class="caption text-muted">Figure 2. <b>Overall Pipeline</b>.</span>

The network works on a per-pose basis, taking a batch of $n$ frames of marker positions $\textbf{X}$ and the associated *pre-weighted* marker configurations $\hat{\textbf{Z}}$. And it outputs a corresponding batch of $n$ joint transforms $\textbf{Y}$ (homogeneous transform matrices of all joints for each of $n$ given frames).

<center>
    <img class="img-fluid" src="/assets/post-images/RobustSolvingofOpticalMotionCaptureDataByDenoising/fig3.png">
</center>
<span class="caption text-muted">Figure 3. <b>Overall architecture of the network</b>.</span>

In a nutshell, the structure of the network used in this paper is a six layer feed-forward ResNet. Each ResNet block of the network uses 2048 hidden units, followed by ReLU activation function. The weights of these layers are initialized following the "LeCun" initialization scheme. Since artificial noises are added to the training dataset, there's no further need for regularization such as dropout.

<center>
    <img class="img-fluid" src="/assets/post-images/RobustSolvingofOpticalMotionCaptureDataByDenoising/fig4.png">
</center>
<span class="caption text-muted">Figure 4. <b>Training algorithm</b>.</span>

The network is trained by following the steps described below:

1. Given a mini-batch of $n$ poses $\textbf{Y}$, a batch of $n$ marker configurations $\textbf{Z}$ is sampled from normal distribution whose mean and covariance is $\textbf{z}^{\mu}$ and $\textbf{z}^{\Sigma}$, respectively.
2. Compute the marker positions $\textbf{X}$ using linear blend skinning: $\textbf{X} = \text{LBS} (\textbf{Y}, \textbf{Z})$.
3. Corrupt marker positions using the function $\text{Corrupt}$, obtaining $\hat{\textbf{X}} = \text{Corrupt}(\textbf{X})$.
4. Compute the *pre-weighted* marker offsets $\hat{\textbf{Z}}$ to summarize the sampled marker configuration.
5. Normalize the computed marker positions and summarized local offsets using the means and standard deviations computed during preprocessing.
6. Forward propagation from the input variables to the predicted global joint tranforms $\hat{\textbf{Y}}$.
7. Denormalize $\hat{\textbf{Y}}$ using previously computed $\textbf{y}^{\mu}$ and $\textbf{y}^{\sigma}$.
8. Compute loss consists of two terms: (1) $L_{1}$ distance between $\textbf{Y}$ and $\hat{\textbf{Y}}$, and (2) $L_{2}$ regularization loss with $\gamma = 0.01$.
9. Compute gradient, and update weights using AmsGrad algorithm.

This procedure is repeated with a mini-batch size of 256 until the training loss converges or the validation loss increases.

Note that there are several hyperparameters in the training routine. The most important one is the user weights $\lambda$. These weight determine the amount of penalty for making incorrect predictions for certain (chosen) joints. That is, some parts may receive more attention than others. For example, one can set the weights accounting for joints at arms higher than others. Then the network trained using this loss is more likely to predict transforms of arm joints with comparably higher accuracy than other parts.

<center>
    <img class="img-fluid" src="/assets/post-images/RobustSolvingofOpticalMotionCaptureDataByDenoising/fig5.png">
</center>
<span class="caption text-muted">Figure 5. <b>Definition of function Corrupt</b>. Used to introduce noises (random occlusion, shifting) to training dataset.</span>

The key for achieving robustness to noises is the corruption function whose overall mechanism is shown above. This algorithm is designed to emulate marker occlusions, marker swaps, and marker noise that often occur during motion capture. This function mimics and simulates such circumstances either by removing markers or adjusting their positions in a stochastic way.

The author empirically discovered the fact that introducing random translation to markers' positions is more effective than swapping the positions of markers in the dataset directly.

The parameters $\sigma^{o}$, $\sigma^{s}$, $\beta$ of this function control the levels of corruption applied to the data. The role of each of them is as follows:

- $\sigma^{o}$: Adjusts the probability of a marker being occluded
- $\sigma^{s}$: Adjusts the probability of a marker being shifted out of place
- $\beta$: Controls the scale of random translations applied to shifted markers

In practice, each of them is set to 0.1, 0.1, and 50 cm, respectively. These numbers are determined so that the network can learn to fix corrupted data, and the information of original animation is preserved.

### Runtime

After training, we can apply this framework to predict Affine transform of each joints of the skeleton from raw marker position data. However, there are several steps required before passing the data to the network:

1. *Outlier Removal*
2. *Solving*
3. *Filtering*
4. *Retargeting*

***Outlier Removal***

To remove outliers the author used a basic outlier removal technique to do so. This technique uses the distances between markers to find badly-positioned markers those possibly hinder the network to function properly. If found, any outlier markers are considered occluded. 

To eliminate outliers, the algorithm first starts by computing the pairwise marker distance $\textbf{D} \in \mathbb{R}^{n \times m \times m}$ for each frame **in the training data** $\textbf{X}$. From this, one can compute the minimum, maximum, and range of distances across all frames $\textbf{D}\_{\text{min}} \in \mathbb{R}^{m \times m}$,  $\textbf{D}\_{\text{max}} \in \mathbb{R}^{m \times m}$, $\textbf{D}\_{\text{range}} = \textbf{D}\_{\text{max}} - \textbf{D}\_{\min}$.

<center>
    <img class="img-fluid" src="/assets/post-images/RobustSolvingofOpticalMotionCaptureDataByDenoising/fig6.png">
</center>
<span class="caption text-muted">Figure 6. <b>Visualization of the distance range matrix</b>. Markers close to each other represent more rigidly associated markers. Markers that are too close or too far to each other are detected by the algorithm and removed.</span>

At test phase, given a new set of marker pairwise distances for every frame in the input $\hat{\textbf{D}} \in \mathbb{R}^{n \times m \times m}$, the algorithm iteratively process each frame by repeatedly removing the marker which violates the following set of constraints to the greatest degree:

$$ \begin{gather} \hat{\textbf{D}} < \textbf{D}\_{\min} - \delta \textbf{D}\_{\text{range}}, \\
\hat{\textbf{D}} > \textbf{D}\_{\max} + \delta \textbf{D}\_{\text{range}}. \end{gather} $$

The algorithm repeats elimination until there's no violation found. The sensitiveness of the algorithm is determined by the value of $\delta$, which is set to 0.1 in this work.

***Solving***

After removing outliers, the local reference frame $\textbf{}$$\textbf{F}$ is derived based on the positions of remaining marker positions and a joint of our choice. And then, the position of occluded markers are set to zero since we've set the position of occluded markers to zero during training. Finally these data - marker positions and pre-weighted marker configurations - are passed to the neural network. The neural network then produces joint transforms that are later transformed back to the world space using the inverse reference frame $\textbf{F}^{-1}$.

***Filtering***

Since this method runs on a per-pose basis, the prediction of joint transforms might be temporally inconsistent. That is, it's possible for a large number of markers appear or disappear across different frames yielding jittery movements in the resulting joint transforms. To address this issue, the author adopted a Savitzky-Golay filter as a basic post-processing method.

***Retargeting***

<center>
    <img class="img-fluid" src="/assets/post-images/RobustSolvingofOpticalMotionCaptureDataByDenoising/fig7.png">
</center>
<span class="caption text-muted">Figure 7. <b>Comparison showing the result before and after retargeting process</b>.</span>

After filtering, there's only one step left before reaching the final goal - predicting the transforms of joints in each frame. One minor flaw of using the neural network for predicting transformation matrices is that the predicted matrices may not represent valid transformations. For example, for a matrix to represent rotation, it must be orthonormal, and has determinant of 1. However, there's no guarantee that the matrices from the network satisfiy such conditions. Therefore, it's better to orthogonalize them (to be more specific, rotational components of homogeneous transform matrices) using SVD. Note that Gram-Schmidt can be an alternative to it.

Finally, the author used Jacobian Inverse Kinematics solver to extract *local (hierarchical)* joint transformations from the set of joint transforms represented in the context of world frame.

# Results (Qualitative Only)

**🤔 NOTE: This post presents only qualitative results from the paper. For details such as quantitative comparisons, please refer to the original paper. 🤔**

For clarification, the **production character** is a high quality, complex character skeleton from a game development environment in production and uses high quality motions captured for use in games.

On the other hand, the research character uses more simple custom marker layout, with the skeleton structure and animation data derived from CMU motion capture dataset.

<center>
    <img class="img-fluid" src="/assets/post-images/RobustSolvingofOpticalMotionCaptureDataByDenoising/fig8.png">
</center>
<span class="caption text-muted">Figure 8. <b>Results of the method applied to raw motion data</b>. Left: Raw uncleaned data. Middle: This method. Right: Hand cleaned data.</span>

<center>
    <img class="img-fluid" src="/assets/post-images/RobustSolvingofOpticalMotionCaptureDataByDenoising/fig9.png">
</center>
<span class="caption text-muted">Figure 9. <b>Results of the method applied to motion data corrupted by the custom noise function</b>. Left: This method. Right: Ground Truth. Top: Production character. Bottom: Research character.</span>

<center>
    <img class="img-fluid" src="/assets/post-images/RobustSolvingofOpticalMotionCaptureDataByDenoising/fig10.png">
</center>
<span class="caption text-muted">Figure 10. <b>Results of the method applied to motion where half of the markers are removed</b>. Left: This method. Right: Ground Truth. Top: Production character. Bottom: Research character.</span>

# Conclusion

- Introduced a neural network based framework that produces a set of joint transforms for each joint in the given skeleton from a set of marker position and their contributions - weights - to each joint.
- While this work seems to be able to reduce the burden of correcting marker data manually, it often **results in inaccurate joint transform prediction when encountered motion that was not seen previously** (limitation of data-driven approach).
- Also, due to the per-pose basis of this method, it still needs traditional algorithms to smooth-out unnatural trajectories in predicted motion (animation) caused by **temporal inconsistency**.