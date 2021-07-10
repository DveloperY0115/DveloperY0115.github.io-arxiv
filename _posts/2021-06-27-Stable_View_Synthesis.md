---
layout: post
title: "Summary of 'Stable View Synthesis'"
# use_math: true
background: '/assets/post-images/SVS/SVS_method_overview.png'
---

<img class="img-fluid" src="/assets/post-images/SVS/SVS_method_overview.png">
<span class="caption text-muted">Figure 1. <b>SVS Method Overview</b>.</span>

# Key Contributions

- A novel method for photorealistic view synthesis, which **embedds feature vectors** needed during rendering **onto a 3D geometric scaffold reconstructed** via various means.
- Image from new viewpoint can be rendered by *identifying view vectors at each point on the scaffold*, then *aggregating feature vectors but with known vectors together*, and finally *pass the feature vectors through convolutional layer* which gives us final output.
- Compared to the previous work, Free View Synthesis from the author, **no heuristic involves** when determining the feature vector associating a point on scaffold and output image pixels. Also, instead of using recurrent neural network, **set operators are used** when calculating the output feature vector. These make the model more stable.
- **Outperforms state-of-the-art methods that are already out there by significant margin**. The method improves widely used metrics such as LPIPS, PSNR, and SSIM on real world datasets.

# Key Concepts

## Overview

**The input for SVS is a set of images** $\\{ \mathcal{I}\_n \\}_{n=1}^{N}$, which are used to obtain a geometric scaffold $\Gamma$. These images are used to determine the basis for the feature vectors embedded on the surface.

**The goal is to render a plausible, realistic image** $\mathcal{O}$ that would've seen from a new viewpoint specified by transform $(\textbf{R}_t, \textbf{t}_t)$ and camera intrinsic $\textbf{K}_t$.

At the core is a 3D geometric scaffold. In order to build it, various methods like standard SfM (Structure from Motion), MVS (Multi-View Stereo), and surface reconstruction are used.

First, SfM is ran to extract camera instrinsics $ \\{ \textbf{K}\_n \\}_{n=1}^{N} $ as well as camera poses such as rotation matrices $ \\{ \textbf{R}\_n \\}\_{n=1}^{N} $ and translation vectors $ \\{ \textbf{t}\_n \\}\_{n=1}^{N} $ from the input images.


<img class="img-fluid" src="/assets/post-images/SVS/SVS_SfM.png">
<span class="caption text-muted">Figure 2. <b>Visualization of Structure from Motion</b>. This classical deterministic method extracts camera poses as well as intrinsics from a set of input image, making a foundation for the reconstructed scene. Source: Vladlen Koltun: Towards Photorealism (Sep. 2020)</span>

**Important notification here is that the symbol $ \\{ \mathcal{I}\_n \\}_{n=1}^{N} $ will be used to denote the rectified images after applying SfM.**

And then MVS is ran on the posed images, to obtain per-image depth maps, and these are fused into a (colored) point cloud.

<img class="img-fluid" src="/assets/post-images/SVS/SVS_MVS.png">
<span class="caption text-muted">Figure 3. <b>Visualization of Multi-View Stereo</b>. With the output (the images and their associated viewpoint information) from SfM, the authors applied MVS to obtain depth maps for each image, and fused them into a point cloud. Source: Vladlen Koltun: Towards Photorealism (Sep. 2020)</span>

The last part of the preprocessing is to apply Delaunay-based 3D surface reconstruction to the point cloud from the previous step to get a 3D surface mesh $\Gamma$.

<img class="img-fluid" src="/assets/post-images/SVS/SVS_surface_reconstruction.png">
<span class="caption text-muted">Figure 4. <b>Visualization of Surface Reconstruction</b>. The final output from the preprocessing step, which is fed into the neural network. These surfaces will be the place where the spatially-varying feature vectors will be installed. Source: Vladlen Koltun: Towards Photorealism (Sep. 2020)</span>

At the same time, **each input image $\mathcal{I}_n$ is encoded by a convolutional network to obtain a feature tensor $\mathcal{F}_n$ containing feature vector per each pixel** of the image. 

Note that **these features will later be back-projected onto the surfaces** we've reconstructed so far, and  **aggregation & re-projection follow to determine the per pixel feature vector at arbitrary viewpoints**.

More rigorously, in order to generate the new view $\mathcal{O}$, the pixels of $\mathcal{O}$ are back projected onto the scaffold $\Gamma$. Then, for each point $\textbf{x} \in \Gamma$ obtained by this back projection, we query the set of input images in which $\textbf{x}$ is visible. For such image $\mathcal{I}_k$, we can obtain a feature vector $\textbf{f}_k$ along the corresponding ray $\textbf{v}_k$ to $\textbf{x}$. 

By doing so, we can **construct a set $\{ (\textbf{v}_k, \textbf{f}_k) \}_k$ of view rays and associated feature vector**. These **elements are then processed by differentiable "set" network which is conditioned on the output view direction $\textbf{u}$**. The network output is the feature vector $\textbf{g}$ corresponding to $\textbf{u}$.

Feature vector $\textbf{g}$ is obtained for every pixel in the output image $\mathcal{O}$, forming **a feature tensor $\mathcal{G}$ which then can be decoded by a convolutional neural network** to produce the final render.

One important takeaway here is that each 3D point aggregates features from a number of source images independent of others.

## Image Encoding

Each source image $ \mathcal{I}\_k $ is encoded into a feature tensor by a U-Net-like convolutional network. Again, *remember that the source image here denotes the input images after preprocessing* steps explained earlier. From now on, this encoding network is denoted by $\phi_{enc}$.

**The encoder part of $\phi_{enc}$ consists of an ImageNet-pretrained ResNet18**, and parameters for batch normalization are fixed.

In **decoder** part, each stage upsamples the feature map, concatenates it with the feature map having the same resolution from the encoder, and convolution and activation layers are applied.

The resulting feature tensor of image is denoted by $ \mathcal{F}\_n = \phi_{enc}(\mathcal{I}_n) $.

## On-Surface Aggregation

<img class="img-fluid" src="/assets/post-images/SVS/SVS_on_surface_aggregation.png">
<span class="caption text-muted">Figure 5. <b>On-surface aggregation</b>. A point on the scaffold is seen in multiple images, with known viewing directions. And features from each of these are images are already encoded into feature tensors. Thus, when one need to predict a feature vector of the point, but seen from different direction, a neural network may help us by interpolating feature vectors in reasonable way.</span>

🤔 **This is the heart of this work. READ CAREFULLY!** 🤔

Think about common rendering techniques in computer graphics. One of the steps in rendering is to find which points & material in the scene contribute to which pixels. This idea is applied in similar manner. **Given a point $\textbf{x} \in \Gamma \subset \mathbb{R}^3$ on the 3D geometric scaffold, for each viewing direction $\textbf{u}$ (I think this models the spatially-varying BRDFs in physically based rendering), a feature vector $\textbf{g}(\textbf{x}, \textbf{u})$ is calculated**. Here, $\textbf{u}$ is the viewing direction vecctor from the target camera center to the surface point $\textbf{x}$.

Actually, there is another argument for $\textbf{g}$, which is the set $ \\{ (\textbf{v}\_k, \textbf{f}\_k(\textbf{x}))\\}_{k=1}^{K} $ where:

- $ \{ \textbf{f}\_k(\textbf{x}) \}_{k=1}^{K} $: The set of features defined at $\textbf{x}$ on the scaffold, extracted from source images where $\textbf{x}$ is visible. Note that each $\textbf{f}_k$ is from the image encoding $\mathcal{F}\_k$ by querying the encoding with point $\textbf{x}$.
- And $ \\{ \textbf{v}\_k \\}_{k=1}^{K} $ are corresponding viewing directions. Note that these could be obtained during preprocessing.

Specifically, $\textbf{f}_k(\textbf{x}) = \mathcal{F}_k(\textbf{K}_k(\textbf{R}_k \textbf{x} + \textbf{t}_k))$ using bilinear interpolation. → Maybe this implies the estimated viewpoint information where source images were taken?

Formally, the target feature vector for a given 3D surface point $\textbf{x}$ and associated novel viewing direction $\textbf{u}$ is computed as:

$ \textbf{g}(\textbf{x}, \textbf{u}) = \phi_{aggr}(\textbf{u}, \\{ (\textbf{v}\_k, \textbf{f}\_k(\textbf{x})) \\}_{k=1}^{K}) $

Here, $K$ is the number of source images that $\textbf{x}$ is visible. And here are important questions:

- **How can we find such $K$ number of images among total $N$ source images?**
- **What the design of $\phi_{aggr}$ is appropriate for our purpose? It should be differentiable and permutation-invariant (otherwise, different order of source images might change the result) → *Differentiable SET operators***

To tackle this issue, the authors tested multiple candidates and chose the one which performs the best.

One possible, yet simple design for $\phi_{aggr}$ is a **weighted average**, where the weights are based on the alignment between the source and target directions *(quantifying similarity with cosine)*:

$$\phi_{aggr}^{WA} = \frac{1}{W} \sum_{k=1}^{K} \text{max}(0, \textbf{u}^{T}\textbf{v}_k)\textbf{f}_k(\textbf{x})$$

Here, $W = \sum_{k=1}^{K} \text{max}(0, \textbf{u}^T\textbf{v}_k)$ is the sum of all weights.

Another candidate is **inspired by PointNet**, which is a well-known architecture for aggregating information in point clouds. Especially, the source and target directions are concatenated, and then passed to an MLP, eventually aggregated by permutation-invariant operator such as $\text{mean}$ or $\text{max}$. More precisely:

$$\phi_{aggr}^{MLP} = \nu_{k=1}^{K} \text{MLP}(\textbf{f}^{\prime}_k)$$

where $\textbf{f}^{\prime}_{k} = [ \textbf{u}, \textbf{v}_k, \textbf{f}_k(\textbf{x})]$ is the concatenation of source and target directions with the feature vector, and $\nu$ is a permutation-invariant operator.

Varation of approach is to use a **graph attention network (GAT)** that operates on a fully-connected graph between the source view per 3D point.

$$ \phi_{aggr}^{GAT} = \nu_{k=1}^{K} \text{GAT}(\{ \textbf{f}^{\prime}\_{k}\}\_{k=1}^{K}) \vert_{k} $$

where $\cdot \vert_k$ is the readout of the feature vector on node $k$.

Alternatively, we can even append $(\textbf{u}, \textbf{g}^{\prime})$ into consideration by forming fully connected graph over set $ \\{ (\textbf{u}, \textbf{g}^{\prime}) \\} \cup \\{ (\textbf{v}\_k, \textbf{f}\_k(\textbf{x}))\\}_{k=1}^{K} $. Here, $\textbf{g}^{\prime}$ is initialized by following the "weighted average" method described in the beginning. Then, the modified aggregation function becomes:

$$\phi_{aggr}^{\text{GAT-RO}} = \text{GAT} (\{ (\textbf{u}, \textbf{g}^{\prime})\} \cup \{ (\textbf{v}\_k, \textbf{f}\_k(\textbf{x}))\}_{k=1}^{K}) \vert_0$$

where $\cdot \vert_0$ denotes the readout of the feature vector associated with the target node.

## Rendering

So far, we discussed how feature tensor $\mathcal{G}$ used for final render is computed but the followings were not yet covered:

- How the surface points $\textbf{x}$ are obtained
- How the output image $\mathcal{O}$ is rendered from feature tensor $\mathcal{G}$

Suppose that a novel viewpoint is specified by camera intrinsic $\textbf{K}_t$, and camera pose $(\textbf{R}_t, \textbf{t}_t)$.

First, a depth map $\mathcal{D} \in \mathbb{R}^{H \times W}$ is calculated from 3D geometric scaffold $\Gamma$.

Then, each pixel center of the target image is unprojected back to 3D space based on the depth map $\mathcal{D}$, giving us **the set of surface points** $ \\{ \textbf{x}\_{h, w} \\}_{h,w=1,1}^{H \times W} $ corresponding to each pixel in $\mathcal{O}$. **Due to the incompleteness of $\Gamma$, some depth values might not be valid for some pixels.** For example, a point could be unprojected into the void(...) or background. For such cases, $\infty$ is assigned to depth value.

Now we have 3D surface points associated with each pixels of the target image $\mathcal{O}$ as well as their viewing directions $\textbf{u}$s, the view-dependent feature vectors $ \\{ \textbf{g}(\textbf{x}\_{h, w}) \\}\_{h,w}^{H \times W} $ can be calculated by means explained earlier, and eventually form a feature tensor $ \mathcal{G} = [\textbf{g}\_{h, w}]\_{h,w = 1, 1}^{H \times W} $. If a 3D surface $ \textbf{x}\_{h, w} $ didn't have any source image seeing it, simply assign 0 to $ \textbf{g}_{h,w} $.

At last, to synthesize the image $\mathcal{O}$ from the feature tensor $\mathcal{G}$, a convolutional neural network $\phi_{\text{render}}$ is used, that is $\mathcal{O} = \phi_{\text{render}}(\mathcal{G})$. For $\phi_{\text{render}}$, a sequence of $L$ U-Nets are used, and the result of previous U-Net is fed to the one at the next step, along with the original $\mathcal{G}$. Rigorously,

$$\phi_{\text{render}}(\mathcal{G}) = \phi^{L}\_{\text{render}}(\mathcal{G} + \phi_{\text{render}}^{L-1}(\mathcal{G} + \dots)) $$

# Training

## Training a scene-*agnostic* model

**→ Network is not aware of which scene it's learning. Many different scenes are sampled and shown to the network in the procedure.**

Three networks $(\phi_{\text{enc}}, \phi_{\text{aggr}},$  and $\phi_{\text{render}})$ are trained end-to-end.

Given a set of scenes, iterate the following process.

1. Sample *"a scene"* and *"a source image"* $\mathcal{I}_n$ that will serve as ground truth
2. From the remaining source images of the scene, a subset of $M$ source images are used as one batch for training
3. Calculate the loss and minimize it

The loss is of form:

$$ \mathbb{L}(\mathcal{O}, \mathcal{I}\_n) = \vert\vert \mathcal{O} - \mathcal{I}\_n \vert\vert_{1} + \sum_{l} \lambda_{l} \vert\vert \phi_{l}(\mathcal{O}) - \phi_{l}(\mathcal{I}_n) \vert\vert_1 $$

where $\phi_l$ are the outputs of some convolutional layers of a pretrained VGG-19 network.

For optimization, Adam is used with $\alpha = 1 \times 10^{-4}$, $\beta_1 = 0.9$, $\beta_2 = 0.9999$, and $\epsilon = 1 \times 10^{-8}$.

## Network fine-tuning

The scene-agnostic training introduced above produces a general network that can be applied to new scenes without retraining or fine-tuning. However, it is possible that the scene that the network is trained on is quite different from what it's about to applied on.

The common solution to fine-tune the network parameters $\theta = [\theta_{\text{enc}}, \theta_{\text{aggr}}, \theta_{\text{render}}]$ on sorce images of the target scene by starting from the pretrained scene-agnostic we have, applying the same training strategy, but this time provide source images of the target scene as sample ground truth $\mathcal{I}_n$.

## Scene fine-tuning

The more powerful method which can significantly improve the result is to **optimize the source images** along with the parameters of $\phi_{\text{enc}}$ as well. The origin of this idea is the point where we notice that the final output $\mathcal{O}$ is a function of the "encoded" source images $ \{ \phi_{\text{enc}}(\mathcal{I}\_m; \theta_{\text{enc}})\}\_{m=1}^{M} $ that are used as input of subsequent networks such as $ \phi_{\text{aggr}} $, $ \phi_{\text{render}} $.

Until now, only $\theta_{\text{enc}}$ are trained through gradient descent. However, we can **come up with an idea of optimizing the encoded source images $ \\{ \phi\_{\text{enc}}(\mathcal{I}\_m; \theta\_{\text{enc}})\\}\_{m=1}^{M} $ together**.

To achieve it, the image encoder should undergo little modification. Specifically, the image encoder becomes a function of form $\phi_{\text{enc}}(m; \theta_{\text{enc}}, \theta_{\text{imgs}})$. In words, the input of the network is altered from a source image $ \mathcal{I}\_m $ to the index $m$. This index points at one of the trainable parameters in $\theta_{\text{imgs}}$, a pool of trainable parameters associated with source images.

**In this context,** **$ \theta\_{\text{imgs}} $ are initialized with the "actual" source images but unlike actual ones, they are optimized** with the parameters of the image encoder network $\theta_{\text{enc}}$. Thus, the source images are now mutable, and can be optimized through the training process. Note that the encoder can be denoted by $\phi_{\text{enc}}(\theta_{\text{imgs}}[m]; \theta_{\text{enc}})$, which is more similar to the previous notation without source image optimization.

As a result of the above discussion, the optimization problem is now $ \text{argmin}\_{\theta, \theta_{\text{imgs}}} \mathbb{L}(\mathcal{O}, \mathcal{I}\_n) $. However, the training process is still the same. **One important point is that while there are trainable parameters $\theta\_{\text{imgs}}$ associated, and initialized with source images $ \\{ \mathcal{I}\_n \\}\_{n=1}^{N} $, but these source images must remain the same since they are used to calculate loss $ \mathbb{L}(\mathcal{O}, \mathcal{I}\_n) $**. Without this constraint on the originality, the network may learn in weird way, producing meaningless images in the end.