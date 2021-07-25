---
layout: post
title: "Summary of 'Texture Fields: Learning Texture Representations in Function Space'"
# use_math: true
background: '/assets/post-images/TextureFields/texturefields_method_overview.png'
---

<img class="img-fluid" src="/assets/post-images/TextureFields/texturefields_method_overview.png">
<span class="caption text-muted">Figure 1. <b>Method Overview</b>. Illustrates high-level idea of Texture Fields</span>

# Method Summary

- Introduce the idea of **vector field which maps a point in 3D space to a point in color space**, designed to assign proper color to each point on the surface of an object.
- **Three jointly connected networks**, each playing different role in the pipeline, such as *capturing shape feature, extracting image feature, and predicting color vector*, are **trained in an end-to-end manner.**
- Conditional pipeline, which takes image embedding as an additional input, is trained with supervision (i.e. ground truth pixel value is known). **→** **Limitation: Images from multiple viewpoints (not necessarily the same as the conditioning image) must be rendered.**
- Unconditional pipeline, however, exploits probabilistic generative models such as GAN, VAE since we don't have information about the apperance of the object.
- **Overall, the idea is quite fresh, but computing loss in 2D image space sounds inefficient, and not elegant. We need to find a way to embedd the appearance information in 3D space directly.**

# Motivation

- **Texture reconstruction of 3D objects** has recevied little attention compared to 3D geometry reconstruction, or image generative models.
- Commonly used representations of texture are inefficient or hard to be integrated into deep learning pipeline.

# Key Contributions

- Introduces Texture Fields, texture representation based on **regressing a continuous 3D function** parametrized with a neural network. → walkaround limiting factors such as *shape discretization* and *parametrization*
- Texture representation independent of shape of the object.
- (Compared to previous methods) More efficient than voxel based texture representation, easier to generalize into various topologies.
- Combined with modern 3D reconstruction neural network, both 3D geometry and texture can be reconstructed end-to-end.
- Able to generate suitable textures given 3D shape model and ***latent texture code***.

# Key Concepts

## Texture Fields

In contrast to discretized 3D shape representation such as meshes or voxels, approaches exploiting implicit surface (e.g. signed distance function) is an ideal starting point for texture reconstruction.

Instead of embedding occupancy in the continuous function, we can think of embedding color information onto it. By combining the existing continuous function-based methods with the idea proposed, a *textured 3D model* can be reconstructed given:

1. Data for geometry reconstruction
2. An image for texture reconstruction

Since both geometry & texture reconstruction is done using the continuous function, the neural network approximating these can be optimized end-to-end.

Let $t$ denote a function mapping a 3D point $\textbf{p} \in \mathbb{R}^3$ to a point in color space $\textbf{c} \in \mathbb{R}^3$. Then the function $t$ is in fact a 3D vector field:

$$t: \mathbb{R}^3 \to \mathbb{R}^3$$

And this function will be parametrized with parameters of neural network, $\theta$. However, **we should add contraints on this function, which is a shape embedding** $\mathcal{s} \in S$. This enables the network to predict well-fitting texture for given geometry by exploiting contextual geometric information (e.g. surface discontinuities).

Further more, note that the geometry information alone cannot exactly specify what suitable texture for the given shape should be. To resolve this issue, we **condition the network on the 2D image taken from an arbitrary viewpoint.**

The image is encoded into a **viewpoint-invariant global feature** representation $\textbf{z} \in \mathcal{Z}$. Therefore, we don't need to consider camera extrinsincs of where the image was taken. Furthermore, the image doesn't necessarily depict the exact shape of 3D model. This is a huge advantage especially in real world applications where images contain limited information of 3D shapes.

To summarize, a Texture Field can be defined as a mapping from 3D point $\textbf{p}$, shape embedding $\textbf{s}$, and conditional (probably representation, or latent vector) $\textbf{z}$ to a point $\textbf{c}$ in color space:

$$t_{\theta} = \mathbb{R}^3 \times \mathcal{S} \times \mathcal{Z} \to \mathbb{R}^3$$

In addition, not only the conditional case where $t_{\theta}$ is conditioned on $\textbf{z}$, the paper also deal with unconditional ones which exploits probabilistic generative models such as VAE, and GAN.

## Model Details

<img class="img-fluid" src="/assets/post-images/TextureFields/texturefields_model_overview.png">
<span class="caption text-muted">Figure 2. <b>Model Overview</b>. Colored arrows show alternative pathways. Red for conditional, green for GAN, and blue for VAE model. The blue and red boxes denote trainable components of the model, parametrized through neural networks.</span>

### Shape Encoder

To generate shape embedding $\textbf{s}$, **points are sampled uniformly from the input shape** (typically triangular mesh) and they're **passed to a PointNet encoder**. This network architecture generates **fixed-dimensional shape embedding** $\textbf{s}$.

### Image Encoder (Conditional Model)

**An input image** depicting the appearance of 3D shape from specific viewpoint is **encoded into a fixed-dimensional latent code** $\textbf{z}$ using **pre-trained residual network** (ResNet).

### Texture Field

Given shape embedding $\textbf{s}$ and image latent code $\textbf{z}$, the Texture Field predicts a color value $\textbf{c}_i$ for any point $\textbf{p}_i$ on the surface of 3D shape. It sounds like we can color 3D meshes directly, but it's not an easy work since additional UV-mapping is required. Thus, **we shall train our model in 2D image space rather than raw 3D space** for regularity and efficiency.

To this end, **depth maps** $\textbf{D}$ and **corresponding color images** $\textbf{X}$ from **arbitrary viewpoints should be rendered**. In this case, OpenGL is used.

Then the color at pixel $\textbf{u}_i$ and depth $d_i$ is predicted as:

$$\hat{\textbf{c}\_i} = t_{\theta}(d_i \textbf{R}\textbf{K}^{-1}\textbf{u}_i + \textbf{t}, \textbf{s}, \textbf{z})$$

where $i$ denotes the index for pixels with finite depth values $d_i$ and $i \in \{1, ..., N\}$.

Here, $N$ denotes the number of foreground pixels in the rendered image (i.e. pixels where the object is visible). The camera intrinsic is denoted by $\textbf{K} \in \mathbb{R}^{3 \times 3}$, and extrinsics are denoted by $\textbf{R} \in \mathbb{R}^{3 \times 3}$ (orientation), and $\textbf{t} \in \mathbb{R}^3$ (translation), respectively. And pixel coordinate $\textbf{u}_i$ is represented in homogeneous coordinates.

The predicted color $\hat{\textbf{c}_i}$ is compared to the ground truth pixel color $\textbf{c}_i$ in the rendered image $\textbf{X}$ during training.

## Training

### Conditional Setting

In this case, the image embedding $\textbf{z}$ is passed to the network. The network $t_{\theta}(\textbf{p}, \textbf{s}, \textbf{z})$ is trained in a supervised setting by minimizing $\ell_1$-loss between the predicted image $\hat{\textbf{X}}$ and the rendered image $\textbf{X}$.

$$\mathcal{L}\_{\text{cond}} = \frac{1}{B}\sum\_{b=1}^{B}\sum\_{i=1}^{N_b} \vert\vert t_{\theta}(\textbf{p}_{b_i}, \textbf{s}\_b, \textbf{z}_b) - \textbf{c}\_{b_i} \vert\vert\_{1}$$

Here, $B$ stands for batch size. **Each element of the mini batch represents an image with $N_b$ foreground pixels.** Also, shape encoding $\textbf{s}_b$ and conditional image encoding $\textbf{z}_b$ depends on the parameters of the shape & image encoder networks (PointNet for shape, ResNet for image). Using the loss above, **three networks - shape encoder, image encoder, and Texture Field - are trained jointly.**

### Unconditional Setting

In the unconditional setting, **the model is given only the 3D shape as its input.** There's no information about the appearance of it. This is quite difficult problem to tackle - giving an object plausible appearance without any information about it - so we need to utilize probabilistic approaches.

First, we tackle this problem with *conditional GAN,* where the generator is conditioned on the 3D shape. Then, the generator is represented as a Texture Field $t_{\theta} : \mathbb{R}^3 \times \mathcal{S} \times \mathcal{Z} \to \mathbb{R}^3$ which maps the latent code $\textbf{z}$ for every given 3D location $\textbf{p}_i$ conditioned on the shape embedding $\textbf{s}$ to an RGB image $\hat{\textbf{X}}$:

$$\hat{\textbf{X}} = G_{\theta} (\textbf{z}\_b \vert \textbf{D}\_b, \textbf{s}\_b) = \\{ t_{\theta}(\textbf{p}_{b_i}, \textbf{s}_b, \textbf{z}_b) \vert i \in \\{ 1,..., N_b \\} \\}$$

The standard image-based discriminator $D_{\phi}(\textbf{X}_b \vert \textbf{D}_b)$ conditioned on the input depth image $D_b$ is used for training. That is, when passing the input image, the depth map is concatenated to it. Also, non-saturating GAN loss with [$R_1$-regularization](https://paperswithcode.com/method/r1-regularization) is used for training.

Secondary strategy for this problem is to use *conditional VAE* (cVAE)*.* In this setting, the **encoder network predicts mean $\mu$ and variance $\sigma$ of which the latent vector $\textbf{z}$ will be sampled from given image $\textbf{X}$ and shape embedding $\textbf{s}$. Then the Texture Field is now used as decoder, but this time taking sampled latent vector from the predicted distribution**, not extracted by image encoder in conditional setting. Then we minimize the following variational lower bound:

$$\mathcal{L}\_{VAE} = \frac{1}{B} \sum\_{b=1}^{B} [ \beta KL(q\_{\phi}(\textbf{z} \vert \textbf{X}\_b, \textbf{s}\_b) \vert\vert p\_0(\textbf{z}\_b)) + \sum\_{i=1}^{N\_b}\vert\vert t\_{\theta}(\textbf{p}\_{b_i}, \textbf{s}\_b, \textbf{z}\_b) - \textbf{c}\_{b_i} \vert\vert\_1]$$

Here, we assume that the distribution of "extracted" latent vector $\textbf{z}_b$ follows the standard normal distribution, that is $\textbf{z}_b \sim \mathcal{N}(\textbf{z}, 0, \textbf{I})$.

 **$\beta$ is a trade-off parameter** between the KL-divergence and the reconstruction loss (the second term), is usually set to 1 in practice. 

Also, as typical VAE models do, reparametrization trick is applied here.

# Implementation Details

## Quick Look

1. **Texture Field** $t_{\theta}(\cdot, \textbf{s}, \textbf{z})$: Fully connected ResNet
2. **Image Encoder**: ResNet-18 architecture. Pre-trained on ImageNet
3. **Shape Encoder**: Adopted version of PointNet
4. **GAN discriminator & VAE Encoder**: Adopted from models introduced in this [paper](https://www.notion.so/Which-training-methods-for-GANs-do-actually-converge-ab13511f53e440ceab2016344ebffd78).
5. Supervised (conditional) model and VAE is optimized with Adam with $\alpha = 1e-4$.
6. GAN is trained with alternating gradient using the RMSProp optimizer with $\alpha = 1e-4$.

## Texture Field

<img class="img-fluid" src="/assets/post-images/TextureFields/texture_fields_structure_overview.png">
<span class="caption text-muted">Figure 3. <b>Texture Field Structure Overview</b></span>

### Description

The novel architecture for generating texture color values for corresponding input points. While other components may vary, this one is used for all experiments introduced in this paper. The network consists of blocks of ResNet building blocks, **note that the embedding vectors $\textbf{s}$, $\textbf{z}$ are first concatenated, and then injected to EACH block.** Different number of ResNet blocks are used for different experiments:

- $L = 6$ for the single image texture reconstruction (conditional)
- $L=4$ for the generative models

### Inputs

1. A collection of $N$ 3D position vector $\textbf{p}_i$, where $i =1,..., N$
2. Fixed-length shape embedding vector $\textbf{s}$
3. Fixed-length latent vector $\textbf{z}$ (either extracted from provided image, or sampled from probability distributions)

### Outputs

1. A collection of $N$ color vector $\textbf{c}_i$ associated with each $\textbf{p}_i$, where $i = 1,..., N$

## Shape Encoder

<img class="img-fluid" src="/assets/post-images/TextureFields/texturefields_shape_encoder.png">
<span class="caption text-muted">Figure 4. <b>Shape Encoder Structure Overview</b></span>

### Description

PointNet based encoder for generating latent vectors corresponding to the given shapes. **Note that the shape embedding $\textbf{s}$ is a global feature of the input point cloud.**

### Inputs

1. Set of $M$ points of a point cloud sampled from the surface of target shape

### Outputs

1. Fixed-length shape embeddig vector $\textbf{s}$

## Image Encoder (Conditional setting ONLY)

<img class="img-fluid" src="/assets/post-images/TextureFields/texturefields_image_encoder.png">
<span class="caption text-muted">Figure 5. <b>Image Encoder Structure Overview</b></span>


### Description

ResNet based encoder for generating latent vectors corresponding to the given images.

### Inputs

1. Image depicting the appearance of the target shape

### Outputs

1. Fixed-length image embedding $\textbf{z}$

## VAE Encoder

<img class="img-fluid" src="/assets/post-images/TextureFields/texturefields_VAE_encoder.png">
<span class="caption text-muted">Figure 6. <b>VAE Encoder Structure Overview</b></span>

## GAN Discriminator

<img class="img-fluid" src="/assets/post-images/TextureFields/texturefields_GAN_discriminator.png">
<span class="caption text-muted">Figure 7. <b>GAN Discriminator Structure Overview</b></span>

## NVS

<img class="img-fluid" src="/assets/post-images/TextureFields/texturefields_NVS_structure.png">
<span class="caption text-muted">Figure 8. <b>Novel View Synthesis (NVS) Structure Overview</b> This module can replace the Texture Field, and used for ablation study discussed later.</span>

# Experimental Details

## Experiment Overview

The experiments can be categorized into three categories:

1. **Representation power**: analyze how well the Texture Field can represent high frequency textures when trained on a single 3D object
2. **Single view texture reconstruction**: Predict full texture of 3D objects given only the 3D shape and a single view of it.
3. **Generative setting**: Generate textures of 3D shapes without providing any image to the model. Instead, embedding $\textbf{z}$ is sampled from some distributions.

## Baseline Overview

There are three baselines for ablation studies:

1. **Projective texture mapping**: Exploit camera information to find corresponding color value for each vertex used in this paper.
2. **Novel-View-Synthesis (NVS)**: Uses the same image encoder, but apply UNet for predicting color values instead of Texture Field.
3. **Im2Avatar**: Voxel-based 3D shape & texture reconstruction pipeline. The official implementation was used.

## Dataset

1. 'cars', 'chairs', 'airplanes', and 'tables' categories from **ShapeNet** dataset is used.
2. For conditional setting, images passed to the image encoder were pre-rendered.
3. For images & depth maps passed to Texture Field, 10 images and depth maps were rendered per object from random viewpoints in the upper hemisphere of models.

## Metrics

Consider three different metrics in **image space**.

1. **Frechet Inception Distance (FID)**: Common metric between distributions of images and widely used in GAN models.
2. **Structure Similarity Image Metric (SSIM)**: More precise measurement for distance between predicted view and ground truth on a per-instance basis (i.e. not in the level of probability distributions, but in each individual images from each distributions).
3. **Feature-$\ell_1$-metric**: Captures the global properties of images, due to the characteristic of *SSIM capturing mostly local properties of images*. **This metric is computed as the mean absolute distance between two points in *feature space***, and **Inception network** is used to map predicted & ground truth images to the feature space.

## Experiment 1. Representation Power

The purpose of this experiment is to check the upper bound of the reconstruction quality by overfitting the Texture Field with training data. At the same time, voxel-based approach is used to generate representation for comparison. **→ Well, is this really good representation...?**

<center>
    <img class="img-fluid" src="/assets/post-images/TextureFields/texturefields_repr_power.png">
</center>
<span class="caption text-muted">Figure 9. <b>Experiment on representation power of Texture Fields</b></span>

## Experiment 2. Single Image Texture Reconstruction

In this experiment, the network is given a 3D model with a 2D image of the object from a random camera view. Among many kinds of models introduced throughout the paper, conditional setting is used for training. At test phase, the model undergoes three different settings:

1. Input ground truth 3D shapes along with synthetic renderings of the object
2. Combine the method with 3D shape reconstruction pipeline → Full 3D shape & texture reconstruction **from a single shot of an object**
3. Real world data → Pictures taken from real camera together with similar shapes in ShapeNet dataset

And the results from each of these experiments (for details, refer to the paper):

1. **GT shapes & Synthetic images**: Qualitative results are promising compared to baseslines (projection, NVS). Texture Field reached the top in both FID and Features-$\ell_1$ distance while NVS achieved the best SSIM. This is because of the characteristic of SSIM capturing local information. **However, global feature is more important when trying to make plausible, natural output.**
2. **Full reconstruction pipeline**: Texture Fields outperforms baselines both in quantitative & qualitative analysis.
3. **Real images**: The model generalizes reasonably considering that the model is trained with synthetic data only. **→ So they say..? Definitely need to check this out in action!**

## Experiment 3. Unconditional Model

Check whether the Texture Field can be applied in generative tasks, where the model is given only 3D shape information without anything about apperance. To this end, the conditional VAE and conditional GAN models were trained with models from ShapeNet's 'cars' category.

During training, we supply t**arget images & associated depth maps to the model but input views** (more precisely, image embedding $\textbf{z}$). Instead, $\textbf{z}$ is sampled from random distribution which either standard normal distribution or distribution formed in VAE manner.

It seems the model generates textures for models.. but they're mostly blurry in VAE and contains lots of artifacts in GANs. **→ Still far from there!**

In the case of VAE, additional experiments were done such as:

1. Interpolations in the latent space giving the smooth texture interpolation. **→ VAE learns meaningful latent space**
2. **Texture transfer** from one model to another is quite plausible.

# Conclusion

1. Texture Field can predict high frequency textures from just a single object view.
2. While there are many points to be improved, the method can also be used as generative models for textures.