---
layout: post
title: "Summary of 'DatasetGAN: Efficient Labeled Data Factory with Minimal Human Effort'"
subtitle: "DatasetGAN: Efficient Labeled Data Factory with Minimal Human Effort (CVPR 2021)"
background: '/assets/post-images/DatasetGAN/fig1.png'
---

# Motivations

<img class="img-fluid" src="/assets/post-images/DatasetGAN/fig1.png">
<span class="caption text-muted">Figure 1. <b>DatasetGAN overview</b>.</span>

- Modern deep neural networks are extremely data-hungry, benefiting from training on large-scale datasets, which are time consuming to annotate.
- Recently, semi-supervised learning has been a popular approach in the tasks of reducing the need for labeled data, by instead using a large unlabeled dataset.
- Several studies on the latent space of GANs (particularly StyleGAN) witnessed that GANs acquire semantic knowledge in their high dimensional latent space, and one can introduce various semantic changes by exploring such latent space.

# Key Contributions

- Introduces DatasetGAN, an automatic procedure to generate massive datsets of high-quality semantically segmented images requiring minimal human effort.

---

# TL;DR

This work proposes DatasetGAN, a novel way to utilize StyleGAN as a generator for massive scale datasets. With only little human supervision, the trained Style Interpreter architecture can synthesize infinite number of semantic (or keypoint) labels while running in parallel to StyleGAN backbone which generates images corresponding to labels from the Style Interpreter.

# Methods

<img class="img-fluid" src="/assets/post-images/DatasetGAN/fig2.png">
<span class="caption text-muted">Figure 2. <b>Overall architecture of DatasetGAN</b>.</span>

This paper introduces DatasetGAN that synthesizes image-annotation pairs. The authors focus on pixel-wise annotation tasks for semantic segmentation and keypoint prediction because these are typical problems that require manually annotated dataset where such annotation is extremely labor-intensive.

**The key insight of DatasetGAN is that generative models (e.g. GANs) trained to synthesize highly realistic images must acquire semantic knowledge in their high dimensional latent space.** DatasetGAN is designed to utilize such powerful properties of image GANs. In particular, the authors trained a very simple MLP which maps the feature vector of each pixel to the semantic label on a small, human-labeled segmentation dataset. And they expected this information, acquired by label supervision, to be effectively propagated across the GAN's latent space.

<img class="img-fluid" src="/assets/post-images/DatasetGAN/fig3.png">
<span class="caption text-muted">Figure 3. <b>Small, detailed human-annotated face and car datasets</b>.</span>

Inspite of its simple structure, the proposed architecture is turned out to be powerful. Specifically, the authors **first synthesized a small number of images by utilizing a GAN architecture**, StyleGAN in this case, and **recorded their corresponding latent feature maps**. On top of that, a human annotator labeled these images with a desired set of labels. Using this latent-annotated image pairs, the authors trained **a simple ensemble of MLP classifiers** that take the StyleGAN's pixel-wise feature vectors, which is referred to as the *Style Interpreter*. One huge advantage this method takes compared to traditional, annotate-by-hand approaches is that this technique requires only a few annotated examples to achieve good accuracy.

After training the Style Interpreter, the authors used it as a subnetwork which runs in parallel with StyleGAN and outputs semantically segmented images that are ready to be used for training any related computer vision architectures.

## Prerequisites

DatasetGAN uses StyleGAN as the generative backbone because it is capable of synthesizing high quality images. The StyleGAN's synthesis network maps a latent code $\textbf{z} \in \mathcal{Z}$ drawn from a normal distribution to a realsitic image. Latent code $z$ is first mapped to an intermediate latent code $\text{w} \in \mathcal{W}$ by a mapping function. $\textbf{w}$ is then transformed to $k$ vectors, $\textbf{w}^{1}, \dots, \textbf{w}^{k}$, through $k$ learned affine transformations. 

These $k$ transformed latent codes are injected as style information into $k / 2$ synthesis blocks in a progressive fashion (i.e. an image is gradually constructed by fusing style codes each accounting for different aspect into it). Specifically, each synthesis block consists of an upsampling layer and two convolutional layers. Each convolutional layer is followed by an adaptive instance normalization (AdaIN) layer controlled by its corresponding $\textbf{w}^{i}$. In this paper, the output feature maps from the $k$ AdaIN layers are denoted as $\\{ S^{0}, S^{1}, \dots, S^{k}\\}$.

## Style Interpreter

In this work, the authors interpreted StyleGAN as a "rendering" engine which takes latent codes representing "graphics attributes" that define what to render.

One important hypothesis which deserves our attention is that:

> A flattened array of features that output a particular RGB pixel contains semantically meaningful information for rendering the pixel realistically.

Using this as the foundation, the authors upsampled all feature maps $\\{ S^{0}, S^{1}, \dots, S^{k} \\}$ from AdaIN layers to the highest output resolution (i.e. the resolution of $S^{k}$) and concatenated them to get a 3D feature tensor $S^{\*} = (S^{0, \*}, S^{1, \*}, \dots, S^{k, \*})$ where $S^{k, \*}$ stands for the upsampled $k$-th feature map and $(\cdot, \cdot)$ denotes concatenation. In this context, each pixel $i$ in the output image has its associated feature  vector $S_{i}^{*} = (S_{i}^{0, *}, S_{i}^{1, *}, \dots, S_{i}^{k, *})$, as shown in figure 2. The authors adopted three-layer MLP classifier on top of each feature vector to predict labels. The weights are shared across all pixels for simplicity.

### Training (of Style Interpreter)

<img class="img-fluid" src="/assets/post-images/DatasetGAN/fig4.png">
<span class="caption text-muted">Figure 4. <b>Examples of synthesized images and labels from DatasetGAN for faces and cars</b>.</span>

Suppose we have already synthesized few images and annotated them manually. Then the strategy used for training Style Interpreter is as follows:

Since per-pixel feature vectors $S_{i}^{*}$ are of high dimensionality (5056) and the feature map has  high spatial resolution (1024 at most), one cannot consume all image feature vectors in a batch due to the shortage of memory space. The authors came up with the idea of performing random sampling of feature vectors from each image, while ensuring that at least one sample is drawn from each labeled region.

Furthermore, different losses are used for different tasks. For semantic segmentation, the classifier is trained with cross-entropy loss. For keypoint prediction, the authors first built a Gaussian heatmap for each keypoint in the training set and let the MLP to fit the heat value for each pixel. **One important point is that the weights of StyleGAN is fixed during the training procedure (i.e. gradients are not back propagated to the StyleGAN).**

The reason for adopting an ensemble of classifier was to amortize the effect of random sampling. In this paper, total $N = 10$ classifiers were trained. The authors used majority voting in each pixel at test time for semantic segmentation. And they averaged the $N$ heat values predicted by each of the $N$ classifiers in the case of keypoint prediction.

## DatasetGAN as a Labeled Data Factory

<img class="img-fluid" src="/assets/post-images/DatasetGAN/fig5.png">
<span class="caption text-muted">Figure 5. <b>Examples of synthesized images and labels from DatasetGAN for birds, cats, and bedrooms</b>.</span>

Once trained, the Style Interpreter can be used as a label-synthesis branch running in parallel to the StyleGAN backbone, forming the entire DatasetGAN architecture. Using the same latent code $\textbf{z} \in \mathcal{Z}$ sampled from StyleGAN's latent space, we can synthesize both photorealistic image and corresponding semantic label simultaneously. In practice, synthesizing an image-annotation pair requires a forward pass through StyleGAN, which takes 9 seconds on average.

However, StyleGAN occasionally fails introducing noise in the synthesized dataset. The authors noticed that the StyleGAN's discriminator score is not a robust measure of failure and also found that utilizing their ensemble of classifiers to measure the uncertainty of a synthesized example is a more robust approach. For measurement of uncertainty, this work adopted the Jensen-Shannon (JS) divergence for a pixel. In the case of image uncertainty, they summed over all image pixels. According to the computed uncertainty, top 10% most uncertain images were filtered out.

# Conclusion

<img class="img-fluid" src="/assets/post-images/DatasetGAN/fig6.png">
<span class="caption text-muted">Figure 6. <b>3D Application</b>.</span>

The authors proposed a simple but powerful approach for semi-supervised learning with few labels. They explored the learned latent space of the state-of-the-art generative model StyleGAN, and developed an effective classifier which can be trained on only a few human-annotated images. The DatasetGAN, consists of StyleGAN backbone and a novel Style Interpeter architecture, is able to synthesize large labeled datasets which then can be used for training various computer vision architectures.