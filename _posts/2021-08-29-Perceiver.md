---
layout: post
title: "Summary of 'Perceiver: General Perception with Iterative Attention'"
subtitle: "Perceiver: General Perception with Iterative Attention (ICML 2021)"
background: '/assets/post-images/Perceiver/fig2.png'
---

# Motivations

- Biological systems perceive the world by simultaneously processing high-dimensional inputs of various forms such as vision, audition, touch, proprioception, etc. On the other hand, the perception models implemented in deep learning often rely on domain specific assumptions and lack multi-modality. → **Architectures must be redesigned from scratch as their input vary.**
- While the inductive biases such as spatial locality in early vision models were valuable since they could increase the efficiency of learning perceptual models by focusing only on data from certain domain. However, considering that there are lots of large scale datasets available, is such decision - introducing constraints to our model - still valid?

# Key Contributions

- Proposes the **Perceiver**, a model which builds upon Transformers, designed to handle arbitrary configurations of different modalities using a single Transformer-based architecture.
- Proposes a novel method which combines the ideas from the Transformer and recurrent neural networks, reducing the computation time and memory usage while maintaining flexibility and performance.

---

# Methods

<center>
    <img class="img-fluid" src="/assets/post-images/Perceiver/fig1.png">
</center>
<span class="caption text-muted">Figure 1. <b>The Perceiver architecture overview</b>.</span>

## The Perceiver architecture

### Overview

The architecture consists of two components:

1. A cross-attention module which maps a byte array (representing input data) and a latent array to a latent array.
2. A Transformer tower that maps a latent array to a latent array (can be thought as self-attention).

Here, the size of the byte array is determined by the input data and is generally large (e.g., images from ImageNet dataset at resolution 224 have 50,176 pixels), while **the size of the latent array is a hyperparameter** which is typically much smaller (e.g., the authors used 512 latents on ImageNet).

The model applies the cross-attention module and the Transformer in an alternative fashion. This is equivalent to repeatedly fusing high-dimensional information of input data into a lower-dimension attention bottleneck. Such low dimensional representation is then used to query the input (byte array) again. This also can be seen as performing a fully end-to-end clustering of the inputs with latent positions as cluster centers.

Since the weights are optionally shared between each instance of the Transformer tower (and between all instances of the cross-attention module but the first), **the model can be interpreted as a recurrent neural network (RNN), but unrolled in depth taking the same input at every step**, not a particular data point in a temporal sequence. Also, all attention modules in the Perceiver are non-causal (i.e., no masks).

### Taming quadratic complexity with cross-attention

<center>
    <img class="img-fluid" src="/assets/post-images/Perceiver/fig2.png">
</center>
<span class="caption text-muted">Figure 2. <b>The authors trained the Perceiver architecture on images from ImageNet (left), video and audio from AudioSet (center), and 3D point clouds from ModelNet40 (right)</b>.</span>

The Perceiver heavily relies on attention mechanisms since it is generally applicable and powerful. Both cross-attention and Transformer modules are based on the query-key-value (QKV) attention. Then, the main challenge the authors encountered while designing such structure was to scale attention architectures to very large and generic inputs which were regarded impossible to process with the traditional Transformers due to the quadratic complexity of QKV self-attention. This is the reason why the Transformer architecture could not be directly applied to other domains.

Prior works usually made compromises in order to avoid applying standard QKV attention directly, which is very expensive, to large scale data such as pixel arrays of images, audio samples, etc. **In contrast, this paper applies attention directly to the inputs by introducing an assymetry into the attention operation.**

Specifically, let $Q \in \mathbb{R}^{M \times D}$, $K \in \mathbb{R}^{M \times C}$, and $V \in \mathbb{R}^{M \times C}$, where $C$ and $D$ are channel dimensions. For example, if self-attention is computed on all pixels of an image, then $M = W \times H$, and $D = 3$ (assuming the image is in RGB format) where $W$ and $H$ are width and height of the image, respectively. Then one can easily observe that the complexity of the QKV attention operation - $\text{softmax} (QK^{T})V$ - is $\mathcal{O}(M^{2})$, as it involves two matrix multiplications with matrices of large dimension $M$. If the image is of size $1024 \times 1024$, the quadratic complexity will immediately explode - $\mathcal{O}(1024^{2}) = \mathcal{O}(2^{20})$. This is the primary reason for introducing assymetry in this work. In this setting, **$K$ and $V$ become projections of the input byte array, while $Q$ is instead a projection of a learned latent array with index dimension** $N \ll M$. As mentioned earlier, the dimensionality $N$ of latents is a hyperparameter. Then the resulting cross-attention operation would have complexity $\mathcal{O}(MN)$.

### Uncoupling depth with a latent Transformer

The output of the cross-attention module has the same shape as the input to the $Q$ network. This is then consumed by deep, expressive Transformers in the latent space which cost only $\mathcal{O}(N^{2})$. This design allows us to implement much deeper Transformers without relying on domain-specific assumptions. It is worth mentioning that a Transformer built on bytes has complexity $\mathcal{O}(LM^{2})$ while a latent Transformer has complexity $\mathcal{O}(LN^{2})$ where $N$ is significantly smaller than $M$, and $L$ is the number of layers.

Then the total complexity becomes $\mathcal{O}(MN + LN^{2})$. Since the Perceiver decouples the input size and the depth, it can embrace additional Transformer layers without any concern on cost related to the size of input data. It is the core idea which enables us to build very large networks on large-scale data (e.g., the best performing Perceiver model on ImageNet has 48 latent Transformer blocks, which was considered impossible with traditional Transformers).

The latent Transformer used in this work adopted the GPT-2 architecture, which is based on the decoder of the original Transformer architecture. In experiments, the authors used values of $N \leq 1024$ for the dimensionality of latents. Furthermore, the latent array is initialized using a learned position encoding.

### Iterative cross-attention & weight sharing

In the previous section, we analyzed how the computational cost is affected by the size of input. Thanks to the smaller latent array and cross-attention between inputs and latents, we can:

1. Use large scale data (e.g., images, audio samples) directly without any optimization or assumption on them.
2. Build deeper Transformers whose cost is solely dependent to the number of latents, not the size of inputs.

On the other hand, **one also must think about the trade-off between the network's ability to capture details from the input data and the computational cost.** Due to the existence of latent bottleneck, the network might struggle to capture necessary information from the input data. And one possible solution for that is to add more cross-attention layers into the model. This will allow the latent array to iteratively extract information from the input data as much as needed. However, while experimental results show that more cross-attends lead to better performance, such gain comes with higher computational cost since the cost of cross-attention has linear dependence on the input size.

Last but not least, the parameter efficiency of the Perceiver can further be improved by exploiting its iterative structure. That is, it is possible to share weights between:

- corresponding blocks of each latent Transformer
- cross-attend modules

Latent self-attention blocks can be shared if only a single cross-attend is used. In their experiments on ImageNet, the authors could reduce the number of parameters by 10 times by sharing weights. Also, this also reduced overfitting and boosted validation performance.

Therefore, the resulting architecture has the functional form of an RNN with:

- A cross-attention input projection
- A bottlenecked latent dimensionality
- A latent Transformer recurrent core

## Position encodings

### Permutation invariance and position information

Attention is a permutation-invariant operation, and this property is still valid in the Perceiver and related models. In other words, a pure attention model will return the same output regardless of the order of its inputs. This is the reason why attention-based architectures are well-suited for many types of data - they make no assumptions about spatial relationships or symmetries. Meanwhile, convolutional layers that are widely adopted in image processing often make several assumptions on 2D spatial structure within images. These assumptions naturally arise from the structure & mechanisms of convolutional layer itself such as:

- Use of filters that look only at local regions of image → makes it easier to capture the relationship between nearby pixels than distant pixels
- Sharing weights across both spatial dimensions (i.e., a single kernel wipes the image along both width and height directions) → helps to model data with translation-invariant statistics
- Applying small filters repeatedly → helps to model data with scale-invariant statistics

However, apart from preventing the Perceiver from making presumptions on data, the model should be able to exploit spatial relationships in its input. Because spatial relationships are essential for sensory reasoning and such limitation is undesirable. To circumvent such issue, position information is usually injected by tagging *position encodings* onto the input features, and this paper use the approach as well. While position encoding was originally introduced in natural language processing literature to encode the position inside word sequences, it can also be used to encode spatial, temporal, and modality identity as well. 

### Scalable Fourier features

<center>
    <img class="img-fluid" src="/assets/post-images/Perceiver/fig3.png">
</center>
<span class="caption text-muted">Figure 3. <b>Attention maps from the first (blue), second (green), and eighth (orange) cross-attention layers of a model on ImageNet with 8 cross-attention modules</b>.</span>

Here, the authors used Fourier feature position encodings which has shown good performance both in language and in vision. They used a parametrization of Fourier features with which so they can:

1. Directly represent the position structure of the input data (preserving 1D temporal or 2D spatial structure for audio or images, respectively, or 3D spatiotemporal structure for videos).
2. Control the number of frequency bands in the position encoding independently of the cutoff frequency.
3. Uniformly sample all frequencies up to a target resolution.

The authors parametrized the frequency encoding to take the values $[\sin (f_{k} \pi x_{d}), \cos (f_{k} \pi x_{d})]$, where the frequency $f_{k}$ is the $k^{\text{th}}$ band of a bank of frequencies spaced equally between 1 and $\frac{\mu}{2}$ and $x_{d} \in [-1, 1]$ is the value of the input position along the $d^{\text{th}}$ dimension. $\frac{\mu}{2}$ can be interpreted as the Nyquist frequency corresponding to a target sampling rate of $\mu$. Providing encodings will encourage the model to learn to compare the values of bytes at any positions in the input array. As other works normally did, these values are then concatenated with the raw position value $x_{d}$ to produce the final representation of position. This gives us a position encoding of size $d(2K + 1)$ (for each dimension, $K$ sine and cosine components and a coordinate along the dimension).

This parametrization scheme is related to the NeRF's position encoding scheme which is built around frequency bands with increasing powers of two (i.e., the $k^{\text{th}}$ band has frequency $2^{k}$). However, this yields very high frequencies for even modest number of bands, thus turned out to be numerically unstable in some experiments conducted by the authors.

Also, it is worth to compare this scheme with that of Transformer. In the Transformer, inputs are produced by adding a position encoding to the input encoding (NB. NLP usually involves encoding of symbols using embedding layer, through the process so called *word embedding*). Note that a position encoding must be of same size as the input encoding when following that scheme. The authors found it beneficial to concatenate the position and input features rather than just adding them together.

### Position encodings are generally applicable

One important thing to note here is that the use of position encodings does NOT contradict the foundation of this work - building a domain agnostic architecture for general perception tasks. There are three reasons:

1. While the architectural imposition of position information hard codes a specific positional prior, **the feature-based approach allows the network to learn how to use the position structure**.
2. **Position encoding can be easily adapted to a new domain since Fourier features are trivial to adapt as long as the input dimensionality is relatively small and known** (NB. lots of signal processing algorithms are built on top of Fourier analysis). The Transformer is one example which shows that simple, learned position encoding is sufficient for good results, and the authors themselves discovered that similar strategy works well on ImageNet (without knowing anything about input 2D structure) and on other kinds of modalities.
3. **Position encodings can be naturally extended to multimodal data** - each domain can use a position encoding with the correct dimensionality for its data, with learned encodings used to distinguish domains.

# Conclusion

This paper presented the Perceiver, a Transformer-based model which scales to more than a hundred thousand inputs. Built upon a novel iterative cross-attention between input byte array and latent array whose size is controlled by a hyperparameter, the Perceiver is able to handle arbitrary sensor configurations, enabling fusion of information at all levels. This makes the Perceiver an architecture for general perception since it makes few assumptions about its inputs.

However, the Perceiver also has drawbacks that are expected to be improved in future works:

1. Its great flexibility comes with great overfitting, thus many parts of its design were intended to mitigate this. → Requires lots of data
2. Although the authors have reduced the amount of modality-specific prior knowledge in the model, they had to employ modality-specific augmentation and position encoding. Thus, searching for a method toward end-to-end modality agnostic learning is still an interesting research topic.