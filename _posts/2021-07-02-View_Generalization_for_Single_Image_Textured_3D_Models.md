---
layout: post
title: "Summary of 'View Generalization for Single Image Textured 3D Models'"
subtitle: "View Generalization for Single Image Textured 3D Models (CVPR 2021)"
background: '/assets/post-images/ViewGeneralizationSingleImage/fig2.png'
---

# Motivation

- Humans can easily infer the underlying 3D geometry and texture of an object only from a single 2D image. **Because we've seen many objects from many views**. It's the experience that enables us to infer a geometry and appearance of a novel object by exploiting the accumulated visual memories from similar category of objects.
- While current computer algorithms can do such task, they often suffer from view generalization problems making awkward images conditioned on novel viewpoints. Either the inferred mesh is distorted or the infferred textures are muddled, blurry, and missing details.

<img class="img-fluid" src="/assets/post-images/ViewGeneralizationSingleImage/fig1.png">
<span class="caption text-muted">Figure 1. <b>Inferring high quality textured 3D models from a single 2D image input</b>.</span>

# Key Contributions

- A controllable convolutional deformation approach for better recovery of 3D geometry of the objects. **→ Specifically, the deformation is not directly done in 3D space, but the geometry is first mapped to 2D UV space, and them deformed there.**
- Two novel cycle consistency losses that improves overall inferred textured 3D models.
- High-fidelity textured 3D model synthesis from a single image both qualitatively and quantitatively.

---

# TL;DR

The network infers both **underlying geometry (mesh) and texture from a single 2D image**. To generate parts of an object (both shape and texture) that were not visible from the input, the network exploits the previously seen images in similar categories but containing objects with different appearance, camera poses, etc. In order to successfully learn such behavior, it's trained under two consistency losses proposed by the paper.

# Method

<img class="img-fluid" src="/assets/post-images/ViewGeneralizationSingleImage/fig1.png">
<span class="caption text-muted">Figure 2. <b>Architecture overview</b>. The framework learns to infer the 3D geometry (represented as a mesh) and the texture of an object given a single image that captures it. The final geometry and obtained by a weighted combination of object templates and refined through deformation.</span>

## Controllable Convolutional Deformation

An encoder-decoder architecture encodes images into 3D geometry and decodes textures, and these are then fed into a differentiable renderer giving reconstructed input images with image-view camera parameters. For differentiable renderer in this pipeline, DIB-R is used.

To estimate 3D geometry with high accuracy, it's necessary to **control the model complexity and mesh deformation**. This is very important since it's often turns out to be difficult to find balance between flexibility for various images and rigidity for realistic rendering of novel views.

For example, a model that is too flexible might work for many kinds of images but each with unsatisfactory quality. On the other hand, a model which is too rigid may reconstruct input images with desirable photorealism, while completely failing to generate results conditioned on unknown viewpoints.

Previous methods used a fully connected linear layer to predict deformation of a template mesh. But this approach had several drawbacks.

1. **Each vertex is independently deformed**, thus the overall geometric structure is not considered.
2. **Fully connected linear layer does not provide controllability** over deformation. While the model should be adjusted so that the deformation is done with flexibility (or rigidity) on certain types of geometry, it's not achievable with fully connected layer. For instance, for non-rigid objects like birds, the model needs to be flexible enough.
3. These **deformations are bound to a specific set of vertices of a template mesh** and difficult to re-sample for any new vertices.

 Thus, instead of approaches suggested in previous works, this paper utilizes a spherical mapping to map a mesh to a sphere. The resulting 2D UV map, thought as an unwrapped sphere having a fixed topology, can be used for deformation later on. ***→ Imagine applying this strategy for deforming shapes having different topology. What would happen?***

With this 2D UV map, **a single spatial deformation UV map, which is a function whose domain is the sphere where the original surface is mapped, is predicted by a convolutional architecture** (this is the reason why it's called convolutional deformation). After prediction, the deformation per vertex of template mesh can be determined by sampling from the predicted deformation UV map using the position of vertex to be deformed.

This method works better than previous deformation methods, especially the most intuitive one - directly deforming the shape in 3D -  and additionally provides capability to adjust the amount of deformation to be applied by adjusting the resolution of the convolutional layer's output.

For large deformation models (non-rigid shapes such as birds), the spatial resolution of deformation map is set to be high ($32 \times 32$), so that the deformation can alter the shape with high flexibility, giving natural shapes. 

On the other hand, for small deformation models (rigid shapes such as cars), the convolutional network predicts low resolution deformation map ($2 \times 2$) or ($4 \times 4$) to restrict the number of vertices (mapped onto the UV map) to be deformed, so that the template mesh does not undergo too much modification (i.e. only small number of vertices in the original model are deformed).

This controllability, refered as degree of deformation (DOD), is one of the key aspect of this work, and used throughout this work. Especially, in the literature, $\text{DOD} = 4$ means that the 2D deformation has $2 \times 2$ spatial resolution.

Since **the deformation is learned for 2D UV map, not the template mesh**, the number of vertices that associated deformations are sampled and then applied can be adjusted freely without extra computational cost.

### Mesh templates

This method refines geometry by starting from a template mesh to be deformed by the convolutional deformation on 2D spherical UV map discussed just before. Unlike previous methods starting with mean templates and learning to deform it, **this method exploits multiple templates provided by PASCAL3D+ dataset, and a mean template for CUB dataset**.

---

**NOTE: This part is a bit ambiguous (to me).**

> *"The network learns w, the template weight, to choose one from $n$ templates it needs to start with for an image. However, we do not have any supervision for these templates, so we pre- dict normalized weights ($w$) of each template ($t$) and take their weighted sum as our predicted mesh template. We also use a learnable scale-factor ($s$) for each of these templates to adapt our templates to appropriate size."*

The following questions emerge from the paragraph above.

1. What's the exact purpose of (normalized) weights? Is this for **picking the most probable template mesh as a starting point**, or **the template mesh is built by mixing the available templates using some inferred weight for each of them?**
2. If it's the second case, **is it possible to define linear combination of two or more meshes?** Is it possible to create a mesh as a weighted sum of base meshes?
3. Is the preprocessing step related to this? What's the condition required to mix two or more different meshes?

For me, the sentence following the above paragraph,

> *"We then refine this predicted mesh by sampling deformation of all vertices ($\Delta V$) using the convolutional UV deformation map. Our final refined mesh vertices are $V = \sum_{i=1}^{n} s_i \times w_i \times V_i^t \,\, + \Delta V$."*

implies that it's possible to add two different meshes. Then, the vertices of template meshes must have correspondance between them, but there's no mention about it.

---

It's shown in the experiments that using only a few additional templates improves textured 3D inference by large margin with only a single 2D image.

And the preprocessing for templates meshes were done as the following:

1. Remove all interior triangles and faces by simplifying mesh into voxels.
2. Convert these voxels into a mesh by running a marching cubes algorithm.
3. Simplify meshes by closing holes and decimating mesh.
4. Use spherical parametrization and map spheres to the final meshes.

**Note that any mesh that fails during each step was discarded and not used.**

***→ (2021. 07. 02. Update) Spheres are mapped to the simplified mesh created from the task 1 ~ 3. Thus, the above discussion (the one that I thought to be ambiguous) makes sense. Also, I think they assumed that the small perturbations applied on the vertices of meshes don't change the relative position between vertices dramatically.***

### Losses

The bases losses are same as DIB-R, which is the work about the differentiable renderer used in this work.

For reconstruction, $\mathcal{L}\_1$ image reconstruction loss between input image $I$ and the rendered image $I_{r}$ and perceptual loss from AlexNet $\Phi$ at different $j$-th feature layers are used. Concretely,

$$ \begin{gather} \mathcal{L}_{\text{recon}} = \vert\vert I - I_r \vert\vert_{1}, \\
\mathcal{L}_{\text{percp}} = \vert\vert \Phi_{j}(I) - \Phi_{j}(I_{r}) \vert\vert_{2} \end{gather} $$

For shapes, an IoU between the silhouette rendered $S_{r}$ and the silhouette $S$ of the input image is used:

$$ \begin{gather} \mathcal{L}\_{\text{sil}} = 1 - \frac{\vert\vert S \odot S_{r} \vert\vert_{1}}{\vert\vert S + S_{r} - S \odot S_{r} \vert\vert} \end{gather} $$

Along with the silhouette loss, the predicted mesh is regularized with a smoothness loss and Laplacian loss ($\mathcal{L}_{\text{lap}}$). The purpose of taking these losses into account is to make normals of neighboring mesh triangles similar.

The camera pose is also predicted. And it's used together with the ground truth camera pose to compute a simple $\mathcal{L}\_2$ regression loss ($\mathcal{L}_{\text{cam}}$) (see the overall architecture).

Keypoints are predictd from the inferred mesh as well. As in the camera pose loss, it's compared with the ground truth and gives us $\mathcal{L}\_{2}$ regression loss ($\mathcal{L}\_{\text{KP}}$).

Finally, the magnitude of deformation is regularized to be small, thus $\mathcal{L}_{\text{deform}}= \vert\vert \Delta V \vert\vert$ is minimized so that it becomes close to zero. Therefore, the baseline loss for this study is:

$$ \begin{gather} \mathcal{L}\_{\text{baseline}} = \lambda_{r} \mathcal{L}\_{\text{recon}} + \lambda_{p} \mathcal{L}\_{\text{percp}} + \lambda_{s} \mathcal{L}\_{\text{sil}} \\ + \lambda_{c} \mathcal{L}\_{\text{cam}} + \lambda_{\text{kp}} \mathcal{L}\_{\text{KP}} + \lambda_{d} \mathcal{L}\_{\text{deform}} + \lambda_{\text{lap}} \mathcal{L}_{\text{lap}} \end{gather} $$

and the values for the coefficients are: $\lambda_{r} = 20$, $\lambda_{p} = 0.5$, $\lambda_{s} = 5.0$, $\lambda_{c} = 1$, $\lambda_{\text{kp}} = 50.0$ , $\lambda_{d} = 2.5$, $\lambda_{\text{lap}} = 5.0$.

Note that there's nothing new in the loss introduced here. This will play the role as a baseline, and the effect of applying loss suggested by the authors will be presented soon.

## Cycle-Consistency Losses

One marvelous thing about human intelligence is that it makes decision based not only on the currently given condition (or circumstance), but also on the experience from the past. For instance, we can infer the overall shape of the mug, even if we only saw the part where the cup handle is. This is because, from the past experience, we already know what the overall shape of normal mug should be. And this is the key idea which should be understood well to achieve high performance in single shot reconstruction task like this.

To this end, this work suggests two cycle-consistency losses to guide the network to preserve both geometric and texture consistency when an object is observed in various viewpoints. By train the network under these losses, both texture and geometric information are *shared* across image collections.

1. **Rotation GAN cycle consistency loss**: Keep rendered images from novel viewpoints as real as possible by implicitly constrain them on a multi-view cycle consistency based on GAN literature. (sharing geometric information across input images)
2. **Texture-mesh alignment cycle consistency loss**: Enforces consistent texture alignment to meshes irrespective of their views and shapes. (sharing texture mapping information across input images, keeping appearance consistency among different objects in the same class)

### Rotation GAN Cycle Consistency

<img class="img-fluid" src="/assets/post-images/ViewGeneralizationSingleImage/fig3.png">
<span class="caption text-muted">Figure 3. <b>Rotation GAN Cycle Consistency</b>.</span>

It's very hard to reconstruct a geometry and the suitable texture for an object given just a single image, since there's no supervision from other viewpoints. Previous methods tried to overcome this problem by imposing a multi-view consistency loss, and rendered multiple views given a single image on synthetic ShapeNet dataset. **→ Not applicable on real world images, since it's difficult to acquire multi-view images.**

That being said, we need to fine another way to achieve multi-view consistency without exploiting a set of images. And the approach proposed by the authors is the following:

Let $M$ be a function which outputs 3D meshes given an input image, $T$ be a texture network, and $R$ be a differentiable renderer which outputs an image.

Given an image $X_1$ and its corresponding camera pose $C_1$, the framework first infer the underlying geometry and texture representation, and then renders a new, intermediate image $I_1$ seen from a novel camera view $C_2$.

$$I_1 = R(M(X_1), T(X_1), C_2)$$

and then **using this intermediate image, the framework then infer the geometry and texture representation, just as it did for $X_1$, and render an image that would've been seen from** $C_1$. More precisely,

$$X_1^{\prime} = R(M(I_1), T(I_1), C_1)$$

Ideally, two images $X_1$ and $X_1^{\prime}$ should be identical, since this indicates that the network is capturing the geometric and texture features well. To optimize the network to behave in that way, the reconstruction loss and perceptual loss are computed with $X_1$ and $X_1^{\prime}$.

In addition, the authors trained a U-Net discriminator to distinguish the real and fake images in both global and pixel-by-pixel level. **This discriminator is then determines whether the intermediate image $I_1$ is fake or not**, making another path for back propagation (through the intermediate image). The GAN loss then penalizes the network if the image from novel views are not realistic, guiding the network to generate realistic intermediate images.

As intermediate images become realistic over training due to the GAN loss, the cycle consistency loss then make use of these intermediate images, leading the network to correctly infer the occluded regions consistently in terms of geometry and texture.

**→ This constraint improves the overall quality of 3D reconstruction.**

### Texture-Mesh Alignment Cycle Consistency

<img class="img-fluid" src="/assets/post-images/ViewGeneralizationSingleImage/fig4.png">
<span class="caption text-muted">Figure 4. <b>Texture Mesh Alignment Cycle Consistency</b>.</span>

On the other hand, synthesizing accurate texture for an object in an image was the another objective of this work. "How can we do that?"

One important fact to notice is that the network can be easily overfitted due to the problem setting of single view reconstruction. The only information that the naive-neural network can utilize for synthesizing the texutre is the given single image. The artifact of overfitting is inconsistent, awkward, and distorted texture when seen from different viewpoints. Therefore, **the key to solve this issue is to find a way to avoid overfitting the network to a specific image**.

Inspired by the insight gained from human perception - depending on not only the information currently available, but refer to the past experience - one might come up with an idea to share the appearance information over the input images. And hopefully, the network can learn the underlying relation between shape and texture (e.g. inferring the location of headlights given an image showing the back of a car using *other images* depicting the front of various cars).

And this can be mimicked by constrain the network on the novel texture-mesh alignment cycle consistency loss. Suppose that we're given two images $X_1$, $X_2$ each depicting different objects (e.g. cars in figure above), along with their corresponding camera poses $C_1$, $C_2$. Then the framework first makes inference on their 3D meshes and textures. Then, we **swap textures** to reconstruct two intermediate images such that:

$$ \begin{gather} I_1 = R(M(X_1), T(X_2), C_1), \\
I_2 = R(M(X_2), T(X_1) , C_2) \end{gather} $$

then the next step is to infer textures from these intermediate images and swap textures back onto their original geometries, reconstructing the original images:

$$ \begin{gather} X_1^{\prime} = R(M(X_1), T(I_2), C_1), \\
X_2^{\prime} = R(M(X_2), T(I_1), C_2) \end{gather} $$

Again, similar to rotation GAN consistency loss, the original image $X_i$ and $X_i^{\prime}$ should match ideally. By regularizing the network using the texture-mesh alignment loss, we can reduce the bias arose by taking only a single image into consideration during inference. Furthermore, another useful aspect of this loss is that the network can learn how the parts of a texture that were originally invisible or occluded should look like from the novel viewpoints. This definitely helps the network to recover accurate & realistic textures.

# Experiments

## Datasets

- **PASCAL3D+**: Primarily used dataset → especially 'car' category for evaluations
- **CUBs**: 'Bird' shape models were used for evaluation as well
- The same train-test split from DIB-R.
- **7 templates** for deforming meshes, these templates were brought from PASCAL3D+ dataset and preprocessed following the steps introduced in the method section.
- But for birds, the framework starts from only a single template.

## Baselines & Ablations

The model was compared mostly with **the previous SOTA, DIB-R**. The pretrained model of DIB-R was used for comparison. While DIB-R does not learn to infer camera poses, this work does. Also, DIB-R doesn't utilize symmetry in both geometry and texture since this rather reduces the quality of inferenced 3D mesh and texture. 

And there are three types of ablation studies:

1. Using **single template vs multiple templates**
2. Effectiveness of **convolutional deformation** over linear layer based deformation
3. Role of **novel cycle consistency losses**

## Model Details

Given an image, weights ($w$) for each template mesh and camera pose prediction is obtained by applying a single linear layer over ResNet18 features. Then the starting point mesh is created by simply compute a weighted average per template vertex, over the set of meshes.

For predicting texture maps and deformation (in the form of 2D UV map), U-Net architecture is used. Texture maps are predicted to have $256 \times 256$ resolution. For predicting deformations, UV maps having $DOD = 4$ or $2 \times 2$ spatial resolution are used for cars, while the ones with $DOD=256$ or $16 \times 16$ spatial resolution are used for birds. 

The deformation is sampled using bilinear grid interpolation, by evaluating the interpolated deformation at the coordinate in 2D spherical UV map, that were mapped from a vertex of a mesh. This method reduces the computational cost by removing the necessity of computing deformation for each of the mesh vertices.

Two types of variants are tested for GAN architecture in the framework. While both of them treat the an image rendered from a novel view as fake, one used *input training images* as real while the other used *the rendered images from original viewpoint* as real. Notably, shapes were improved when the input training images are used as real, while appearances got better when the rendered images are used as real.

## Evaluation

Since there's no ground truth 3D geometry and texture maps to evaluate, this study applied the same evaluation principles used in DIB-R. For geometry, **mask and silhouette projection accuracy** are used while **FID scores** are computed from the rendered images as a metric for realistic appearance reconstruction. Besides, **user study** was done to evaluate human preferances comparing this method with previous methods.

## Results

<img class="img-fluid" src="/assets/post-images/ViewGeneralizationSingleImage/fig5.png">
<span class="caption text-muted">Figure 5. <b>Textured 3D inference on PASCAL3D+ cars</b>.</span>

The qualitative comparison between DIB-R, the previous SOTA, is shown above. While DIB-R generates good quality reconstructions from the original viewpoint, it suffers from distortion when conditioned on novel viewpoints compared to this method. The quantitative comparison in the table below represents this well. While DIB-R achieves the highest 2D mIoU conditioned on known viewpoints, still FID measure is quite high (higher, the worse) meaning that the reconstructed images are a bit unrealistic. Meanwhile, along with ablations, the newly suggested pipeline achieves better performance both in terms of mIoU and FID.

<img class="img-fluid" src="/assets/post-images/ViewGeneralizationSingleImage/fig6.png">
<span class="caption text-muted">Figure 6. <b>Quantitative results on PASCAL3D+ car dataset</b>.</span>

The similar discussion can be done on another category of shape - birds. Note that even for non-rigid geometry and appearance (organic structure such as a body of an organism) this method works quite well.

<img class="img-fluid" src="/assets/post-images/ViewGeneralizationSingleImage/fig7.png">
<span class="caption text-muted">Figure 7. <b>Textured 3D inference on the CUB dataset</b>.</span>


<img class="img-fluid" src="/assets/post-images/ViewGeneralizationSingleImage/fig8.png">
<span class="caption text-muted">Figure 8. <b>Quantitative results on CUB dataset</b>.</span>

Another aspect we need to look carefully is that the effect of resolution in convolutional deformation. In both qualitative and quantitative evaluation, one can clearly see that it's necessary to differentiate the resolution of deformation according to the target shape.

<img class="img-fluid" src="/assets/post-images/ViewGeneralizationSingleImage/fig9.png">
<span class="caption text-muted">Figure 9. <b>DOD Ablation study (Quantitative)</b>. Note that increasing DOD for rigid shapes (like cars) leads to worse performance while the reverse happens for flexible shapes (like birds).</span>

<img class="img-fluid" src="/assets/post-images/ViewGeneralizationSingleImage/fig10.png">
<span class="caption text-muted">Figure 10. <b>DOD Ablation study (Qualitative)</b></span>

# Conclusion

- Introduced a new 3D inference pipeline which reconstructs 3D mesh and its texture at the same time, from a single image.
- Convolutional deformation allow us to control the degree of deformation (DOD) by adjusting the resolution of the convolution feature map. And it's computationally efficient since the deformation for each vertex is evaluated via sampling and interpolation from this deformation map.
- Two cycle consistency losses - rotation GAN, texture-mesh alignment - each related to view-dependent geometry and appearance consistency.
- Thanks to these novel components, this method achieved the SOTA performance, making improvements from the previous one.