---
layout: post
title: "Summary of 'Attention Is All You Need'"
subtitle: "Attention Is All You Need (NIPS 2017)"
background: '/assets/post-images/Transformer/transformer.jpg'
---

# Motivations

- While recurrent neural networks (RNN), long short-term memory (LSTM), and gated recurrent (GRU) neural networks have been established as state-of-the-art approaches in sequence modeling and transduction problems (e.g., language modeling, machine translation), their inherent sequential nature precludes parallelization within training examples, slowing down both training and inference. Furthermore, such models perform poorly on long sequences where the memory constraints become a bottleneck.
- On the other hand, there has been tremendous attempts adopting attention mechanisms to the existing recurrent models, allowing modeling of dependencies without regard to their distance in the input or output sequences. However, the limitation originated from the nature of recurrent models prevents us from seeing the full potential of attention mechanisms.

# Key Contributions

- Proposes the **Transformer**, a model architecture without any recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output.

---

# Methods

## Model Architecture

<img class="img-fluid" src="/assets/post-images/Transformer/fig1.png">
<span class="caption text-muted">Figure 1. <b>The architecture of the Transformer</b>.</span>

Most of neural sequence transduction models before the Transformer had an encoder-decoder structure. 

In such models, the encoder first maps an input sequence of symbol representations $(x_{1}, \dots, x_{n})$ to a sequence of continuous representations $\textbf{z} = (z_{1}, \dots, z_{n})$. Only after that, the decoder given $\textbf{z}$ generates an output sequence $(y_{1}, \dots, y_{n})$ of symbols one element at a time. Such models are said to be auto-regressive since they consume the previously generated symbols as additional input for the next prediction.

The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder.

### Encoder and Decoder Stacks

**< Encoder >**

The encoder of the Transformer is composed of a stack of $N=6$ identical layers, and each layer is composed of two sub-layers. The former is a multi-head self-attention mechanism, and the latter is a simple, position-wise fully connected feed-forward network. The authors adopted a residual connection around each of the two sub-layers, followed by layer normalization. In other words, the output of each sub-layer can be formally expressed as:

$$\text{LayerNorm} \big(x + \text{Sublayer}(x) \big),$$

where $\text{Sublayer}(x)$ is the function implemented by the sub-layer. Note that all sub-layers in the model, as well as embedding layers, produce outputs of dimension $d_{\text{model}} = 512$.

**< Decoder >**

The decoder is also composed of a stack of $N = 6$ identical layers. **However, the structure of decoder is different from that of the encoder in the sense that there is a third sub-layer in each layer. This layer performs multi-head attention over the output of the encoder stack.** As in the encoder, the authors employed residual connections around each of the sub-layers, followed by layer normalization and addition. **Another important difference to focus on is that the self-attention sub-layer in the decoder stack is modified to prevent positions from attending to subsequent positions** (i.e., the decoder must not know about the rest of a sequence before predicting a symbol at a specific step). This modification, combined with the output embeddings shifted by one position, ensures that the predictions for position $i$ is only dependent to the known outputs at positions less than $i$.

## Attention

An attention function can be described as the following:

> An attention function $\text{Attention}(\textbf{Q}, \textbf{K}, \textbf{V})$ is a mapping which maps a query $\textbf{Q}$ and a set of key-value $(\textbf{K}, \textbf{V})$  pairs to an output where the $\textbf{Q}$, $\textbf{K}$, $\textbf{V}$, and outputs are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function (i.e., how two quantities are related) of the query with the corresponding key.

<img class="img-fluid" src="/assets/post-images/Transformer/fig2.png">
<span class="caption text-muted">Figure 2. <b>(left) Scaled Dot-Product Attention. (right) Multi-Head Attention</b>. Note that multiple attentions are computed in parallel in the multi-head attention structure.</span>

**< Scaled Dot-Product Attention >**

This paper proposes a new attention mechanism called "Scaled Dot-Product Attention", where the input consists of queries and keys of dimension $d_{k}$, and values of dimension $d_{v}$. Using this method, we first compute the dot products of the query with all keys, and then divide each by $\sqrt{d_{k}}$, and apply a softmax function to obtain the weights on the values.

In the actual implementation, the attention function can be computed on a set of queries simultaneously, by packing a queries into a matrix $\textbf{Q}$. Similarly, the keys and values can also be packed together into matrices $\textbf{K}$ and $\textbf{V}$, respectively. More precisely, the scaled dot-product attention is defined as:

$$\text{Attention}(\textbf{Q}, \textbf{K}, \textbf{V}) = \text{softmax} \big( \frac{\textbf{Q}\textbf{K}^{T}}{\sqrt{d_{k}}}\big) \textbf{V}$$

Unlike conventional dot-product attention, the authors decided to scale the output of the matrix multiplication since the dot product grows large in magnitude as $d_{k}$ gets larger, pushing the softmax function into regions where it has extremely small gradients. Thus, the dot products are scaled by $1 / \sqrt{d_{k}}$ to counteract this effect.

**< Multi-Head Attention >**

The authors found it beneficial to linear project the queries, keys, and values $h$ times with **different**, **learned** linear projections to $d_{k}$, $d_{k}$ and $d_{v}$ dimensions, respectively. The attention function is computed on each different version of queries, keys and values while taking advantage of massive parallelism thanks to the capability of modern GPUs. The result of each computation is $d_{v}$ dimensional output values. These are concatenated and once again projected, resulting in the final values. Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. That is, attentions on different embeddings focus differently on compatibility between elements in the input & output sequences. Mathematically, the multi-head attention is defined as:

$$ 
\begin{gather}
\text{MultiHead}(\textbf{Q}, \textbf{K}, \textbf{V}) = \text{Concat}(\text{head}_{1}, \dots, \text{head}_{h}) W^{O} \\
\text{where } \text{head}_{i} = \text{Attention}(\textbf{Q} \textbf{W}_{i}^{\text{Q}}, \textbf{K} \textbf{W}_{i}^{\textbf{K}}, \textbf{V} \textbf{W}_{i}^{\textbf{V}}),
\end{gather}
$$

where $\textbf{W}\_{i}^{\textbf{Q}} \in \mathbb{R}^{d\_{\text{model}} \times d\_{k}}$, $\textbf{W}\_{i}^{K} \in \mathbb{R}^{d\_{\text{model}} \times d\_{k}}$, $\textbf{W}\_{i}^{\textbf{V}} \in \mathbb{R}^{d\_{\text{model}} \times d\_{v}}$, $\textbf{W}^{O} \in \mathbb{R}^{hd\_{v} \times d_{\text{model}}}$ are matrices representing trainable parameters. In this work, the authors used $h = 8$ parallel attention layers, or heads. For each of these they used $d_{k} = d_{v} = d_{\text{model}} / h = 64$. Note that the total computational cost is similar to that of single-head attention with full dimensionality due to the reduced dimension of each head.

**< Applications of Attention in the Model >**

The Transformer uses multi-head attention in three different ways:

- In "encoder-decoder attention" layer only in the attention "blocks" of the decoder, the queries are from the previous decoder layer, and the memory keys and values come from the output of the encoder. **This structure allows every position in the decoder to attend over all positions in the input sequence.** This behavior is similar to the typical encoder-decoder attention mechanisms in Seq2Seq models.
- The encoder itself has self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. **Each position in the encoder can attend to all positions in the previous layer of the encoder (and that's why it is called *"self"*-attention).**
- Similar to the encoder, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder **up to and including that position**. It would not make any sense if a word at some timestep in the *predicted* sentence attends to other words that are not even predicted yet (i.e., unknown to the model). That being said, the Transformer has to prevent leftward information flow in the decoder to preserve the auto-regressive property. In practice, this can be done inside of scaled dot-product attention layer by masking out all values (i.e., set to $\infty$), that are not allowed to be observed by the decoder itself, in the input of the softmax.

### Position-wise Feed-Forward Networks

Each of the layers in the encoder and decoder also contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between:

$$\text{FFN}(x) = \max (0, x \textbf{W}\_{1} + b_{1}) \textbf{W}\_{2} + b\_{2},$$

where the matrices $\textbf{W}\_{1}$ and $\textbf{W}\_{2}$ represent the trainable parameters. Note that while the different positions share the same linear transformation, the parameters of transformations differ from layer to layer. This can be implemented by two convolutions with kernel size 1. The dimensionality of input and output is $d_{\text{model}} =512$, and the inner-layer has dimensionality $d_{ff} = 2048$.

### Embeddings and Softmax

Just like other sequence transduction models, this work used learned embeddings to convert the input tokens and output tokens to vectors of dimension $d_{\text{model}}$. The authors also used the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities. In the Transformer, the same weight matrix is shared among the two embedding layers and the pre-softmax linear transformation. Moreover, the weights of the embedding layers are scaled by multiplying $\sqrt{d_{\text{model}}}$.

### Positional Encoding

Here is one important fact about the Transformer:

> The Transformer contains no recurrence and no convolution.

The reason why recurrent neural networks had been widely adopted in various problems involving sequences was clear - their sequential structure fits well, and naturally models sequential data. **Then, without recurrence and convolution, how can we teach the Transformer to be aware of the order of tokens in the given sequences?** To do so, one must provide the Transformer some information about the relative or absolute position of the tokens in the sequence. To this end, this paper proposes to add "positional encodings" to the input embeddings at the bottom of the encoder and decoder stacks. The positional encodings have the same dimensionality $d_{\text{model}}$ as the embeddings, so that the two can be summed. While there are many choices of positional encodings available, the authors used sine and cosine functions of different frequencies:

$$
\begin{gather}
\text{PE}_{(pos, 2i)} = \sin (pos / 10000^{2i / d_{\text{model}}}) \\
\text{PE}_{(pos, 2i + 1)} = \cos (pos / 10000^{2i / d_{\text{model}}}),
\end{gather} $$

where $pos$ is the position and $i$ is the dimension. In other words, each dimension of the positional encoding corresponds to a sinusoid. Note that the wavelengths form a geometric progression from $2\pi$ to $10000 \cdot 2\pi$.

## Why Self-Attention

<img class="img-fluid" src="/assets/post-images/Transformer/table1.png">
<span class="caption text-muted">Figure 2. <b>Maximum path lengths, per-layer complexity and minimum number of sequential operations for different layer types</b>.</span>

The authors also provide thorough comparisons between the proposed self-attention layers and the recurrent and convolutional layers that were commonly used for mapping one variable-length sequence of symbol representations $(x_{1}, \dots, x_{n})$ to another sequence of equal length $(z_{1}, \dots, z_{n})$, with $x_{i}, z_{i} \in \mathbb{R}^{d}$ (e.g., a hidden layer in a typical sequence transduction encoder or decoder). The authors focus on three aspects:

1. The total computational complexity per layer.
2. The amount of computation that can be parallelized, measured by the minimum number of sequential operations required.
3. The path length between long-range dependencies in the network. This is important since learning long-range dependencies is a key challenge in many sequence transduction tasks. And one key factor affecting the ability to learn such dependencies is the length of the paths forward and backward signals have to traverse in the network. The shorter, the better.

The table 1 presents side-by-side comparison on each case. In terms of the maximum path length, a self-attention layer is able to connect all positions with a constant number of sequentially executed operations, while a recurrent layer requires $O(n)$ sequential operations.

<img class="img-fluid" src="/assets/post-images/Transformer/fig3.png">
<span class="caption text-muted">Figure 3. <b>An example of the attention mechanism following long-distance dependencies in one self-attention layer of the encoder</b>.</span>

When it comes to computational complexity, self-attention layers are much faster than recurrent layers when $n << d$, which is common for sequences and their representations handled in state-of-the-art models in machine translations (e.g., word-piece, byte-pair, etc). For very long sequences, one may come up with a restricted version of self-attention which considers only a neighborhood of size $r$ in the input sequence centered around the respective output position.

Finally, a convolutional layer with kernel size $k < n$ cannot connect all pairs of input and output positions. Otherwise, it requires a stack of $O(n / k)$ convolutional layers in the case of contiguous kernels, or $O \big( \log_{k} (n) \big)$ in the case of dilated convolutions. At the same time, convolutional layers are more expensive than recurrent layers, by a factor of $k$.

<img class="img-fluid" src="/assets/post-images/Transformer/fig4.png">
<span class="caption text-muted">Figure 4. <b>Many of the attention heads exhibit behaviour that seems related to the structure of the sentence</b>.</span>

Additionally, self-attention can be used to analyze the internal of models. The output of self-attention layers are in fact the data so called *attention distributions*. Experimental results tell us that different attention heads learn to perform different tasks, and at the same time, many of them seem to be related to the syntactic and semantic structure of the sentences.

# Conclusion

This paper presents the Transformer, the first sequence transduction model based entirely on attention. Compared with most commonly used encoder-decoder architectures, the Transformer benefits from multi-headed self-attention layers in both encoder and decoder in terms of performance and efficiency.