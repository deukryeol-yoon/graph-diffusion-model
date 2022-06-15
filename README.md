# Graph-Diffusion-Model
Minyoung Choe, Juwon Kim, Soo Yong Lee, and Deukryeol Yoon

Source code for the term project of AI618(Generative Model and Unsupervised Learning) in KAIST.

We implement the graph generation model using a [Denoising Diffusion Implicit Models](https://github.com/ermongroup/ddim) algorithm. 

Our algorithm works by following three steps.
 - Graph Auto Encoder) Learn the graph embedding of an input graph using [Graph Autoencoder](https://github.com/zfjsail/gae-pytorch).
 - Graph Diffusion) Learn the DDIM model.
 - Graph Generation) Generate the graphs using DDIM model and the decoder.