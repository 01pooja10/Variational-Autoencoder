# Implementation of Variational Autoencoders
This repository will demonstrate how to code and run inference using a variational autoencoder neural network from scratch using the PyTorch deep learning framework.
The code is available in the source (src) folder.
### What is a VAE?
A variational autoencoder uses the encoder-decoder architecture along with variational inference to generate data points. They use probability distributions for mapping inputs in the latent space hence providing more flexibility and better generation capabilities.

![architecture](https://lilianweng.github.io/posts/2018-08-12-vae/autoencoder-architecture.png)

We obtain mean and variance values for all the data points after passing our inputs through the encoder network. these points are compressed to a latent representation in the bottleneck before the decoder can begin the generation process. The decoder samples new data points from the probability distribution in order to reconstruct images accurately.
