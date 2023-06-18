# Implementation of Variational Autoencoders
This repository will demonstrate how to code and run inference using a variational autoencoder neural network from scratch using the PyTorch deep learning framework.
The code is available in the source (src) folder.
### What is a VAE?
A variational autoencoder uses the encoder-decoder architecture along with variational inference to generate data points. They use probability distributions for mapping inputs in the latent space hence providing more flexibility and better generation capabilities.

![architecture](https://lilianweng.github.io/posts/2018-08-12-vae/autoencoder-architecture.png)

We obtain mean and variance values for all the data points after passing our inputs through the encoder network. these points are compressed to a latent representation in the bottleneck before the decoder can begin the generation process. The decoder samples new data points from the probability distribution in order to reconstruct images accurately.

## Instructions to run

To implement this repository, follow the instructions given below:
```
$ git clone https://github.com/01pooja10/Variational-Autoencoder
$ cd src 
$ python model.py 
$ python train.py
```

This will allow you to train your model from scratch. Don't forget to experiment with the hyperparameters for achieving better results!
Further, you can generate your own version of images that resemble the MNIST dataset by running the following line
``` $ python inference.py ```

Thanks for visiting my repository. Hope you found it informative and useful.

## Contributor

<td width:25%>

Pooja Ravi

<p align="center">
<img src = "https://avatars3.githubusercontent.com/u/66198904?s=460&u=06bd3edde2858507e8c42569d76d61b3491243ad&v=4"  height="120" alt="Pooja Ravi">
</p>
<p align="center">
<a href = "https://github.com/01pooja10"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/pooja-ravi-9b88861b2/">
<img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/>
</a>
</p>
</td>

## License
MIT © Pooja Ravi

This project is licensed under the MIT License - see the [License](LICENSE) file for details

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)
