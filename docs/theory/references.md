[Theory index](README.md) · previous: [Convolution and attention](05-convolution-and-attention.md)

# References

Only papers whose technique exists in the code. Each entry says where it is used.

| Reference | Technique | Where in the code | Chapter |
| --- | --- | --- | --- |
| D. E. Rumelhart, G. E. Hinton, R. J. Williams. *Learning representations by back-propagating errors*. Nature 323, 1986. | Backpropagation | `Layer::backward` of every layer, `Network::fit` (`src/my_torch/Network.cpp`) | [2](02-backpropagation.md) |
| Y. LeCun, L. Bottou, G. B. Orr, K.-R. Müller. *Efficient BackProp*. In *Neural Networks: Tricks of the Trade*, Springer, 1998. | LeCun initialisation, variance $1/n_{\text{in}}$ | `init=lecun_uniform`, `lecun_normal` (`src/my_torch/Dense.cpp`) | [4](04-optimization.md) |
| Y. LeCun, L. Bottou, Y. Bengio, P. Haffner. *Gradient-based learning applied to document recognition*. Proc. IEEE 86(11), 1998. | Convolutional networks with pooling | `Conv2D`, `MaxPool2D`, `AvgPool2D` (`src/my_torch/Convolution.cpp`) | [5](05-convolution-and-attention.md) |
| X. Glorot, Y. Bengio. *Understanding the difficulty of training deep feedforward neural networks*. AISTATS 2010. | Xavier initialisation | `init=xavier_uniform`, `xavier_normal`; the projections of `SelfAttention` | [4](04-optimization.md) |
| K. He, X. Zhang, S. Ren, J. Sun. *Delving deep into rectifiers*. ICCV 2015. | He initialisation | `init=he_uniform`, `he_normal`, default of `conv2d` and `token_dense` | [4](04-optimization.md) |
| K. He, X. Zhang, S. Ren, J. Sun. *Deep residual learning for image recognition*. CVPR 2016. | Residual blocks with identity shortcut | `Residual` (`src/my_torch/Containers.cpp`) | [5](05-convolution-and-attention.md) |
| A. M. Saxe, J. L. McClelland, S. Ganguli. *Exact solutions to the nonlinear dynamics of learning in deep linear neural networks*. ICLR 2014. | Orthogonal initialisation | `init=orthogonal` (`fillOrthogonal`, `src/my_torch/Dense.cpp`) | [4](04-optimization.md) |
| A. L. Maas, A. Y. Hannun, A. Y. Ng. *Rectifier nonlinearities improve neural network acoustic models*. ICML Workshop, 2013. | Leaky ReLU | `LeakyReLU` (`src/my_torch/Activations.cpp`) | [3](03-activations-and-losses.md) |
| P. J. Huber. *Robust estimation of a location parameter*. Ann. Math. Statist. 35(1), 1964. | Huber loss | `Huber` (`src/my_torch/Losses.cpp`) | [3](03-activations-and-losses.md) |
| C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, Z. Wojna. *Rethinking the Inception architecture for computer vision*. CVPR 2016. | Label smoothing | `cross_entropy smoothing=` (`CrossEntropy`, `src/my_torch/Losses.cpp`) | [3](03-activations-and-losses.md) |
| Y. Nesterov. *A method for solving the convex programming problem with convergence rate O(1/k²)*. Soviet Math. Dokl. 27, 1983; as used in I. Sutskever, J. Martens, G. Dahl, G. Hinton, *On the importance of initialization and momentum in deep learning*, ICML 2013. | Momentum and Nesterov momentum | `sgd momentum= nesterov=` (`src/my_torch/Optimizers.cpp`) | [4](04-optimization.md) |
| J. Duchi, E. Hazan, Y. Singer. *Adaptive subgradient methods for online learning and stochastic optimization*. JMLR 12, 2011. | Adagrad | `Adagrad` | [4](04-optimization.md) |
| T. Tieleman, G. Hinton. *Lecture 6.5, RMSProp*. COURSERA: Neural Networks for Machine Learning, 2012. | RMSProp | `RMSProp` | [4](04-optimization.md) |
| D. P. Kingma, J. Ba. *Adam: A method for stochastic optimization*. ICLR 2015. | Adam with bias correction | `Adam` | [4](04-optimization.md) |
| S. J. Reddi, S. Kale, S. Kumar. *On the convergence of Adam and beyond*. ICLR 2018. | AMSGrad | `adam amsgrad=1` | [4](04-optimization.md) |
| I. Loshchilov, F. Hutter. *Decoupled weight decay regularization*. ICLR 2019. | AdamW | `AdamW` | [4](04-optimization.md) |
| I. Loshchilov, F. Hutter. *SGDR: Stochastic gradient descent with warm restarts*. ICLR 2017. | Cosine annealing (here without restarts) | `CosineAnnealing` (`src/my_torch/Schedulers.cpp`) | [4](04-optimization.md) |
| N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, R. Salakhutdinov. *Dropout: a simple way to prevent neural networks from overfitting*. JMLR 15, 2014. | Dropout (inverted form) | `Dropout` (`src/my_torch/Dropout.cpp`) | [4](04-optimization.md) |
| Y. Gal, Z. Ghahramani. *Dropout as a Bayesian approximation*. ICML 2016. | Monte Carlo dropout | `forwardMonteCarlo` ([ADR-0034](../adr/0034-mc-dropout.md)) | [4](04-optimization.md) |
| S. Ioffe, C. Szegedy. *Batch normalization*. ICML 2015. | Batch normalization | `BatchNorm` (`src/my_torch/Normalization.cpp`) | [4](04-optimization.md) |
| J. L. Ba, J. R. Kiros, G. E. Hinton. *Layer normalization*. arXiv:1607.06450, 2016. | Layer normalization | `LayerNorm`, `TokenNorm` | [4](04-optimization.md), [5](05-convolution-and-attention.md) |
| R. Pascanu, T. Mikolov, Y. Bengio. *On the difficulty of training recurrent neural networks*. ICML 2013. | Gradient clipping by global norm | `train clip_norm=` (`clip`, `src/my_torch/Network.cpp`) | [4](04-optimization.md) |
| A. Vaswani et al. *Attention is all you need*. NeurIPS 2017. | Scaled dot-product, multi-head self-attention | `SelfAttention`, `Positional` (`src/my_torch/Attention.cpp`) | [5](05-convolution-and-attention.md) |

Background for the derivations of chapters 2 and 3: I. Goodfellow, Y. Bengio, A. Courville,
*Deep Learning*, MIT Press, 2016, chapters 6 (backpropagation) and 8 (optimization).
