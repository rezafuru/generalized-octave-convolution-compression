I wanted to use the generalized octave convolution layers as introduced by Akbari et al. for my own experiments and noticed that they did not provide any code for their compression model.

This is a quick and dirty pytorch implementation of 

A. Generalized (Transpose) Octave Convolutions 

![](imgs/goconvs.png?raw=true "")

B. The proposed Compression Model

![](imgs/compression_model.png?raw=true "")

References:

Akbari, M., Liang, J., Han, J. and Tu, C., 2020. Generalized octave convolutions for learned multi-frequency image compression. *arXiv preprint arXiv:2002.10032*

Ballé, J., Minnen, D., Singh, S., Hwang, S.J. and Johnston, N., 2018. Variational image compression with a scale hyperprior. *arXiv preprint arXiv:1802.01436*.

Ballé, J., Laparra, V. and Simoncelli, E.P., 2016. End-to-end optimized image compression. *arXiv preprint arXiv:1611.01704*.

Minnen, D., Ballé, J. and Toderici, G.D., 2018. Joint autoregressive and hierarchical priors for learned image compression. *Advances in neural information processing systems*, *31*.

Bégaint, J., Racapé, F., Feltman, S. and Pushparaja, A., 2020.  Compressai: a pytorch library and evaluation platform for end-to-end  compression research. *arXiv preprint arXiv:2011.03029*.