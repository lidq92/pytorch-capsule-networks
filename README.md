## PyTorch Implementation of Capsule Networks in NIPS2017 and ICLR2018
[![License](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](LICENSE)

This repository is a PyTorch implementation of the following papers ([[1]](http://papers.nips.cc/paper/6975-dynamic-routing-between-capsules.pdf) and [[2]](https://openreview.net/pdf?id=HJWLfGWRb)):
```
@incollection{NIPS2017_6975,
title = {Dynamic Routing Between Capsules},
author = {Sabour, Sara and Frosst, Nicholas and Hinton, Geoffrey E},
booktitle = {Advances in Neural Information Processing Systems 30},
editor = {I. Guyon and U. V. Luxburg and S. Bengio and H. Wallach and R. Fergus and S. Vishwanathan and R. Garnett},
pages = {3856--3866},
year = {2017},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/6975-dynamic-routing-between-capsules.pdf}
}
@inproceedings{
e2018matrix,
title={Matrix capsules with {EM} routing},
author={Geoffrey E Hinton and Sara Sabour and Nicholas Frosst},
booktitle={International Conference on Learning Representations},
year={2018},
url={https://openreview.net/forum?id=HJWLfGWRb},
}
```

You can also refer to the [official TensorFlow implemenation](https://github.com/Sarasra/models/tree/master/research/capsules) of the first paper, which is provided by the first author [Sara Sabour](https://github.com/Sarasra/).

### Capsule framework
![Capsule framework](capsule-framework.png)
> Image credits: [_@bojone_](https://github.com/bojone)
 
### How to run?
```bash
CUDA_VISIBLE_DEVICES=0 python main.py
```
Please see main.py and config.yaml for more details.
```bash
tensorboard --logdir='./logs' --port=6006 # TensorBoard Visualization
```

### Acknowledgements
Thanks to [_@danielhavir_](https://github.com/danielhavir/capsule-network) and [_@ducminhkhoi_](https://github.com/ducminhkhoi/EM_Capsules)

### Other Implementations of CapsNet in NIPS 2017
- PyTorch:
  - [danielhavir/capsule-network](https://github.com/danielhavir/capsule-network)
  - [XifengGuo/CapsNet-Pytorch](https://github.com/XifengGuo/CapsNet-Pytorch)
  - [timomernick/pytorch-capsule](https://github.com/timomernick/pytorch-capsule)
  - [gram-ai/capsule-networks](https://github.com/gram-ai/capsule-networks)
  - [nishnik/CapsNet-PyTorch](https://github.com/nishnik/CapsNet-PyTorch.git)
  - [leftthomas/CapsNet](https://github.com/leftthomas/CapsNet)
  
- Keras:   
  - [XifengGuo/CapsNet-Keras](https://github.com/XifengGuo/CapsNet-Keras)   
  
- TensorFlow:
  - [naturomics/CapsNet-Tensorflow](https://github.com/naturomics/CapsNet-Tensorflow.git)   
  - [InnerPeace-Wu/CapsNet-tensorflow](https://github.com/InnerPeace-Wu/CapsNet-tensorflow)   
  - [chrislybaer/capsules-tensorflow](https://github.com/chrislybaer/capsules-tensorflow)
  
- MXNet:
  - [AaronLeong/CapsNet_Mxnet](https://github.com/AaronLeong/CapsNet_Mxnet)
  
- Chainer:
  - [soskek/dynamic_routing_between_capsules](https://github.com/soskek/dynamic_routing_between_capsules)

- Matlab:
  - [yechengxi/LightCapsNet](https://github.com/yechengxi/LightCapsNet)

### Other Implementations of CapsNet in ICLR 2018
- PyTorch:
  - [ducminhkhoi/EM_Capsules](https://github.com/ducminhkhoi/EM_Capsules)
