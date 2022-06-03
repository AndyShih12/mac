# Training and Inference on Any-Order Autoregressive Models the Right Way

This repository contains code for the paper:

[Training and Inference on Any-Order Autoregressive Models the Right Way](https://arxiv.org/abs/2205.13554) \
by Andy Shih, Dorsa Sadigh, Stefano Ermon

<br>

Any-Order Autoregressive Models (AO-ARMs) are a powerful model family that can compute arbitrary conditionals and marginals. Broadly defined, some examples of AO-ARMs are:
- [A Deep and Tractable Density Estimator](https://arxiv.org/abs/1310.1757)
- [BERT](https://arxiv.org/abs/1810.04805)
- [XLNet](https://arxiv.org/abs/1906.08237)
- [Arbitrary Conditioning with Energy](https://arxiv.org/abs/2102.04426)
- [Autoregressive Diffusion Models](https://arxiv.org/abs/2110.02037)

We introduce **MAC: Mask-Tuned Arbitrary Conditional Models**, which improve AO-ARMs by training on a smaller set of univariate conditionals while still maintaining support for efficient arbitrary conditional and marginal inference. In short, MAC improves model performance without sacrificing tractability.

--------------------

## Installation
```
pip install -r requirements.txt
```

## Commands

The current batch sizes assume a GPU with 48GB memory.

### ARDM
```
python image_main.py dataset=CIFAR10 mask.strategy=none mask.order=random batch_size=24

python image_main.py dataset=IMAGENET32 mask.strategy=none mask.order=random batch_size=24

python lang_main.py dataset=TEXT8 mask.strategy=none mask.order=random batch_size=180
```

### MAC
```
python image_main.py dataset=CIFAR10 mask.strategy=marginal mask.order=spaced mask.normalize_cardinality=True batch_size=24

python image_main.py dataset=IMAGENET32 mask.strategy=marginal mask.order=spaced mask.normalize_cardinality=True batch_size=24

python lang_main.py dataset=TEXT8 mask.strategy=marginal mask.order=spaced mask.normalize_cardinality=True batch_size=180
```