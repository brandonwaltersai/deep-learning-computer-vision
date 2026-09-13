# Deep Learning: CNN Architecture Search & DCGAN on Fashion-MNIST

Two deep-learning workshops from my data analytics coursework (DATA 655 —
Deep Learning & Neural Networks) — a discriminative model (CNN classifier)
and a generative model (DCGAN) on the same benchmark dataset, included here
as evidence of core computer-vision fundamentals rather than novel research.

## What's here

### `notebooks/cnn_fashion_mnist_architecture_experiments.ipynb`
Trains a baseline CNN on Fashion-MNIST, then runs four controlled
architecture variants against the same baseline to isolate the effect of
a single design choice at a time:

| Variant | Change from baseline | Test accuracy |
|---|---|---|
| Baseline | 3×3 kernels, 2 conv layers, max pooling | **91.86%** |
| Larger kernels | 5×5 kernels | 91.86% |
| Deeper network | +1 conv layer (3 total) | 91.28% |
| Stride=2, no pooling | strided convs replace pooling | 90.77% |
| Average pooling | max pool → average pool | 90.24% |

**Result: going wider on kernel size bought nothing here, but going
deeper or swapping pooling strategy quietly cost 0.6–1.6 points of test
accuracy** — on a small, low-resolution benchmark like Fashion-MNIST, the
simplest architecture won. Full per-epoch curves, confusion matrices, and
per-class precision/recall are in the notebook.

### `notebooks/dcgan_fashion_mnist.ipynb`
A DCGAN (Radford et al., 2016) trained from scratch on Fashion-MNIST:
strided-convolution discriminator, transposed-convolution generator,
one-sided label smoothing on real labels, Adam (β₁=0.5) for both
networks. Trained for 40 epochs on Apple Silicon GPU (Metal).

- Generator/discriminator loss curves: [`docs/figures/training_losses.png`](docs/figures/training_losses.png)
- Real vs. generated sample grids: [`docs/figures/real_samples_grid.png`](docs/figures/real_samples_grid.png) vs. [`docs/figures/samples_grid_final.png`](docs/figures/samples_grid_final.png)
- Generation quality over training: [`docs/figures/`](docs/figures/) (sampled every 2 epochs)

See [`docs/results.md`](docs/results.md) for the full methodology and honest
discussion of what a from-scratch, non-conditional DCGAN can and can't do
on this dataset in 40 epochs.

## Why these two together

A CNN classifier and a GAN generator are trained with opposite objectives
on the same pixels — one learns to compress an image down to "which of 10
classes," the other learns to expand noise up to "a plausible image."
Pairing a discriminative and a generative model on the same dataset is a
deliberate choice to show both halves of the vision toolkit rather than
picking whichever one had a flashier result.

## Stack

Python · TensorFlow/Keras · scikit-learn (metrics) · seaborn/matplotlib ·
NumPy

## Author

Brandon Walters — [LinkedIn](https://www.linkedin.com/in/bw172b29208/)
