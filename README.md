# Data augmentation based on random spatial deformations

Authors: F. [Allender](https://igg.icube.unistra.fr/index.php/Florian_Allender), R. [Allègre](https://igg.icube.unistra.fr/index.php/R%C3%A9mi_All%C3%A8gre), C. [Wemmert](https://wemmertc.github.io/webpage/), J.-M. [Dischler](https://dpt-info.di.unistra.fr/~dischler).

## Deformation models

* Random Displacement Fields, [Simmard et al., 2003](https://ieeexplore.ieee.org/document/1227801)
* Grid-Based, [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
* Moving Least Square, [Schaefer et al., 2006](https://dl.acm.org/doi/10.1145/1141911.1141920)
* Continuous Piecewise-Affine velocity fields Based, [Freifeld et al., 2017](https://backend.orbit.dtu.dk/ws/portalfiles/portal/139267800/07814343.pdf)
* Fractal Brownian Motion with Perlin Noise, [Perlin, 1985](https://dl.acm.org/doi/10.1145/325334.325247)

## Requirements

* Python >= 3.7
* NumPy, Pillow, scikit-image
* TensorFlow 2
* TensorFlow Addons Image module
* Pytorch

## How to use

### Data preparation

1. Provide CPAB_aug.py, MLS_aug.py and prepare_data.py with the adequate parameters (folders, image size, wanted deformations and parameter values, etc.).
2. Call python3 CPAB_aug.py or python3 MLS_aug.py or python3 prepare_data.py.

### Data segmentation

1. See parser.py for the parameters and how to use them.
2. Call python3 unet.py (for precalculated augmentations) or python3 unet_live.py (for live augmentations).

## References

In progress.
