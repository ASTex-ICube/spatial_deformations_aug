# Data augmentation based on random spatial deformations

Authors: F. [Allender](https://igg.icube.unistra.fr/index.php/Florian_Allender), R. [Allègre](https://igg.icube.unistra.fr/index.php/R%C3%A9mi_All%C3%A8gre), C. [Wemmert](https://wemmertc.github.io/webpage/), J.-M. [Dischler](https://dpt-info.di.unistra.fr/~dischler).

## Deformation models

In progress.

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
