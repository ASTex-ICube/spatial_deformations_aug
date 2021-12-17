# Data augmentation based on spatial deformations

## Deformation models

In progress.

## Requirements

* Python packages : numpy, Pillow, skimage
* Tensorflow 2
* Tensorflow addons : tfa.image
* Pytorch

## How to use

### Data preparation

1. Provide CPAB_aug.py, MLS_aug.py and prepare_data.py with the adequate parameters (folders, image size, wanted deformations and parameters...)
2. Call python3 CPAB_aug.py or python3 MLS_aug.py or python3 prepare_data.py

### Data segmentation

1. See parser.py for the parameters and how to use them.
2. Call python3 unet.py (for precalculated augmentations) or python3 unet_live.py (for live augmentations)

## References

In progress.
