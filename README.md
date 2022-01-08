# Data augmentation based on random spatial deformations

Authors: F. [Allender](https://igg.icube.unistra.fr/index.php/Florian_Allender), R. [Allègre](https://igg.icube.unistra.fr/index.php/R%C3%A9mi_All%C3%A8gre), C. [Wemmert](https://wemmertc.github.io/webpage/), J.-M. [Dischler](https://dpt-info.di.unistra.fr/~dischler).

## Deformation models

* Random Displacement Fields, [[Simard et al., 2003](https://ieeexplore.ieee.org/document/1227801)]
* Grid-based deformations, [[Ronneberger et al., 2015](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28)]
* Moving Least Squares as-rigid-as-possible deformations, [[Schaefer et al., 2006](https://dl.acm.org/doi/10.1145/1141911.1141920)]
* Deformations based on Continuous Piecewise-Affine velocity fields (CPAB),  
[[Freifeld et al., 2017](https://ieeexplore.ieee.org/document/7814343)], 
[[Detlefsen, 2018](https://github.com/SkafteNicki/libcpab)]
* Fractal Brownian Motion implemented with Perlin Noise, [[Perlin, 1985](https://dl.acm.org/doi/10.1145/325334.325247)], 
[[Lagae et al., 2010](https://diglib.eg.org/handle/10.2312/egst.20101059.001-019)]

## Requirements

* Python >= 3.7
* NumPy, Pillow, scikit-image
* TensorFlow 2
* TensorFlow Addons Image module
* PyTorch

## How to use

### Data preparation

1. Provide prepare_data.py with the adequate parameter values (folders,
image size, wanted deformations and parameter values, etc.).

2. In order to use the deformation model based on cell nuclei centers (CNB), you
first have to run a cell nuclei segmentation method. We recommend the method by
[[Mahmood et al., 2019](https://ieeexplore.ieee.org/document/8756037)]. A copy of the
[authors' code](https://github.com/mahmoodlab/NucleiSegmentation) is available
in the nuclei_segmentation directory, with some specific processing of the segmentation
results to get the cell nuclei centers. You have to download the pretrained models
(see [here](https://github.com/mahmoodlab/NucleiSegmentation#testing)) or train
your own models. Then adjust parameter values in nuclei_segmentation.py and run
`python3 nuclei_segmentation.py`.

3. Call `python3 prepare_data.py` to generate the augmented data.

### Data segmentation

1. See parser.py for the parameters and how to use them.
2. Call `python3 unet.py` (for precalculated augmentations) or
`python3 unet_live.py` (for live augmentations).

## References

In progress.
