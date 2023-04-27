# Data augmentation based on random spatial deformations

This is the page of our paper entitled *Data augmentation based on spatial deformations for histopathology:
An evaluation in the context of glomeruli segmentation* published in Computer Methods and Programs in Biomedicine, Volume 221, 106919, 2022, ISSN 0169-2607, https://doi.org/10.1016/j.cmpb.2022.106919.

Authors: F. [Allender](https://igg.icube.unistra.fr/index.php/Florian_Allender), R. [Allègre](https://igg.icube.unistra.fr/index.php/R%C3%A9mi_All%C3%A8gre), C. [Wemmert](https://wemmertc.github.io/webpage/), J.-M. [Dischler](https://dpt-info.di.unistra.fr/~dischler).

A preprint version of the paper is available following [this link](https://seafile.unistra.fr/f/fa3e480dcda54b3392e5/?dl=1).

A complementary work has been presented at the 2022 IEEE-EMBS International Conference on Biomedical and Health Informatics (BHI), in the paper *Conditional image synthesis for improved segmentation of glomeruli in renal histopathological images*, https://doi.org/10.1109/BHI56158.2022.9926880.

A preprint version of the paper is available following [this link](https://seafile.unistra.fr/f/a5b1e797205542f0a905/?dl=1)

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

If you find this code useful for your research, consider citing:

```
@article{allender2022cmpb,
	title = {Data augmentation based on spatial deformations for histopathology: An evaluation in the context of glomeruli segmentation},
	journal = {Computer Methods and Programs in Biomedicine},
	volume = {221},
	pages = {106919},
	year = {2022},
	issn = {0169-2607},
	doi = {https://doi.org/10.1016/j.cmpb.2022.106919},
	url = {https://www.sciencedirect.com/science/article/pii/S0169260722003017},
	author = {Florian Allender and Rémi Allègre and Cédric Wemmert and Jean-Michel Dischler},
	keywords = {Histopathological images, Glomeruli segmentation, Data augmentation, Random spatial deformations}
}
```

```
@inproceedings{allender2022bhi,
	author={Allender, Florian and Allégre, Rémi and Wemmert, Cédric and Dischler, Jean-Michel},
	booktitle={2022 IEEE-EMBS International Conference on Biomedical and Health Informatics (BHI)}, 
	title={Conditional image synthesis for improved segmentation of glomeruli in renal histopathological images}, 
	year={2022},
	volume={},
	number={},
	pages={1-5},
	doi={10.1109/BHI56158.2022.9926880}}
```
