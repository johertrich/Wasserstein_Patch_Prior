# Wasserstein Patch Prior for Image Superresolution

This repository contains the implementations corresponding to the paper [1] availible at https://arxiv.org/abs/2109.12880.
Please cite the paper, if you use the code.

The repository implements image superresolution for textures or material microstructures.
The file `wgenpatex.py` is adapted from the implementation of [2] availible at https://github.com/ahoudard/wgenpatex.

For questions and bug reports, please contact Johannes Hertrich (j.hertrich@math.tu-berlin.de).

## Requirements and Usage

The code is written in Python using 

- PyTorch 1.9.0 
- Pykeops 1.5
- scipy 1.6.2
- Scikit-image 0.18.1
- Numpy

Usually the code is also compatible with some other versions of the corresponding packages.

The scripts `run_diam.py` and `run_FS.py` reproduce the 2D examples of Section 4.2 of the paper [1].
The core function is the function `optim_synthesis` from the file `wgenpatex.py`.
A short discription of each script/function can be found in the corresponding header.

## References

[1] J. Hertrich, A. Houdard and C. Redenbach.  
Wasserstein Patch Prior for Image Superresolution.  
ArXiv Preprint#2109.12880

[2] A. Houdard, A. Leclaire, N. Papadakis and J. Rabin.  
Wasserstein Generative Models for Patch-based Texture Synthesis.  
ArXiv Preprin#2007.03408

