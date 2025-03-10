# Decentralized Projected Riemannian Gradient Descent (DPRGD)
This is a repository for the paper *Decentralized Online Riemannian Optimization with Dynamic Environments*. Our experiments cover two Hadamard manifolds: hyperbolic spaces and the space of symmetric prositive definite (SPD) matrices. 

## Getting Started

Create and activate conda environments and install necessary dependencies.

```
conda create --name opt python=3.10
conda activate opt
pip install -r requirements.txt
```

## Files

1. hyperboloid.py and spd.py in utils provide basic functions for hyperbolic spaces and the space of SPD matrices.
2. simulation.ipynb presents all the simulation experiments.
3. data.ipynb presents real data analysis using [FLUXNET2015 dataset](https://fluxnet.org/data/fluxnet2015-dataset/)

## Citation

```
@article{chen2024decentralized,
  title={Decentralized Online Riemannian Optimization with Dynamic Environments},
  author={Chen, Hengchao and Sun, Qiang},
  journal={arXiv preprint arXiv:2410.05128},
  year={2024}
}
```
