# [CGBA: Curvature-aware Geometric Black-box Attack](https://arxiv.org/abs/2308.03163)
Welcome to our repository featuring the official implementation of the CGBA algorithm. This work is accepted for publication in ICCV 2023. <br>
The arXiv version of the paper is available [here](https://arxiv.org/abs/2308.03163).
## Requirements
Before executing the code, ensure that the following packages are installed in your environment:
* PyTorch and Trochvision
* Numpy
* Os
* SciPy
  
Or you can type the following to create an environment:  

```
conda env create -f cgba_env.yml
```

## Run
1. To execute the non-targeted attack, run `Non_targeted_attack.py`.
2. To execute the targeted attack, run `Targeted_attack.py`.

## Citation
Please consider citing our paper in your publications if it contributes to your research.
```
@article{reza2023cgba,
  title={CGBA: Curvature-aware Geometric Black-box Attack},
  author={Reza, Md Farhamdur and Rahmati, Ali and Wu, Tianfu and Dai, Huaiyu},
  journal={arXiv preprint arXiv:2308.03163},
  year={2023}
}
```

