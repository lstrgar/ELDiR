# ELDiR: Evolution and learning in differentiable robots

**This branch (`2024-RSS-Strgar`) of the repository contains code to reproduce experimental results from our paper.**

**We've also created an adaptable and modular framework capturing the same and more functionality in the `main` [branch](https://github.com/lstrgar/ELDiR/tree/main). This can be used to study evolutionary algorithms with CUDA-accelerated robot learning and evaluation in a differentiable simulator. We HIGHLY recommend using the `main` branch for any further experimentation.**

<p align="center">
  <img src="./assets/1.gif" alt="animated" style="width:700px; height:auto"/>
</p>

### :star: Project Site & Video: [ELDiR](https://sites.google.com/view/eldir)

### :star: Paper: [Evolution and learning in differentiable robots, Strgar et al.](https://arxiv.org/abs/2405.14712)

### :star: Citation: If our work is useful to you please [cite our paper](https://github.com/lstrgar/ELDiR/blob/2024-RSS-Strgar/README.md#citation)

### :star: Appearing in: 
- #### *[Proceedings of Robotics: Science and Systems 2024](https://roboticsconference.org/)*
- #### *[Virtual Creatures Competition 2024](https://sites.google.com/view/vcc-2024)*
- #### *[Conference on Artificial Life (Late Breaking Abstract)](https://2024.alife.org/)*

## Abstract

The automatic design of robots has existed for 30 years but has been constricted by serial non-differentiable design evaluations, premature convergence to simple bodies or clumsy behaviors, and a lack of sim2real transfer to physical machines. Thus, here we employ massively-parallel differentiable simulations to rapidly and simultaneously optimize individual neural control of behavior across a large population of candidate body plans and return a fitness score for each design based on the performance of its fully optimized behavior. 

Non-differentiable changes to the mechanical structure of each robot in the population — mutations that rearrange, combine, add, or remove body parts — were applied by a genetic algorithm in an outer loop of search, generating a continuous flow of novel morphologies with highly-coordinated and graceful behaviors honed by gradient descent. This enabled the exploration of several orders-of-magnitude more designs than all previous methods, despite the fact that robots here have the potential to be much more complex, in terms of number of independent motors, than those in prior studies. 

We found that evolution reliably produces “increasingly differentiable” robots: body plans that smooth the loss landscape in which learning operates and thereby provide better training paths toward performant behaviors. Finally, one of the highly differentiable morphologies discovered in simulation was realized as a physical robot and shown to retain its optimized behavior. This provides a cyberphysical platform to investigate the relationship between evolution and learning in biological systems and broadens our understanding of how a robot’s physical structure can influence the ability to train policies for it.

## Installation
<pre lang="bash">
git clone git@github.com:lstrgar/ELDiR.git; cd ELDiR
conda create --name ELDiR python=3.10.13 --yes
conda activate ELDiR
pip install -r requirements.txt
</pre>

**Notes:** 
- We find that an isolated `conda` environment is easy to work with; however, this is not required so long as you have a working `pip` installation.
- You may need to install `ffmpeg` in order to run the `visualize-results.ipynb` notebook. This can be accomplished using `homebrew` on mac or `apt` on linux. 

## Usage

This repository assumes you are working on a machine with at least one CUDA enabled GPU. The code can be modified to run on CPU by changing only a few lines of code.

Executing `python run.py` is all that is required to begin evolution and learning in a population of random robots. `run.py` will run until it is manually interrupted. Since the process may be running for an extended period of time and since `run.py` prints basic logging information to stdout you may want to redirect stdout and/or stderr such as: `python run.py &> run.log`.

**However**,`run.py` can accept several command line arguments. Execute `python run.py --help` to learn more about what these are. Depending on your hardware, it is likely you will need to modify the population size, the number of learning iterations, and/or numeric GPU IDs for population-level parallelism. The corresponding flags for these parameters are: `--n_robots`, `iters`, `gpu_ids`.

`visualize-results.ipynb` can be used to visualize population level performance as well as learned robotic locomtion of evolved bodies. Both analyses can be performed after `run.py` is stopped or at any point while it is running. 

## Citation

If our work is useful to you please cite our paper:

```
@inproceedings{
strgar2024evolutionandlearning,
title={Evolution and learning in differentiable robots},
author={Strgar, Luke and Matthews, David and Hummer, Tyler and Kriegman, Sam},
booktitle={Robotics: Science and Systems},
year={2024},
url={https://arxiv.org/abs/2405.14712}
}
```

<p align="center">
  <img src="./assets/5.gif" alt="animated" style="width:700px; height:auto"/>
</p>
