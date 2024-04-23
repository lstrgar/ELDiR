# ELDiR: Evolution & learning in differentiable robots

Code for [Evolution and learning in differentiable robots](), Strgar et al. 

To appear the proceedings of [Robotics: Science and Systems 2024](https://roboticsconference.org/).

The ELDiR project website is located [here](https://sites.google.com/view/eldir)

<!---
<p align="center">
  <img src="https://github.com/lstrgar/ELDiR/assets/16822502/05230e5d-5f5e-487d-95c6-895e58b3590a" alt="animated" />
</p>
-->

![fig2](https://github.com/lstrgar/ELDiR/assets/16822502/07409858-bfcd-45ea-9e05-e6598b28c44e)

## Abstract

The automatic design of robots has existed for 30 years but has been constricted by serial non-differentiable design evaluations, premature convergence to simple bodies or clumsy behaviors, and a lack of sim2real transfer to physical machines. Thus, here we employ massively-parallel differentiable simulations to rapidly and simultaneously optimize individual neural control of behavior across a large population of candidate body plans and return a fitness score for each design based on the performance of its fully optimized behavior. 

Non-differentiable changes to the mechanical structure of each robot in the population — mutations that rearrange, combine, add, or remove body parts — were applied by a genetic algorithm in an outer loop of search, generating a continuous flow of novel morphologies with highly-coordinated and graceful behaviors honed by gradient descent. This enabled the exploration of several orders-of-magnitude more designs than all previous methods, despite the fact that robots here have the potential to be much more complex, in terms of number of independent motors, than those in prior studies. 

We found that evolution reliably produces “increasingly differentiable” robots: body plans that smooth the loss landscape in which learning operates and thereby provide better training paths toward performant behaviors. Finally, one of the highly differentiable morphologies discovered in simulation was realized as a physical robot and shown to retain its optimized behavior. This provides a cyberphysical platform to investigate the relationship between evolution and learning in biological systems and broadens our understanding of how a robot’s physical structure can influence the ability to train policies for it.

## Citation

If you find our paper or this repository useful or relevant to your work please consider citing us.

```
@inproceedings{
strgar2024evolution&learning,
title={Evolution and learning in differentiable robots},
author={Strgar, Luke and Matthews, David and Hummer, Tyler and Kriegman, Sam},
booktitle={Robotics: Science and Systems},
year={2024},
url={https://openreview.net/forum?id=gDYszdccm7}
}
```

## Installation

Clone this repository and install the following using your preferred python environment or package managment tool:

```
python        3.10.13
taichi        1.7.0
numpy         1.24.3
scikit-image  0.22.0
matplotlib    3.8.3
seaborn       0.13.2
moviepy       1.0.3
tqdm          4.66.1
jupyter       1.0.0
ipykernel     6.29.4
```

Below is a detailed installation using `conda` and `pip`:

```
$ conda create --name ELDiR
$ conda activate ELDiR
$ conda install python=3.10.13
$ pip install taichi==1.7.0 numpy==1.24.3 scikit-image==0.22.0 matplotlib==3.8.3 seaborn==0.13.2 moviepy==1.0.3 tqdm==4.66.1 jupyter==1.0.0 ipykernel==6.29.4
```

This installation process was tested on MacOS Sonoma and Ubuntu 22.04.

## Usage

This repository assumes you are working on a machine with at least one CUDA enabled GPU. The code can be modified to run on CPU by changing only a few lines of code. 

Executing `python run.py` is all that is required to begin evolution and learning in a population of random robots. `run.py` will run until it is manually interrupted. Since the process may be running for an extended period of time and since `run.py` prints basic logging information to stdout you may want to redirect stdout and/or stderr such as: `python run.py &> run.log`.

**However**,`run.py` can accept several command line arguments. Execute `python run.py --help` to learn more about what these are. Depending on your hardware, it is likely you will need to modify the population size, the number of learning iterations, and/or numeric GPU IDs for population-level parallelism. The corresponding flags for these parameters are: `--n_robots`, `iters`, `gpu_ids`.

`analyze.ipynb` can be used to visualize population level performance as well as learned robotic locomtion of evolved bodies. Both analyses can be performed after `run.py` is stopped or at any point while it is running. 

<!---
Below is an example of the type of video that will be produced:

<p align="center">
  <img src="https://github.com/lstrgar/ELDiR/assets/16822502/01e13408-72e4-46cd-8229-da5de30c9ceb" alt="animated" />
</p>
-->


## Future Work

Note: non-flat terrain is not yet supported in this repository. It will be soon (as of 5/16/24).

We welcome contributions or suggestions from both the evolutionary robotics / computation and differentiable physics simulation communities. 

Please do not hesitate to reach out directly or open a github issue to start a conversation. Thank you for your interest in our work. 
