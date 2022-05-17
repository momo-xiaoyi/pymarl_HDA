# Rethinking Individual Global Max in Cooperative Multi-Agent Reinforcement Learning

## note
This codebase is based on PyMARL and SMAC codebases which are open-sourced. The implementation of the following methods can also be found in this codebase, which are finished by the authors of following papers:

- [**QPLEX**: QPLEX: Duplex Dueling Multi-Agent Q-Learning](https://arxiv.org/abs/2008.01062)
- [**QMIX**: QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [**DMIX**: DFAC Framework: Factorizing the Value Function via Quantile Mixture for Multi-Agent Distributional Q-Learning](https://arxiv.org/abs/2102.07936)

And thanks for the advice of RIIT, we set all algorithm to almost the same parameters (but probably not the optimal ones).

- [**RIIT**: Rethinking the Implementation Tricks and Monotonicity Constraint in Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2102.03479)

## Installation instructions

Build the Dockerfile using 
```shell
cd docker
bash build.sh
```

Set up StarCraft II and SMAC:
```shell
bash install_sc2.sh
```

This will download SC2 into the 3rdparty folder and copy the maps necessary to run over.

The requirements.txt file can be used to install the necessary packages into a virtual environment (not recomended).

## Run an experiment of zero observation range

```shell
# bash run_0.sh config_name_list map_name_list (threads_num arg_list gpu_list experinments_num)
bash run_0.sh qmix,qmix_HDA， 5m_vs_6m, 1 , 0, 3
```

`xxx_list` is separated by `,`.

All results will be stored in the `Results` folder.

# Citation
```
wait for update 
```

