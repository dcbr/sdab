# State-dependent action bounds
This repository provides the necessary code to reproduce all supplementary experiments of the *"Enforcing Hard State-Dependent Action Bounds on Deep Reinforcement Learning Policies"* paper [1, Appendix B].

An example implementation of state-dependent action bounds is provided for the SAC method, using [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) and [Pytorch](https://pytorch.org).
Custom state-dependent action bounds are defined for the `Pendulum-v1` and `LunarLanderContinuous-v2` [OpenAI gym environments](https://gymlibrary.ml).
Refer to the paper's supplementary material for further details.

# Installation

1. Clone this repository.

   ``git clone https://github.com/dcbr/sdab``

   ``cd sdab``

2. Install the required packages.
   Optionally, create a virtual environment first (using e.g. conda or venv).

   ``python -m pip install -r requirements.txt``

# Usage

Run the `action_bounds` script with suitable arguments to train the models or evaluate and analyze their performance.
For example

``python action_bounds.py --mode train --envs LunarLanderContinuous-v2 --rescale lin hyp``

to train on the lunar lander environment (with stabilizing action bounds) for both the linear and hyperbolic rescaling function.

To reproduce all results of Appendix B, first train all models with ``python action_bounds.py``, followed by the analysis ``python action_bounds.py --mode analyze``. Beware that this might take a while to complete, depending on your hardware!

A summary of the most relevant parameters to this script is provided below.
Check ``python action_bounds.py --help`` for a full overview of supported parameters.

| Parameter   | Supported values                          | Description                                                                                                                                                                                                |
|:------------|:------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--mode`    | `train`, `eval`, `analyze`                | Run mode. Either train models, evaluate (and visualize) them or analyze and summarize the results (creating the plots shown in the paper).                                                                 |
| `--envs`    | `Pendulum-v1`, `LunarLanderContinuous-v2` | OpenAI gym environment ID.                                                                                                                                                                                 |
| `--algs`    | `sac`, `bsac`                             | Reinforcement learning algorithm to use. Either the bounded SAC algorithm (`bsac`), with enforced state-dependent action bounds, or the default SAC algorithm (`sac`), without enforcement of such bounds. |
| `--rescale` | `lin`, `pwl`, `hyp`, `clip`               | Rescaling function &sigma; to use. Either linear (`lin`), piecewise linear (`pwl`) or hyperbolic (`hyp`) rescaling; or clipping (`clip`).                                                                  |
| `--seeds`   | Any integer number *N*                    | Experiments are repeated for all of the provided seeds. Can also be a negative number *-N* in which case *N* seeds are randomly chosen.                                                                    |

# References

[1] De Cooman, B., Suykens, J., Ortseifen, A.: Enforcing hard state-dependent action bounds on deep reinforcement learning policies. Accepted for *8th International Conference on Machine Learning, Optimization & Data Science, LOD 2022*.
