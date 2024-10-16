# Code for OCCAM: Online Continuous Controller Adaptation with Meta-Learned Models
This repository contains the code for our paper, [OCCAM: Online Continuous Controller Adaptation with Meta-Learned Models](https://openreview.net/forum?id=xeFKtSXPMd), appearing at CoRL 2024. We are currently working on adding documentation and cleaning up the API for others to use. The "main" branch currently contains the version of the code used to train models, generate data, and run experiments for the paper submission.

## Paper Abstract

Control tuning and adaptation present a significant challenge to the usage of robots in diverse environments. It is often nontrivial to find a single set of control parameters by hand that work well across the broad array of environments and conditions that a robot might encounter. Automated adaptation approaches must utilize prior knowledge about the system while adapting to significant domain shifts to find new control parameters quickly. In this work, we present a general framework for online controller adaptation that deals with these challenges. We combine meta-learning with Bayesian recursive estimation to learn prior predictive models of system performance that quickly adapt to online data, even when there is significant domain shift. These predictive models can be used as cost functions within efficient sampling-based optimization routines to find new control parameters online that maximize system performance. Our framework is powerful and flexible enough to adapt controllers for four diverse systems: a simulated race car, a simulated quadrupedal robot, and a simulated and physical quadrotor.

## Installing Dependencies
To install python dependencies, create a virtual environment, activate it, and run 
```
pip install -r requirements.txt
pip install -e .
```
