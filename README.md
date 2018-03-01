# MADDPG-PyTorch
PyTorch Implementation of MADDPG from [*Multi-Agent Actor-Critic for Mixed
Cooperative-Competitive Environments*](https://arxiv.org/abs/1706.02275) (Lowe et. al. 2017)

## Requirements

* [OpenAI baselines](https://github.com/openai/baselines), commit hash: 98257ef8c9bd23a24a330731ae54ed086d9ce4a7
* My [fork](https://github.com/shariqiqbal2810/multiagent-particle-envs) of Multi-agent Particle Environments
* [PyTorch](http://pytorch.org/), version: 0.3.0.post4
* [OpenAI Gym](https://github.com/openai/gym), version: 0.9.4
* [Tensorboard](https://github.com/tensorflow/tensorboard), version: 0.4.0rc3 and [Tensorboard-Pytorch](https://github.com/lanpa/tensorboard-pytorch), version: 1.0 (for logging)

The versions are just what I used and not necessarily strict requirements.

## How to Run

All training code is contained within `main.py`. To view options simply run:

```
python main.py --help
```

## Results

### Physical Deception

In this task, the two blue agents are rewarded by minimizing the closest of their distances to the green landmark (only one needs to be close to get optimal reward), while maximizing the distance of the red adversary from the green landmark. The red adversary is rewarded by minimizing it's distance to the green landmark; however, on any given trial, it does not know which landmark is green, so it must follow the blue agents. As such, the blue agents should learn to deceive the red agent by covering *both* landmarks.

<img src="assets/physical_deception/1.gif?raw=true" width="33%"> <img src="assets/physical_deception/2.gif?raw=true" width="33%"> <img src="assets/physical_deception/3.gif?raw=true" width="33%">

### Cooperative Communication

This task involves two agents, one that is stationary and one that can move. The stationary agent sees the color of the other agent as its observation, and outputs a one-hot communication vector as its action. The moving agent receives the communication vector, as well as its relative distance to all landmarks on the screen; however, it does not know its own color. The goal of both agents is for the moving agent to reach the landmark that matches its own color. Thus, the agents must learn to communicate such that the moving agent knows where to go on each randomized trial.

<img src="assets/cooperative_communication/1.gif?raw=true" width="33%"> <img src="assets/cooperative_communication/2.gif?raw=true" width="33%"> <img src="assets/cooperative_communication/3.gif?raw=true" width="33%">

### Predator-Prey

This task involves a single prey agent (in green) and a team of three predators (in red). The prey agent is 30% faster than the predators, so the predators must learn how to team up in order to catch the prey.

In the trials below, the prey agent uses DDPG as its learning algorithm.

<img src="assets/predator_prey/1.gif?raw=true" width="33%"> <img src="assets/predator_prey/2.gif?raw=true" width="33%"> <img src="assets/predator_prey/3.gif?raw=true" width="33%">

## Not Implemented

There are a few items from the paper that have not been implemented in this repo

* Ensemble Training
* Inferring other agents' policies
* Mixed continuous/discrete action spaces

## Acknowledgements

The OpenAI baselines [Tensorflow implementation](https://github.com/openai/baselines/tree/master/baselines/ddpg) and Ilya Kostrikov's [Pytorch implementation](https://github.com/ikostrikov/pytorch-ddpg-naf) of DDPG were used as references. After the majority of this codebase was complete, OpenAI released their [code](https://github.com/openai/maddpg) for MADDPG, and I made some tweaks to this repo to reflect some of the details in their implementation (e.g. gradient norm clipping and policy regularization).