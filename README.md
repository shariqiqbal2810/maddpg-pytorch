# MADDPG-PyTorch
PyTorch Implementation of MADDPG from [*Multi-Agent Actor-Critic for Mixed
Cooperative-Competitive Environments*](https://arxiv.org/abs/1706.02275) (Lowe et. al. 2017)

## Requirements

* [OpenAI baselines](https://github.com/openai/baselines)
* My [fork](https://github.com/shariqiqbal2810/multiagent-particle-envs) of Multi-agent Particle Environments
* [PyTorch](http://pytorch.org/)
* [OpenAI Gym](https://github.com/openai/gym)
* [Tensorboard-Pytorch](https://github.com/lanpa/tensorboard-pytorch) (for logging)

## How to Run

All training code is contained within `main.py`. To view options simply run:

```
python main.py --help
```

## Results (in progress)

### Physical Deception

In this task, the two blue agents are rewarded by minimizing the closest of their distances to the green landmark (only one needs to be close to get optimal reward), while maximizing the distance of the red adversary from the green landmark. The red adversary is rewarded by minimizing it's distance to the green landmark; however, on any given trial, it does not know which landmark is green, so it must follow the blue agents. As such, the blue agents should learn to deceive the red agent by covering *both* landmarks.

<img src="assets/physical_deception/1.gif?raw=true" width="33%"> <img src="assets/physical_deception/2.gif?raw=true" width="33%"> <img src="assets/physical_deception/3.gif?raw=true" width="33%">

## Not Implemented

There are a few items from the paper that have not been implemented in this repo

* Ensemble Training
* Inferring other agents' policies
* Mixed continuous/discrete action spaces

## Acknowledgements

The OpenAI baselines [Tensorflow implementation](https://github.com/openai/baselines/tree/master/baselines/ddpg) and Ilya Kostrikov's [Pytorch implementation](https://github.com/ikostrikov/pytorch-ddpg-naf) of DDPG were used as references. After the majority of this codebase was complete, OpenAI released their [code](https://github.com/openai/maddpg) for MADDPG, and I made some tweaks to this repo to reflect some of the details in their implementation (e.g. gradient norm clipping and policy regularization).