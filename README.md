# MADDPG-PyTorch
PyTorch Implementation of MADDPG from *Multi-Agent Actor-Critic for Mixed
Cooperative-Competitive Environments* (Lowe et. al. 2017)

## Requirements

* [OpenAI baselines](https://github.com/openai/baselines)
* My [fork](https://github.com/shariqiqbal2810/multiagent-particle-envs) of Multi-agent Particle Environments
* [PyTorch](http://pytorch.org/)
* [OpenAI Gym](https://github.com/openai/gym)
* [Tensorboard-Pytorch](https://github.com/lanpa/tensorboard-pytorch) (for logging)


The OpenAI baselines [Tensorflow implementation](https://github.com/openai/baselines/tree/master/baselines/ddpg) and Ilya Kostrikov's [Pytorch implementation](https://github.com/ikostrikov/pytorch-ddpg-naf) of DDPG were used as references. After the majority of the codebase was complete, OpenAI released their [code](https://github.com/openai/maddpg) for MADDPG, and I made some tweaks to this repo to reflect some of the details in their implementation.