from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from .networks import MLPNetwork
from .misc import hard_update, gumbel_softmax, onehot_from_logits
from .noise import OUNoise
import numpy as np

class DDPGAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, hidden_dim=64,
                 lr=0.01, discrete_action=True, K = 1):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.K = K
        self.policy = []
        self.target_policy = []
        self.policy_optimizer = []
        for k in range(K):
            self.policy.append(MLPNetwork(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim,
                                 constrain_out=True,
                                 discrete_action=discrete_action))
            self.target_policy.append(MLPNetwork(num_in_pol, num_out_pol,
                                            hidden_dim=hidden_dim,
                                            constrain_out=True,
                                            discrete_action=discrete_action))
            self.policy_optimizer.append(Adam(self.policy.parameters(), lr=lr))
            hard_update(self.target_policy[k], self.policy[k])

        self.critic = MLPNetwork(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False)
        self.target_critic = MLPNetwork(num_in_critic, 1,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False)
        hard_update(self.target_critic, self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        if not discrete_action:
            self.exploration = OUNoise(num_out_pol)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, explore=False, k = 0):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        action = self.policy[k](obs)
        if self.discrete_action:
            if explore:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)
        else:  # continuous action
            if explore:
                action += Variable(Tensor(self.exploration.noise()),
                                   requires_grad=False)
            action = action.clamp(-1, 1)
        return action

    def get_params(self):
        param_dict = {}
        for k in range(self.K):
            param_dict['policy' + str(k)] = self.policy[k].state_dict()
            param_dict['target_policy' + str(k)] = self.target_policy[k].state_dict()
            param_dict['policy_optimizer' + str(k)] = self.policy_optimizer[k].state_dict()

        param_dict['critic'] = self.critic.state_dict()
        param_dict['target_critic'] = self.target_critic.state_dict()
        param_dict['critic_optimizer'] = self.critic_optimizer.state_dict()
        return param_dict
    def load_params(self, params):
        for k in range(self.K):
            self.policy.load_state_dict(params['policy' + str(k)])
            self.target_policy.load_state_dict(params['target_policy' + str(k)])
            self.policy_optimizer.load_state_dict(params['policy_optimizer' + str(k)])
        self.critic.load_state_dict(params['critic'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])
