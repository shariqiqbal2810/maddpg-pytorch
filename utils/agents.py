from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from .networks import MLPNetwork
from .misc import hard_update
from .noise import OUNoise

class DDPGAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, lr=0.01):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.policy = MLPNetwork(num_in_pol, num_out_pol, constrain_out=True)
        self.critic = MLPNetwork(num_in_critic, 1, constrain_out=False)
        self.target_policy = MLPNetwork(num_in_pol, num_out_pol, constrain_out=True)
        self.target_critic = MLPNetwork(num_in_critic, 1, constrain_out=False)
        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        self.exploration = OUNoise(num_out_pol)

    def step(self, obs, training=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            training (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        action = self.policy(obs)
        if training:
            action += Variable(Tensor(self.exploration.noise()))
        return action.clamp(-1, 1)

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])
