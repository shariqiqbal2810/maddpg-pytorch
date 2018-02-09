import torch
from utils.networks import MLPNetwork
from utils.misc import soft_update, average_gradients
from utils.agents import DDPGAgent

MSELoss = torch.nn.MSELoss()

class MADDPG(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, action_spaces, observation_spaces, agent_types,
                 gamma=0.95, tau=0.01, lr=0.01):
        """
        Inputs:
            action_spaces: Action spaces corresponding to each agent (from env)
            observation_spaces: Observation spaces corresponding to each agent.
            agent_types: Learning algorithm for each agent (DDPG or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
        """
        assert(len(action_spaces) == len(observation_spaces) == len(agent_types),
               "Inputs do not all correspond to the same number of agents")
        self.nagents = len(agent_types)
        self.agent_types = agent_types
        self.agents = []
        for acsp, obsp, atype, i in zip(action_spaces, observation_spaces,
                                        agent_types, range(self.nagents)):
            num_in_pol = obsp.shape[0]
            num_out_pol = acsp.shape[0]
            num_in_critic = obsp.shape[0]
            if atype == "MADDPG":
                for oacsp in action_spaces:
                    num_in_critic += oacsp.shape[0]
            else:
                num_in_critic += acsp.shape[0]
            self.agents.append(DDPGAgent(num_in_pol, num_out_pol, num_in_critic, lr=lr))

        self.gamma = gamma
        self.tau = tau
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def scale_noise(self, scale):
        """
        Scale Ornstein-Uhlenbeck noise for each agent
        Inputs:
            scale (float): scale of noise
        """
        for a in self.agents:
            a.exploration.scale = scale

    def reset_noise(self):
        for a in self.agents:
            a.exploration.reset()

    def step(self, observations, training=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            training (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        return [a.step(obs, training=training) for a, obs in zip(self.agents,
                                                                 observations)]

    def update(self, sample, agent_i, parallel=False):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
        """
        obs, acs, rews, next_obs, dones = sample
        curr_agent = self.agents[agent_i]

        curr_agent.critic_optimizer.zero_grad()
        if self.agent_types[agent_i] == 'MADDPG':
            all_trgt_acs = [pi(nobs) for pi, nobs in zip(self.target_policies,
                                                         next_obs)]
            trgt_vf_in = torch.cat((next_obs[agent_i], *all_trgt_acs), dim=1)
        else:  # DDPG
            trgt_vf_in = torch.cat((next_obs[agent_i],
                                    curr_agent.target_policy(next_obs[agent_i])),
                                   dim=1)
        target_value = (rews[agent_i].view(-1, 1) + self.gamma *
                        curr_agent.target_critic(trgt_vf_in))

        if self.agent_types[agent_i] == 'MADDPG':
            vf_in = torch.cat((obs[agent_i], *acs), dim=1)
        else:  # DDPG
            vf_in = torch.cat((obs[agent_i], acs[agent_i]), dim=1)
        actual_value = curr_agent.critic(vf_in)
        vf_loss = MSELoss(actual_value, target_value.detach())
        vf_loss.backward()
        if parallel:
            average_gradients(curr_agent.critic)
        curr_agent.critic_optimizer.step()

        curr_agent.policy_optimizer.zero_grad()
        if self.agent_types[agent_i] == 'MADDPG':
            all_pol_acs = [pi(ob) for pi, ob in zip(self.policies, obs)]
            vf_in = torch.cat((obs[agent_i], *all_pol_acs), dim=1)
        else:  # DDPG
            vf_in = torch.cat((obs[agent_i], curr_agent.policy(obs[agent_i])),
                              dim=1)
        pol_loss = -curr_agent.critic(vf_in).mean()
        pol_loss.backward()
        if parallel:
            average_gradients(curr_agent.policy)
        curr_agent.policy_optimizer.step()

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_policy, a.policy, self.tau)

    def prep_training(self, device='gpu'):
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                a.critic = fn(a.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                a.target_critic = fn(a.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device

    def save(self):
        """
        Save trained parameters of all agents into one file
        """
        pass

    @classmethod
    def load(cls, filename):
        """
        Instantiate instance of this class from saved file
        """
        pass