import torch
import torch.nn.functional as F
from gym.spaces import Box, Discrete
from ..utils.networks import MLPNetwork
from ..utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax, enable_gradients, disable_gradients
from ..utils.agents import DDPGAgent

MSELoss = torch.nn.MSELoss()

class MADDPG(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, agent_init_params, alg_types, agent_types,
                 gamma=0.95, tau=0.01, lr=0.01, pol_hidden_dim=64,
                 critic_hidden_dim=64, discrete_action=False, share=False):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
            agent_types (list of str): List of whether each agent is a good agent
                                       or adversary
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
            share (bool): whether to share policies per "team"
        """
        self.nagents = len(alg_types)
        self.alg_types = alg_types
        self.agent_types = agent_types
        self.good_inds = [i for i, atype in enumerate(agent_types)
                           if atype == 'agent']
        self.adv_inds = [i for i, atype in enumerate(agent_types)
                         if atype == 'adversary']
        if share:
            self.learners = [DDPGAgent(lr=lr, discrete_action=discrete_action,
                                       pol_hidden_dim=pol_hidden_dim,
                                       critic_hidden_dim=critic_hidden_dim,
                                       **agent_init_params[self.good_inds[0]]),
                             DDPGAgent(lr=lr, discrete_action=discrete_action,
                                       pol_hidden_dim=pol_hidden_dim,
                                       critic_hidden_dim=critic_hidden_dim,
                                       **agent_init_params[self.adv_inds[0]])]
            # make different class for each agent that shares networks and
            # optimizers, but has a unique noise process
            self.agents = [DDPGAgent.share_params(self.learners[0])
                           if atype == 'agent'
                           else DDPGAgent.share_params(self.learners[1])
                           for atype in agent_types]
        else:
            self.learners = [DDPGAgent(lr=lr, discrete_action=discrete_action,
                                       pol_hidden_dim=pol_hidden_dim,
                                       critic_hidden_dim=critic_hidden_dim,
                                       **params)
                             for params in agent_init_params]
            self.agents = self.learners
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.discrete_action = discrete_action
        self.share = share
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def scale_noise(self, scale):
        """
        Scale noise for each agent
        Inputs:
            scale (float): scale of noise
        """
        for a in self.agents:
            a.scale_noise(scale)

    def reset_noise(self):
        for a in self.agents:
            a.reset_noise()

    def add_team_index(self, inp, i):
        """
        For use when "teams" share networks. Adds index of agent, within its
        team, to input
        Input:
            inp (PyTorch Variable): Input to net
            i (int): index of agent
        """
        if i in self.good_inds:
            team_i = self.good_inds.index(i)
            one_hot_dims = len(self.good_inds)
        else:
            team_i = self.adv_inds.index(i)
            one_hot_dims = len(self.adv_inds)
        one_hot = (torch.arange(one_hot_dims) == team_i).float()
        one_hot_stacked = torch.stack([one_hot] * inp.shape[0], dim=0)
        return inp, torch.autograd.Variable(one_hot_stacked, requires_grad=False)

    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        if self.share:
            # include index of agent in observation
            return [a.step(self.add_team_index(obs, i), explore=explore)
                    for i, a, obs in
                    zip(range(len(self.agents)), self.agents, observations)]
        return [a.step(obs, explore=explore) for a, obs in zip(self.agents,
                                                               observations)]

    def update(self, sample, agent_i, parallel=False, logger=None):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """
        obs, acs, rews, next_obs, dones = sample
        curr_agent = self.agents[agent_i]

        if self.alg_types[agent_i] == 'MADDPG':
            if self.share:
                trgt_pol_in = [self.add_team_index(nobs, i) for i, nobs
                               in enumerate(next_obs)]
            else:
                trgt_pol_in = next_obs
            if self.discrete_action: # one-hot encode action
                all_trgt_acs = [onehot_from_logits(pi(nobs)) for pi, nobs in
                                zip(self.target_policies, trgt_pol_in)]
            else:
                all_trgt_acs = [pi(nobs) for pi, nobs in zip(self.target_policies,
                                                             trgt_pol_in)]
            trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs), dim=1)
        else:  # DDPG
            if self.share:
                trgt_pol_in = self.add_team_index(next_obs[agent_i],
                                                   agent_i)
            else:
                trgt_pol_in = next_obs[agent_i]
            if self.discrete_action:
                trgt_vf_in = torch.cat((next_obs[agent_i],
                                        onehot_from_logits(
                                            curr_agent.target_policy(
                                                trgt_pol_in))),
                                       dim=1)
            else:
                trgt_vf_in = torch.cat((next_obs[agent_i],
                                        curr_agent.target_policy(trgt_pol_in)),
                                       dim=1)
        if self.share:
            trgt_vf_in = self.add_team_index(trgt_vf_in, agent_i)
        target_value = (rews[agent_i].view(-1, 1) + self.gamma *
                        curr_agent.target_critic(trgt_vf_in) *
                        (1 - dones[agent_i].view(-1, 1)))

        if self.alg_types[agent_i] == 'MADDPG':
            vf_in = torch.cat((*obs, *acs), dim=1)
        else:  # DDPG
            vf_in = torch.cat((obs[agent_i], acs[agent_i]), dim=1)
        if self.share:
            vf_in = self.add_team_index(vf_in, agent_i)
        actual_value = curr_agent.critic(vf_in)
        vf_loss = MSELoss(actual_value, target_value.detach())
        if self.share:
            # divide loss by number of agents on team, so that updating all
            # agents is equivalent to one update
            if agent_i in self.good_inds:
                scaled_vf_loss = vf_loss / len(self.good_inds)
            else:
                scaled_vf_loss = vf_loss / len(self.adv_inds)
            scaled_vf_loss.backward()
        else:
            vf_loss.backward()
        if parallel:
            # average gradients across workers
            average_gradients(curr_agent.critic)
        if (self.share and (agent_i == self.good_inds[-1] or
                            agent_i == self.adv_inds[-1])) or not self.share:
            # only update networks for last agent on team if sharing networks
            torch.nn.utils.clip_grad_norm(curr_agent.critic.parameters(), 0.5)
            curr_agent.critic_optimizer.step()
            curr_agent.critic_optimizer.zero_grad()

        if self.share:
            pol_in = [self.add_team_index(obs, i)
                      for i, obs in enumerate(obs)]
        else:
            pol_in = obs
        if self.discrete_action:
            # Forward pass as if onehot (hard=True) but backprop through a differentiable
            # Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop
            # through discrete categorical samples, but I'm not sure if that is
            # correct since it removes the assumption of a deterministic policy for
            # DDPG. Regardless, discrete policies don't seem to learn properly without it.
            curr_pol_out = curr_agent.policy(pol_in[agent_i])
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
        else:
            curr_pol_out = curr_agent.policy(pol_in[agent_i])
            curr_pol_vf_in = curr_pol_out
        if self.alg_types[agent_i] == 'MADDPG':
            all_pol_acs = []
            for i, pi, ob in zip(range(self.nagents), self.policies, pol_in):
                if i == agent_i:
                    all_pol_acs.append(curr_pol_vf_in)
                elif self.discrete_action:
                    all_pol_acs.append(onehot_from_logits(pi(ob)))
                else:
                    all_pol_acs.append(pi(ob))
            vf_in = torch.cat((*obs, *all_pol_acs), dim=1)
        else:  # DDPG
            vf_in = torch.cat((obs[agent_i], curr_pol_vf_in),
                              dim=1)
        if self.share:
            vf_in = self.add_team_index(vf_in, agent_i)
        pol_loss = -curr_agent.critic(vf_in).mean()
        pol_loss += (curr_pol_out**2).mean() * 1e-3  # policy regularization
        # don't want critic to accumulate gradients from policy loss
        disable_gradients(curr_agent.critic)
        if self.share:
            # divide loss by number of agents on team, so that updating all
            # agents is equivalent to one update
            if agent_i in self.good_inds:
                scaled_pol_loss = pol_loss / len(self.good_inds)
            else:
                scaled_pol_loss = pol_loss / len(self.adv_inds)
            scaled_pol_loss.backward()
        else:
            pol_loss.backward()
        enable_gradients(curr_agent.critic)
        if parallel:
            average_gradients(curr_agent.policy)
        if (self.share and (agent_i == self.good_inds[-1] or
                            agent_i == self.adv_inds[-1])) or not self.share:
            # only update networks for last agent on team if sharing networks
            torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), 0.5)
            curr_agent.policy_optimizer.step()
            curr_agent.policy_optimizer.zero_grad()

        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i,
                               {'vf_loss': vf_loss,
                                'pol_loss': pol_loss},
                               self.niter)

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for l in self.learners:
            soft_update(l.target_critic, l.critic, self.tau)
            soft_update(l.target_policy, l.policy, self.tau)
        self.niter += 1

    def prep_training(self, device='gpu'):
        for a in self.agents:
            a.policy.train()
            a.target_policy.train()
            # need bc sometimes we combine with other algs that don't have a separate critic per agent
            if hasattr(a, 'critic'):
                a.critic.train()
                a.target_critic.train()
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
        for a in self.agents:
            a.policy.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'learner_params': [l.get_params() for l in self.learners]}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, good_alg="MADDPG", adversary_alg="MADDPG",
                      gamma=0.95, tau=0.01, lr=0.01, pol_hidden_dim=64,
                      critic_hidden_dim=64, share=False, **kwargs):
        """
        Instantiate instance of this class from multi-agent environment

        env: Multi-agent Gym environment
        good_alg: algorithm to use for good agents
        adversary_alg: algorithm to use for adversaries
        gamma: discount factor
        tau: rate of update for target networks
        lr: learning rate for networks
        hidden_dim: number of hidden dimensions for networks
        share: whether to share policies per "team" (action and observation
               spaces must be the same for all sharing agents)
        """
        agent_init_params = []
        alg_types = [adversary_alg if atype == 'adversary' else good_alg for
                     atype in env.agent_types]
        num_good = sum(atype == 'agent' for atype in env.agent_types)
        num_adversaries = len(env.agent_types) - num_good
        for acsp, obsp, atype, algtype in zip(env.action_space,
                                              env.observation_space,
                                              env.agent_types,
                                              alg_types):
            num_in_pol = obsp.shape[0]
            if isinstance(acsp, Box):
                discrete_action = False
                get_shape = lambda x: x.shape[0]
            else:  # Discrete
                discrete_action = True
                get_shape = lambda x: x.n
            num_out_pol = get_shape(acsp)
            if algtype == "MADDPG":
                num_in_critic = 0
                for oobsp in env.observation_space:
                    num_in_critic += oobsp.shape[0]
                for oacsp in env.action_space:
                    num_in_critic += get_shape(oacsp)
            else:
                num_in_critic = obsp.shape[0] + get_shape(acsp)
            if share:
                onehot_dim = sum(atype == at for at in env.agent_types)
            else:
                onehot_dim = 0
            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_in_critic': num_in_critic,
                                      'onehot_dim': onehot_dim})
        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                     'pol_hidden_dim': pol_hidden_dim,
                     'critic_hidden_dim': critic_hidden_dim,
                     'alg_types': alg_types,
                     'agent_init_params': agent_init_params,
                     'discrete_action': discrete_action,
                     'share': share,
                     'agent_types': env.agent_types}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for l, params in zip(instance.learners, save_dict['learner_params']):
            l.load_params(params)
        if instance.share:
            instance.agents = [DDPGAgent.share_params(instance.learners[0])
                               if atype == 'agent'
                               else DDPGAgent.share_params(instance.learners[1])
                               for atype in instance.agent_types]
        else:
            instance.agents = instance.learners
        return instance