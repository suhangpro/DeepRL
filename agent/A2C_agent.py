#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
from network import *
from utils import *
from component import *
from .BaseAgent import *
import pickle
import os
import time

class A2CAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.policy = config.policy_fn()
        self.total_steps = 0
        self.states = self.task.reset()
        self.episode_rewards = np.zeros(config.num_workers)
        self.last_episode_rewards = np.zeros(config.num_workers)

        if hasattr(self.network, 'is_rnn') and self.network.is_rnn:
            self.network.rnn_reset_state(np.ones(config.num_workers, dtype=np.int))

    def iteration(self):
        config = self.config
        rollout = []
        states = self.states
        for _ in range(config.rollout_length):
            prob, log_prob, value = self.network.predict(config.state_normalizer(states))
            actions = [self.policy.sample(p) for p in prob.cpu().detach().numpy()]
            next_states, rewards, terminals, _ = self.task.step(actions)
            if hasattr(self.network, 'is_rnn') and self.network.is_rnn:
                self.network.rnn_reset_state(terminals)
            self.episode_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.last_episode_rewards[i] = self.episode_rewards[i]
                    self.episode_rewards[i] = 0

            rollout.append([prob, log_prob, value, actions, rewards, 1 - terminals])
            states = next_states

        self.states = states
        _, _, pending_value = self.network.predict(config.state_normalizer(states))
        rollout.append([None, None, pending_value, None, None, None])

        processed_rollout = [None] * (len(rollout) - 1)
        advantages = self.network.tensor(np.zeros((config.num_workers, 1)))
        returns = pending_value.detach()
        for i in reversed(range(len(rollout) - 1)):
            prob, log_prob, value, actions, rewards, terminals = rollout[i]
            terminals = self.network.tensor(terminals).unsqueeze(1)
            rewards = self.network.tensor(rewards).unsqueeze(1)
            actions = self.network.tensor(actions).unsqueeze(1).long()
            next_value = rollout[i + 1][2]
            returns = rewards + config.discount * terminals * returns
            if not config.use_gae:
                advantages = returns - value.detach()
            else:
                td_error = rewards + config.discount * terminals * next_value.detach() - value.detach()
                advantages = advantages * config.gae_tau * config.discount * terminals + td_error
            processed_rollout[i] = [prob, log_prob, value, actions, returns, advantages]

        prob, log_prob, value, actions, returns, advantages = map(lambda x: torch.cat(x, dim=0), zip(*processed_rollout))
        policy_loss = -log_prob.gather(1, actions) * advantages
        entropy_loss = torch.sum(prob * log_prob, dim=1, keepdim=True)
        value_loss = 0.5 * (returns - value).pow(2)

        self.policy_loss = np.mean(policy_loss.cpu().detach().numpy())
        self.entropy_loss = np.mean(entropy_loss.cpu().detach().numpy())
        self.value_loss = np.mean(value_loss.cpu().detach().numpy())

        self.optimizer.zero_grad()
        (policy_loss + config.entropy_weight * entropy_loss +
         config.value_loss_weight * value_loss).mean().backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()

        if hasattr(self.network, 'is_rnn') and self.network.is_rnn:
            self.network.rnn_detach_state()

        self.evaluate(config.rollout_length)

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps
