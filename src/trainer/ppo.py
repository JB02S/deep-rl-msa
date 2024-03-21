import numpy as np
import torch
from torch.distributions import Categorical


class PPOTrainer:
    def __init__(self,
                 model,
                 ppo_clip_val=0.2,
                 target_kl_div=0.01,
                 max_policy_train_iters=80,
                 value_train_iters=80,
                 policy_lr=3e-4,
                 value_lr=3e-4):

        self.model = model
        self.ppo_clip_val = ppo_clip_val
        self.target_kl_div = target_kl_div
        self.max_policy_train_iters = max_policy_train_iters
        self.value_train_iters = value_train_iters

        policy_params = list(self.model.shared_layers.parameters()) + list(self.model.policy_layers.parameters())

        self.policy_optim = torch.optim.Adam(policy_params, lr=policy_lr)

        value_params = list(self.model.shared_layers.parameters()) + list(self.model.value_layers.parameters())

        self.value_optim = torch.optim.Adam(value_params, lr=value_lr)

    def train_policy(self, obs, acts, old_log_probs, gaes):

        for _ in range(self.max_policy_train_iters):

            self.policy_optim.zero_grad()
            new_logits = self.model.policy(obs)
            new_logits = Categorical(logits=new_logits)
            new_log_probs = new_logits.log_prob(acts)

            policy_ratio = torch.exp(new_log_probs - old_log_probs)

            clipped_ratio = policy_ratio.clamp(
                1 - self.ppo_clip_val, 1 + self.ppo_clip_val
            )

            clipped_loss = clipped_ratio * gaes
            full_loss = policy_ratio * gaes
            policy_loss = -torch.min(full_loss, clipped_loss).mean()

            policy_loss.backward()
            self.policy_optim.step()

            kl_div = (old_log_probs - new_log_probs).mean()

            if kl_div > self.target_kl_div:
                break

    def train_value(self, obs, returns):

        for _ in range(self.value_train_iters):

            self.value_optim.zero_grad()
            values = self.model.value(obs)
            value_loss = (returns - values) ** 2
            value_loss = value_loss.mean()

            value_loss.backward()
            self.value_optim.step()

def rollout(model, env, max_steps=1000):

    train_data = [[], [], [], [], []]  # obs, act, reward, values, act_log_probs
    obs = env.reset()

    encoded_obs = encoder.encode(obs)
    encoded_obs = encoded_obs.flatten()
    input_state = torch.tensor([encoded_obs],
                               dtype=torch.float32,
                               device=DEVICE)

    ep_reward = 0
    for _ in range(max_steps):
        logits, val = model(input_state)

        act_distribution = Categorical(logits=logits)
        act = act_distribution.sample()
        act_log_prob = act_distribution.log_prob(act).item()

        act, val = act.item(), val.item()

        next_obs, reward, done, _ = env.step(act)

        for i, item in enumerate((obs, act, reward, val, act_log_prob)):
            train_data[i].append(item)

        obs = next_obs
        ep_reward = reward
        if done:
            break

        encoded_obs = encoder.encode(obs)
        encoded_obs = encoded_obs.flatten()
        input_state = torch.tensor([encoded_obs], dtype=torch.float32, device=DEVICE)

    train_data = [np.asarray(x) for x in train_data]

    # Do train data filtering
    train_data[3] = calculate_gaes(train_data[2], train_data[3])

    return train_data, ep_reward