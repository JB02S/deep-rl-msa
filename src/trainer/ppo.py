import torch
from torch.distributions import Categorical


class PPO:
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
