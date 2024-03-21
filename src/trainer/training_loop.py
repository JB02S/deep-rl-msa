import numpy as np
import torch
from torch.distributions import Categorical


def train(n_episodes, ep_rewards, agent, env, DEVICE, encoder, print_freq, ppo):
    for episode_idx in range(n_episodes):

        # Perform rollout
        train_data, reward = rollout(model=agent,
                                     env=env,
                                     encoder=encoder,
                                     DEVICE=DEVICE)
        ep_rewards.append(reward)

        # Shuffle
        permute_idxs = np.random.permutation(len(train_data[0]))

        encoded_obs_train = [encoder.encode(train_data[0][i]).flatten() for i in range(len(train_data[0]))]
        obs = torch.tensor(np.array(encoded_obs_train)[permute_idxs], dtype=torch.float32, device=DEVICE)
        acts = torch.tensor(train_data[1][permute_idxs], dtype=torch.int32, device=DEVICE)
        gaes = torch.tensor(train_data[3][permute_idxs], dtype=torch.int32, device=DEVICE)
        act_log_probs = torch.tensor(train_data[4][permute_idxs], dtype=torch.int32, device=DEVICE)

        # Value data
        returns = discount_rewards(train_data[2])[permute_idxs]
        returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)

        # Train model
        ppo.train_policy(obs, acts, act_log_probs, gaes)
        ppo.train_value(obs, returns)

        if (episode_idx + 1) % print_freq == 0:
            print('Episode {} | Avg Reward {:.1f}'.format(
                episode_idx + 1, np.mean(ep_rewards[-print_freq:])
            ))

def rollout(model, env, encoder, DEVICE, max_steps=1000):
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

def calculate_gaes(rewards, values, gamma=0.99, decay=0.97):
    next_values = np.concatenate([values[1:], [0]])
    deltas = [rew + gamma * next_val - val for rew, val, next_val in zip(rewards, values, next_values)]

    gaes = [deltas[-1]]
    for i in reversed(range(len(deltas) - 1)):
        gaes.append(deltas[i] + decay * gamma * gaes[-1])

    return np.array(gaes[::-1])

def discount_rewards(rewards, gamma=0.99):

    new_rewards = [float(rewards[-1])]
    for i in reversed(range(len(rewards) - 1)):
        new_rewards.append(float(rewards[i]) + gamma * new_rewards[-1])
    return np.array(new_rewards[::-1])