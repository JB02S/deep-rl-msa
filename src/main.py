import argparse

import numpy as np
import torch
from torch.distributions import Categorical

from aligner.drl_aligner import DRLAligner
from aligner.rl_aligner import RLAligner
from trainer.trainer import PPOTrainer
from utils.one_hot_encoder import OneHotEncoder
from utils.file_reader import read_fasta
from environment.sequence_alignment_environment import SequenceAlignmentEnv


# from trainer.trainer import train_agent

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Multiple Sequence Alignment using Deep Reinforcement Learning")
    parser.add_argument('-p', '--path', type=str, required=True, help="Path to the directory containing FASTA files")
    parser.add_argument('-v', action='store_true', help="Print out aligned sequences after alignment")
    args = parser.parse_args()

    # Read input FASTA file
    sequences = read_fasta(args.path)
    if not sequences:
        print("No sequences found in the provided path.")
        return

    # Initialise the environment and agent
    env = SequenceAlignmentEnv(sequences)

    # These two lines just for testing purposes
    env.state[0][0] = "A"
    env.state[0][1] = "-"

    print(env.toString())
    print("Initial SP Score:")
    print(env.calculate_sp_score())
    # action = env.encode_action(0, 0, 1)
    # new_state, reward, done, info= env.step(action)
    # print(reward, '\n\n')
    # print("Final SP Score:")
    # print(env.calculate_sp_score())
    # print(env.toString())
    #
    # action = env.encode_action(0, 1, 0)
    # new_state, reward, done, info= env.step(action)
    # print(reward, '\n\n')
    # print("Final SP Score:")
    # print(env.calculate_sp_score())
    # print(env.toString())
    # exit(0)

    obs_space_size = np.prod(env.observation_space.shape)
    encoder = OneHotEncoder()

    encoded_obs = encoder.encode(env.state)
    aligner = DRLAligner(obs_space_size * 21, env.action_space.n)
    DEVICE = 'cpu'
    aligner = aligner.to(DEVICE)
    obs = env.reset()
    encoded_obs = encoder.encode(obs)
    encoded_obs = encoded_obs.flatten()
    input_state = torch.tensor([encoded_obs],
                               dtype=torch.float32,
                               device=DEVICE)

    logits, val = aligner(input_state)

    act_distribution = Categorical(logits=logits)
    act = act_distribution.sample()
    act_log_prob = act_distribution.log_prob(act).item()

    next_obs, reward, done, _ = env.step(act.item())

    obs = next_obs
    ep_reward = reward

    def calculate_gaes(rewards, values, gamma=0.99, decay=0.97):
        next_values = np.concatenate([values[1:], [0]])
        deltas = [rew + gamma * next_val - val for rew, val, next_val in zip(rewards, values, next_values)]

        gaes = [deltas[-1]]
        for i in reversed(range(len(deltas) - 1)):
            gaes.append(deltas[i] + decay * gamma * gaes[-1])

        return np.array(gaes[::-1])

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

    n_episodes = 500
    print_freq = 20
    ep_rewards = []

    ppo = PPOTrainer(
        aligner,
        policy_lr=3e-4,
        value_lr=1e-4,
        target_kl_div=0.02,
        max_policy_train_iters=40,
        value_train_iters=40
    )

    # for i in range(129):
    #     action = env.decode_action(i)
    #     if action[0] > 3 or action[0] < 0:
    #         print(f'ERROR, INVALID ACTION: {action}')
    #     elif action[1] > 15 or action[1] < 0:
    #         print(f'ERROR, INVALID ACTION: {action}')
    #     elif action[2] not in [0, 1]:
    #         print(f'ERROR, INVALID ACTION: {action}')
    # print("valid")
    # exit(0)

    def discount_rewards(rewards, gamma=0.99):

        new_rewards = [float(rewards[-1])]
        for i in reversed(range(len(rewards) - 1)):
            new_rewards.append(float(rewards[i]) + gamma * new_rewards[-1])
        return np.array(new_rewards[::-1])

    for episode_idx in range(n_episodes):

        # Perform rollout
        train_data, reward = rollout(aligner, env)
        ep_rewards.append(reward)

        # Shuffle
        permute_idxs = np.random.permutation(len(train_data[0]))

        # Policy data
        # print(train_data[0][0])
        # print('\n')
        # print(train_data[0][98])
        # print(train_data[0][0].tolist())
        # exit(0)

        encoded_obs_tran = [encoder.encode(train_data[0][i]).flatten() for i in range(len(train_data[0]))]
        obs = torch.tensor(np.array(encoded_obs_tran)[permute_idxs], dtype=torch.float32, device=DEVICE)
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

    # Print alignment result to stdout
    print("Final SP Scoret:")
    print(env.calculate_sp_score())

    # Print out alignment if -v
    if args.v:
        print(env.toString())


if __name__ == "__main__":
    main()
