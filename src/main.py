import argparse


import numpy as np
import torch

from torch.distributions import Categorical


from aligner.drl_aligner import DRLAligner
from trainer.ppo import PPO
from utils.one_hot_encoder import OneHotEncoder
from utils.file_reader import read_fasta
from environment.sequence_alignment_environment import SequenceAlignmentEnv
from trainer.training_loop import train


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

    print(env.toString())
    print("Initial SP Score:")
    print(env.calculate_sp_score())

    obs_space_size = np.prod(env.observation_space.shape)
    encoder = OneHotEncoder()
    aligner = DRLAligner(obs_space_size * 21, env.action_space.n)
    DEVICE = 'cpu'
    aligner = aligner.to(DEVICE)

    n_episodes = 100
    print_freq = 20
    ep_rewards = []

    ppo = PPO(
        aligner,
        policy_lr=3e-4,
        value_lr=1e-4,
        target_kl_div=0.02,
        max_policy_train_iters=40,
        value_train_iters=40
    )

    train(n_episodes=n_episodes,
          ep_rewards=ep_rewards,
          agent=aligner,
          env=env,
          DEVICE=DEVICE,
          encoder=encoder,
          print_freq=print_freq,
          ppo=ppo)

    # Print alignment result to stdout
    print("Final SP Scoret:")
    print(env.calculate_sp_score())

    # Print out alignment if -v
    if args.v:
        print(env.toString())


if __name__ == "__main__":
    main()
