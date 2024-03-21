import random
import gymnasium as gym
import numpy as np


class SequenceAlignmentEnv(gym.Env):
    def __init__(self, sequences):

        # storing original sequences for reset function
        self.original_sequences = self.pad_sequences(sequences)
        self.state = self.original_sequences

        self.n_sequences = len(sequences)
        self.max_sequence_length = max(len(seq) for seq in self.state)

        # One action for each board position, multiplied by 2 for left or right
        self.action_space = gym.spaces.Discrete(self.n_sequences * self.max_sequence_length * 2)

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.n_sequences, self.max_sequence_length),
                                                dtype=np.float32)

        # Number of times the agent can make actions before the alignment is finished
        self.numIters = 100

    def step(self, action):
        seq_id, pos_id, direction = self.decode_action(action)

        # Penalizing invalid actions and if action is valid perform the action

        if seq_id >= len(self.state) or seq_id < 0:
            reward = -1.0
        elif pos_id >= len(self.state[seq_id]) or pos_id < 0:
            reward = -1.0
        elif self.state[seq_id][pos_id] == "-":
            reward = -1.0
        elif direction == 0:
            if pos_id == 0:
                reward = -1.0
            elif self.state[seq_id][pos_id - 1] != "-":
                reward = -1.0
            else:
                reward = self.move_aa(seq_id, pos_id, direction)
        elif direction == 1:
            if pos_id == (len(self.state[seq_id]) - 1):
                reward = -1.0
            elif self.state[seq_id][pos_id + 1] != "-":
                reward = -1.0
            else:
                reward = self.move_aa(seq_id, pos_id, direction)
        else:
            reward = -1.0

        self.numIters -= 1
        info = {}
        # print(self.decode_action())
        return self.state, reward, self.numIters <= 0, info

    def render(self):
        pass

    def toString(self):
        env = ""
        for sequence in self.state:
            env += ''.join(sequence) + "\n"
        return env

    def reset(self):
        self.state = self.original_sequences
        return self.state

    def move_aa(self, seq_id, pos_id, direction):

        old_sp_score = self.calculate_sp_score()

        if direction == 0:
            self.state[seq_id][pos_id - 1] = self.state[seq_id][pos_id]
            self.state[seq_id][pos_id] = "-"

        elif direction == 1:
            self.state[seq_id][pos_id + 1] = self.state[seq_id][pos_id]
            self.state[seq_id][pos_id] = "-"

        return self.normalize_score(self.calculate_sp_score() - old_sp_score)

    def encode_action(self, seq_id, pos_id, direction):
        action_id = seq_id * (self.max_sequence_length * 2) + pos_id * 2 + direction
        return action_id

    def decode_action(self, action_id):
        seq_id = action_id // (self.max_sequence_length * 2)
        pos_and_dir = action_id % (self.max_sequence_length * 2)
        pos_id = pos_and_dir // 2
        direction = pos_and_dir % 2
        return seq_id, pos_id, direction

    def pad_sequences(self, sequences):
        max_length = max(len(seq) for seq in sequences) + 10

        # Function to pad a single sequence
        def pad_sequence(seq):
            padding_length = max_length - len(seq)
            padded_sequence = list(seq)
            for _ in range(padding_length):
                # Choose a random position to insert a dash
                pos = random.randint(0, len(padded_sequence))
                padded_sequence.insert(pos, "-")
            return padded_sequence

        return [pad_sequence(seq) for seq in sequences]

    def calculate_sp_score(self):
        match_score = 1
        mismatch_score = -1
        gap_penalty = -2
        sp_score = 0

        # Iterate over each column
        for col in range(len(self.state[0])):
            # Iterate over each pair of sequences in the MSA
            for i in range(len(self.state)):
                for j in range(i + 1, len(self.state)):
                    char1 = self.state[i][col]
                    char2 = self.state[j][col]

                    # Calculate the score for this pair
                    if char1 == char2:
                        if char1 != '-':  # Both are matching nucleotides
                            sp_score += match_score
                        # If both are gaps, do not add any score (skip)
                    else:
                        if char1 == '-' or char2 == '-':  # One is a gap
                            sp_score += gap_penalty
                        else:  # Mismatch
                            sp_score += mismatch_score

        return sp_score

    def normalize_score(self, score):

        min_score = (self.max_sequence_length * ((self.n_sequences * (self.n_sequences - 1)) / 2)) * -1
        max_score = (self.max_sequence_length * ((self.n_sequences * (self.n_sequences - 1)) / 2)) * 1
        normalized_score = (2 * ((score - min_score) / (max_score - min_score))) - 1
        return normalized_score
