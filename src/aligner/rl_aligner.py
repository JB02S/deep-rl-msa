import numpy as np

class RLAligner:
    def __init__(self, num_sequences, sequence_length):
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.values = np.zeros((num_sequences, sequence_length)) # Value Function

    def select_action(self):
        # Randomly select a sequence and a position to modify
        sequence_idx = np.random.randint(0, self.num_sequences)
        amino_acid_pos = np.random.randint(0, self.sequence_length)
        action = np.random.randint(0, 2)
        action = "l" if action == 1 else "r"
        return action, (sequence_idx, amino_acid_pos)

    def update_value_function(self, state, reward, next_state, alpha):
        # TD(0) update rule
        current_seq, current_pos = state
        next_seq, next_pos = next_state
        td_target = reward + self.values[next_seq, next_pos]
        td_error = td_target - self.values[current_seq, current_pos]
        self.values[current_seq, current_pos] += alpha * td_error
