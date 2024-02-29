import random


class SequenceAlignmentEnvironment:
    def __init__(self, sequences):
        # Keeping original for reset function during training
        self.original_sequences = sequences
        self.sequences = self.pad_sequences(sequences)
        self.possible_actions = ["l", "r"]

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
            return ''.join(padded_sequence)

        return [pad_sequence(seq) for seq in sequences]

    def get_actions(self):
        return self.possible_actions

    def calculate_sp_score(self):
        match_score = 1
        mismatch_score = -1
        gap_penalty = -2
        sp_score = 0

        # Iterate over each column
        for col in range(len(self.sequences[0])):
            # Iterate over each pair of sequences in the MSA
            for i in range(len(self.sequences)):
                for j in range(i + 1, len(self.sequences)):
                    char1 = self.sequences[i][col]
                    char2 = self.sequences[j][col]

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

    def step(self, action, pos):
        """

        Apply an action to a specific position in the sequences.

        Args:
            action: The action to be performed
            pos: a tuple (row, col) indicating where the action should be performed

        Returns:
            new_state: The new state of the environment after the action.
            reward: The reward received for taking the action.
            done: A boolean indicating if the episode has ended.

        """
        print(action, pos)
        initial_sp = self.calculate_sp_score()
        sequence_index, amino_acid_position = pos
        sequence = list(self.sequences[sequence_index])
        target = sequence[amino_acid_position]

        if target == "-":
            return self.sequences, -0.5, False  # Can't move a dash

        if action == "l":
            new_position = amino_acid_position - 1
        elif action == "r":
            new_position = amino_acid_position + 1
        else:
            return self.sequences, -0.5, False  # Invalid action

        # Update position if it is possible to make the move (there is empty space)
        if 0 <= new_position < len(sequence) and sequence[new_position] == "-":
            sequence[new_position], sequence[amino_acid_position] = sequence[amino_acid_position], sequence[new_position]
            self.sequences[sequence_index] = ''.join(sequence)
            return self.sequences, self.calculate_sp_score(), True
        else:
            return self.sequences, -0.5, False

    def toString(self):
        env = ""
        for sequence in self.sequences:
            env += sequence + "\n"
        return env

    def reset(self):
        self.sequences = self.original_sequences


