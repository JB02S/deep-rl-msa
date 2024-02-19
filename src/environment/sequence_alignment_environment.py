import random


class SequenceAlignmentEnvironment():
    def __init__(self, sequences):
        self.sequences = self.pad_sequences(sequences)
        print(self.sequences)

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
