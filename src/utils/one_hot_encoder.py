import numpy as np

class OneHotEncoder:
    def __init__(self):
        # Mapping for amino acids and gap character to integers
        self.amino_acids = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
                            'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '-']
        self.char_to_int = {char: idx for idx, char in enumerate(self.amino_acids)}
        self.num_classes = len(self.amino_acids)

    def encode(self, sequences):
        num_sequences = len(sequences)
        sequence_length = len(sequences[0])
        encoded = np.zeros((num_sequences, sequence_length, self.num_classes), dtype=np.float32)

        for i, seq in enumerate(sequences):
            for j, char in enumerate(seq):
                char_idx = self.char_to_int[char]
                encoded[i, j, char_idx] = 1.0

        return encoded
