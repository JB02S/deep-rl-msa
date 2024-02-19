def read_fasta(filename):
    """

    Reads a FASTA file and returns a dictionary where the keys are the sequence
    identifiers and the values are the sequences.

    Args:
        filename: Path to the FASTA file.

    Returns:
        A list of the sequences in the FASTA file.

    """
    sequences = []
    with open(filename, 'r') as file:
        sequence = []
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if sequence:
                    sequences.append(''.join(sequence))
                    sequence = []
            else:
                sequence.append(line)
        if sequence:
            sequences.append(''.join(sequence))  # Add the last read sequence
    return sequences
