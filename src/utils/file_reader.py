def read_fasta(filename):
    """
    Reads a FASTA file and returns a dictionary where the keys are the sequence
    identifiers and the values are the sequences.

    Parameters:
    - filename: Path to the FASTA file.

    Returns:
    - A dictionary with sequence identifiers as keys and sequences as values.
    """
    sequences = {}
    with open(filename, 'r') as file:
        sequence_id = None
        sequence = []
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if sequence_id is not None:
                    sequences[sequence_id] = ''.join(sequence)
                sequence_id = line[1:].split()[0]  # Get the part of the identifier before the first space
                sequence = []
            else:
                sequence.append(line)
        if sequence_id is not None:
            sequences[sequence_id] = ''.join(sequence)  # Add the last read sequence
    return sequences
