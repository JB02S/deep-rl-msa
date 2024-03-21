def read_fasta(filename):

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
