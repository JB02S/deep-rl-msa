# deep-rl-msa
Using deep reinforcement learning for multiple sequence alignment of proteins

## File formatting and reading in data
The input file is given in a fasta format and must contain a > to differentiate between 
amino acid sequences. Files are read in through the read_fasta function in src/utils/file_reader.py
The sequences that are read in are stored in a 2d array where arr[i][j] would correspond to amino acid j
of sequence i.

## Defining the environment

### State
The initial state is created when the object is instantiated. The SequenceAlignmentEnv class is contained in sequence_alignment_environment.py
The constructor takes the 2d array as an argument and stores this as the initial state, it also performs "padding" which adds
random dashes into each sequence. The number of these dashes is the length of the largest sequence + 10, these dashes represent
empty spaces which amino acids may be moved into. 

