# deep-rl-msa
Using deep reinforcement learning for multiple sequence alignment of proteins

## Usage
Need to install required packages first using pip install -r requirements.txt
 

# Brief overview

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

### Action Space
Can move each amino acid left or right, however these moves are only legal if there is an empty space it can move into.
If the agent tries any of the following illegal moves it receives a negative reward:

- Tries to move a dash
- Ties to move amino acid into non-empty space
- Tries to move amino acid outside of border

For any other legal move the agent receives a reward equal to the old similarity pairing score subtracted from the new similarity pairing score, the result
is also normalized to a range [-1, 1] where -1 represents the worst possible SP score for the sequences and 1 represents the best possible SP score.

### State Space
The state space every possible state the sequence alignment could be in, e.g everytime the position of an amino acid changes

## Defining the agent


