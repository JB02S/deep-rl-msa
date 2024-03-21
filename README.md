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

### Actor critic model

The agent utilises an actor critic model to decide the action to take

Actor

The Actor is responsible for choosing actions. Based on the current state of the environment, it decides what action to take next. The goal of the Actor is to learn a policy that maps states to the best possible actions to maximize rewards over time.
How it learns: The Actor adjusts its policy based on feedback from the Critic. Essentially, it learns to take better actions over time by understanding which actions lead to higher rewards.

Critic

The Critic evaluates the actions taken by the Actor. It looks at the current state and the action taken by the Actor, and then it estimates the potential future rewards (or value) that result from that action.
The Critic learns by comparing its predictions with the actual rewards received and then adjusting its estimates to be more accurate in the future. The difference between the predicted rewards and the actual rewards received is often referred to as the "temporal difference error" or TD error.

How they work together

The Actor takes actions in the environment based on its current policy.
The Critic assesses these actions by estimating how good the action is — that is, the expected future rewards.
The Critic then provides feedback to the Actor, indicating how off its predictions were compared to the actual outcome.
The Actor uses this feedback to update its policy, aiming to take actions that lead to higher rewards in the future, based on the Critic's evaluations.

### Proximal policy optimization
The decision to use PPO here was to avoid large policy updates since PPO limits (clips) the amount by which the policy can change in a single training step

### How the agent acts on the environment
As the state is represented as a 2d array of characters and the actor and critic networks need numerical values, the state is first one hot encoded (there are
21 possible values for each amino acid) and then passed into the network. The agent then returns one of the possible actions to take, the action is an encoded integer value
which represents a sequence, amino acid, and direction to move it.