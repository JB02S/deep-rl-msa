# deep-rl-msa
Using deep reinforcement learning for multiple sequence alignment of proteins

## Agent-environment model

Currently using a finite MDP to model the problem.

Sequences are read in and each sequence is padded so that it is the length of the largest sequence + 10,
the extra spaces are filled with dashes in random positions