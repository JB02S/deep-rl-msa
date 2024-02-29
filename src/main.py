import argparse
from aligner.drl_aligner import DRLAligner
from aligner.rl_aligner import RLAligner
from utils.file_reader import read_fasta
from environment.sequence_alignment_environment import SequenceAlignmentEnvironment
from trainer.trainer import train_agent

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Multiple Sequence Alignment using Deep Reinforcement Learning")
    parser.add_argument('-p', '--path', type=str, required=True, help="Path to the directory containing FASTA files")
    parser.add_argument('-v', action='store_true', help="Print out aligned sequences after alignment")
    args = parser.parse_args()

    # Read input FASTA file
    sequences = read_fasta(args.path)
    if not sequences:
        print("No sequences found in the provided path.")
        return

    # Initialise the environment and agent
    env = SequenceAlignmentEnvironment(sequences)
    aligner = RLAligner(len(env.sequences[0]), len(env.sequences))

    # Train agent
    train_agent(aligner, env)

    # Perform alignment
    alignment_result = 0

    # Print out alignment if -v
    if args.v:
        print(env.toString())



    # Print alignment result to stdout
    print("Alignment Result:")
    print(SequenceAlignmentEnvironment.calculate_sp_score(env))

if __name__ == "__main__":
    main()