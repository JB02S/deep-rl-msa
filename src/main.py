import argparse
from aligner.drl_aligner import DRLAligner
from utils.file_reader import read_fasta

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Multiple Sequence Alignment using Deep Reinforcement Learning")
    parser.add_argument('-p', '--path', type=str, required=True, help="Path to the directory containing FASTA files")
    args = parser.parse_args()

    # Read input FASTA file
    sequences = read_fasta(args.path)
    if not sequences:
        print("No sequences found in the provided path.")
        return

    # Initialize the DRL-based aligner
    aligner = DRLAligner()

    # Perform alignment
    alignment_result = aligner.align(sequences)

    # Optionally, you can add code here to save the alignment_result to a file or process it further

    # Print alignment result to stdout or process further as needed
    print("Alignment Result:")
    print(alignment_result)

if __name__ == "__main__":
    main()