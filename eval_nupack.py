"""
A sample script to demonstrate calling nupack oracle
"""
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from oracles import nupackScore
from oracle import numbers2letters


def add_args(parser):
    """
    Adds command-line arguments to parser

    Returns:
        argparse.Namespace: the parsed arguments
    """
    args2config = {}
    parser.add_argument(
        "--data_npy",
        default=None,
        type=str,
        help="Path to npy file containing DNA aptamer sequences as zero padded ndarray",
    )
    parser.add_argument(
        "--data_csv",
        default=None,
        type=str,
        help="Path to CSV file containing DNA aptamer sequences as ATCG",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        type=str,
        help="Output CSV file to store DNA aptamer sequences as ATCG",
    )
    return parser


ALPHABET_ORIG = {0: "A", 1: "T", 2: "C", 3: "G"}
ALPHABET_INV_ORACLE = {v: k + 1 for k, v in ALPHABET_ORIG.items()}


def main(args):
    # Read data CSV
    if args.data_csv:
        df = pd.read_csv(args.data_csv, index_col=0)
        seqs_letters = df["letters"].values
        n_seqs = len(seqs_letters)
        horizon = np.max([len(seq) for seq in seqs_letters])
        scores_orig = df["scores"].values
        seqs_idx = np.zeros((n_seqs, horizon), dtype=int)
        for idx, seq in enumerate(seqs_letters):
            seqs_idx[idx, : len(seq)] = [ALPHABET_INV_ORACLE[el] for el in seq]
    # Read data NPY
    elif args.data_npy:
        data_dict = np.load(args.data_npy, allow_pickle=True).item()
        seqs_idx = data_dict["samples"]
        scores_orig = data_dict["scores"]
        if args.output_csv:
            letters = numbers2letters(seqs_idx)
            df = pd.DataFrame({"letters": letters, "scores": scores_orig})
            df.to_csv(args.output_csv)
    else:
        raise ValueError("No data file provided")
    # Compute nupack energies by calling the oracle
    scores_new = nupackScore(seqs_idx, returnFunc="energy")
    assert all(np.isclose(scores_orig, scores_new))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    main(args)
