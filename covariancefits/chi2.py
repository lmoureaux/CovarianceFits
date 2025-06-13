import argparse
import numpy as np
import pickle


def chi2(dx, cov):
    """
    Calculates the chi2 between two distributions. The arguments are the
    bin-wise differences between the distributions and the sum of the covariance
    matrices.
    """

    return dx @ np.linalg.lstsq(cov, dx, rcond=None)[0]


def select_bins(dx, cov, first_bin=0, last_bin=-1):
    """
    Selects the bins to use. Returns dx and cov with only the needed values, and
    the used range().
    """

    last_bin = len(dx) + last_bin if last_bin < 0 else last_bin
    bins = range(first_bin, last_bin + 1)

    return dx[bins], cov[bins].T[bins].T, bins


def chi2tool():
    """
    Entry point to calculate the chi2 between two files for the same distribution.
    """

    parser = argparse.ArgumentParser(
        description="Computes a chi2 between two distributions.",
        epilog="""
            This program takes two files in Pickle .pkl format created by
            data2numpy or yoda2numpy and computes the chi2 between the two
            histograms.""",
    )
    parser.add_argument("input", nargs=2, help="Input files")
    parser.add_argument(
        "--first-bin",
        default=0,
        type=int,
        help="First bin index to consider (starting from 0)",
    )
    parser.add_argument(
        "--last-bin",
        default=-1,
        type=int,
        help="Last bin index to consider (starting from 0)",
    )
    parser.add_argument(
        "--naive",
        default=False,
        action="store_true",
        help="Perform a naive calculation without taking correlations into account",
    )
    args = parser.parse_args()

    with open(args.input[0], "rb") as stream:
        x1 = pickle.load(stream)
    with open(args.input[1], "rb") as stream:
        x2 = pickle.load(stream)

    assert (x1["bins"] == x2["bins"]).all(), "Binnings do not match"

    dx = x1["data"] - x2["data"]
    cov = np.sum(list(x1["covs"].values()) + list(x2["covs"].values()), axis=0)
    if args.naive:
        cov = np.diag(np.diag(cov))  # Remove off-diagonal elements

    print(f"Loaded histograms with {len(dx)} bins")
    print(f"Found {len(x1['covs'])} plus {len(x2['covs'])} covariance matrices")

    dx, cov, bins = select_bins(dx, cov, args.first_bin, args.last_bin)
    print()
    print(f"Calculating chi2 for bins {bins.start}:{bins.stop} ({len(dx)} bins)")

    c2 = chi2(dx, cov)

    print(f"chi2:      {c2:.2f}")
    print(f"ndof:      {len(dx)}")
    print(f"chi2/ndof: {c2 / len(dx):.2f}")
