import argparse
import sys
import numpy as np
import pickle
from scipy import stats


def chi2(dx, cov):
    """
    Calculates the chi2 between two distributions. The arguments are the
    bin-wise differences between the distributions and the sum of the covariance
    matrices.
    """

    return dx @ np.linalg.lstsq(cov, dx, rcond=None)[0]


def _get_bins(args, max_count: int) -> list[int]:
    """
    Returns the list of bins to use.
    """

    if args.bins is not None:
        if args.first_bin > 0 or args.last_bin >= 0:
            raise ValueError(
                "Cannot use --bins and --first-bin/--last-bin at the same time"
            )
        return args.bins

    last_bin = max_count + args.last_bin if args.last_bin < 0 else args.last_bin
    return list(range(args.first_bin, last_bin))


def total_covariance(covs):
    """
    Calculates the total covariance matrix.
    """

    return np.sum(list(covs.values()), axis=0)


def select_bins(x, bins):
    """
    Selects the bins to use in the input array (works with any number of
    dimensions).
    """

    for i in range(len(x.shape)):
        x = np.take(x, bins, axis=i)
    return x


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Adds chi2 configuration arguments to the parser.
    """

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
        "--bins",
        default=None,
        type=int,
        nargs="+",
        help="List of bin indices to consider",
    )
    parser.add_argument(
        "--shape-only",
        default=False,
        action="store_true",
        help="Normalizes the second argument to the sum of the first. This removes one degree of freedom.",
    )
    parser.add_argument(
        "--naive",
        default=False,
        action="store_true",
        help="Perform a naive calculation without taking correlations into account",
    )


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
    _add_common_arguments(parser)
    args = parser.parse_args()

    with open(args.input[0], "rb") as stream:
        x1 = pickle.load(stream)
    with open(args.input[1], "rb") as stream:
        x2 = pickle.load(stream)

    bins = _get_bins(args, len(x1["data"]))

    assert np.all(x1["bins"] == x2["bins"]), "Binnings do not match"

    print(f"Loaded histograms with {len(x1['data'])} bins")
    print(f"Found {len(x1['covs'])} plus {len(x2['covs'])} covariance matrices")
    print(f"Calculating chi2 for {len(bins)} bins ({", ".join(map(str, bins))})")

    x1, cov1 = select_bins(x1["data"], bins), select_bins(
        total_covariance(x1["covs"]), bins
    )
    x2, cov2 = select_bins(x2["data"], bins), select_bins(
        total_covariance(x2["covs"]), bins
    )
    ndof = len(x1)

    if args.shape_only:
        # This approach works surprisingly well despite not taking uncs into account.
        factor = np.sum(x1) / np.sum(x2)
        print(f"Scaling {args.input[1]} by {factor:.3f}.")
        x2 *= factor
        cov2 *= factor**2
        ndof -= 1

    dx = x1 - x2
    cov = cov1 + cov2
    if args.naive:
        cov = np.diag(np.diag(cov))  # Remove off-diagonal elements

    c2 = chi2(dx, cov)
    pval = 1 - stats.chi2.cdf(c2, ndof)

    print()
    print(f"chi2:      {c2:.2f}")
    print(f"ndof:      {ndof}")
    print(f"chi2/ndof: {c2 / ndof:.2f}")
    print(f"p-value:   {pval:.3f}")
