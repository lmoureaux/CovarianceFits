import argparse
import pickle
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
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
        "--k-factor",
        default=None,
        type=float,
        help="Applies a k-factor to the second argument. This does not affect the number of degrees of freedom.",
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


@dataclass
class Chi2Result:
    chi2: float = 0.0
    ndof: int = 0
    pval: float = np.nan
    dx: Optional[np.ndarray] = None
    cov: Optional[np.ndarray] = None
    bins: Optional[list[int]] = None
    k_factor: float = 1


def _chi2_with_args(x1, x2, args) -> Chi2Result:
    """
    Compute the chi2 between x1 and x2, taking command-line arguments from
    _add_common_arguments() into account.
    """

    result = Chi2Result()
    result.bins = _get_bins(args, len(x1["data"]))

    assert np.all(x1["bins"] == x2["bins"]), "Binnings do not match"

    x1, cov1 = select_bins(x1["data"], result.bins), select_bins(
        total_covariance(x1["covs"]), result.bins
    )
    x2, cov2 = select_bins(x2["data"], result.bins), select_bins(
        total_covariance(x2["covs"]), result.bins
    )
    result.ndof = len(x1)

    if args.k_factor:
        result.k_factor = args.k_factor
    elif args.shape_only:
        # This approach works surprisingly well despite not taking uncs into account.
        result.k_factor = np.sum(x1) / np.sum(x2)
        result.ndof -= 1

    x2 *= result.k_factor
    cov2 *= result.k_factor

    result.dx = x1 - x2
    result.cov = cov1 + cov2
    if args.naive:
        result.cov = np.diag(np.diag(result.cov))  # Remove off-diagonal elements

    result.chi2 = chi2(result.dx, result.cov)
    result.pval = stats.chi2.sf(result.chi2, result.ndof)

    return result


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

    print(f"Loaded histograms with {len(x1['data'])} bins")
    print(f"Found {len(x1['covs'])} plus {len(x2['covs'])} covariance matrices")

    result = _chi2_with_args(x1, x2, args)
    bins = ", ".join(map(str, result.bins))
    print(f"Calculating chi2 for {len(result.bins)} bins ({bins})")

    if args.shape_only:
        print(f"Scaling {args.input[1]} by {result.k_factor:.3f}.")

    print()
    print(f"chi2:      {result.chi2:.2f}")
    print(f"ndof:      {result.ndof}")
    print(f"chi2/ndof: {result.chi2 / result.ndof:.2f}")
    print(f"p-value:   {result.pval:.3f}")


def chi2scan():
    """
    Helper for chi2 scans across multiple hypotheses.
    """

    parser = argparse.ArgumentParser(
        description="Scans the chi2 across multiple prediction files.",
        epilog="""
            This program is equivalent to applying the chi2 command between the
            data and each prediction separately. It prints the results to stdout
            in csv format.""",
    )
    parser.add_argument("data", help="Input data file")
    parser.add_argument("predictions", nargs="+", help="Input predictions")
    parser.add_argument(
        "--poi", nargs="+", default=None, help="Scanned parameter of interest"
    )
    _add_common_arguments(parser)
    args = parser.parse_args()

    with open(args.data, "rb") as stream:
        x1 = pickle.load(stream)

    pois = [0] * len(args.predictions) if args.poi is None else args.poi
    assert len(pois) == len(args.predictions)

    print("file,poi,chi2,ndof,chi2,pval")
    print("file,poi,chi2,ndof,chi2/ndof,pval")

    for pred, poi in zip(args.predictions, pois):
        with open(pred, "rb") as stream:
            x2 = pickle.load(stream)

        result = _chi2_with_args(x1, x2, args)
        print(
            f"{pred},{poi},{result.chi2},{result.ndof},{result.chi2 / result.ndof},{result.pval}"
        )
