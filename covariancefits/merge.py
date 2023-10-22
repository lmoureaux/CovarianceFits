import argparse
import numpy as np
import pickle
import re
import scipy.linalg

def select_bins(x, covs, first_bin=0, last_bin=-1):
    """
    Selects the bins to use. Returns x and covs with only the needed values.
    Also returns the range used to obtain them.
    """

    last_bin = len(x) + last_bin if last_bin < 0 else last_bin
    bins = range(first_bin, last_bin + 1)

    selected_covs = {}
    for key, cov in covs.items():
        selected_covs[key] = cov[bins].T[bins].T

    return x[bins], selected_covs, bins


def mergetool():
    """
    Entry point to merge histograms from multiple files.
    """

    parser = argparse.ArgumentParser(
        description="Merges histograms from multiple files.",
        epilog="""
            This program takes multiple files in Pickle .pkl format created by
            data2numpy or yoda2numpy and combines the histograms into a bigger
            one. This can be used to obtain a combined chi2.""",
    )
    parser.add_argument("files", metavar="file", nargs="+", help="Input files")
    parser.add_argument("-o", "--output", required=True, help="Output file")
    parser.add_argument("--ranges", metavar="range", nargs="+", help="Bins to use, as first:last for each input file")
    parser.add_argument("--uncorrelated", nargs="*", help="List of uncorrelated uncertainties (between mass bins). Regular expressions are supported")
    args = parser.parse_args()

    if args.ranges:
        assert len(args.files) == len(args.ranges), "Different number of files and ranges"
    else:
        args.ranges = ["0:-1" for _ in args.files]

    all_files = []
    all_cov_names = set()
    for path in args.files:
        with open(path, "rb") as stream:
            all_files.append(pickle.load(stream))
        all_cov_names |= all_files[-1]["covs"].keys()

    print("Treatment of uncertainties between input files:")
    for name in sorted(list(all_cov_names)):
        if any(map(lambda pattern: re.match(pattern, name), args.uncorrelated)):
            print(f"{name:15s} ... uncorrelated")
        else:
            print(f"{name:15s} ... correlated")

    all_data = []
    all_covs = {}
    for x, bins in zip(all_files, args.ranges):
        first_bin, last_bin = map(int, bins.split(":"))
        data, covs, bins = select_bins(x["data"], x["covs"], first_bin, last_bin)
        all_data.append(data)
        for name in all_cov_names:
            # Some covs may not be present for all variables
            cov = covs[name] if name in covs else np.zeros(shape=[len(data)] * 2)
            all_covs[name] = all_covs.get(name, []) + [cov]

    merged_data = np.concatenate(all_data)
    merged_covs = {}
    for name, covs in all_covs.items():
        if any(map(lambda pattern: re.match(pattern, name), args.uncorrelated)):
            # Make a block-diagonal matrix
            merged_covs[name] = scipy.linalg.block_diag(*covs)
        else:
            # Make a fully correlated matrix
            variances = np.concatenate([np.diag(c) for c in covs])
            # Calculate standard deviations
            std = np.sqrt(variances)
            # A fully correlated matrix is an outer product
            merged_covs[name] = np.outer(std, std)

    print(f"Saving {len(merged_data)} bins to {args.output}")

    with open(args.output, "wb") as stream:
        pickle.dump(
            {
                "bins": np.linspace(0, len(merged_data), len(merged_data) + 1),
                "data": merged_data,
                "covs": merged_covs,
            },
            stream,
        )
