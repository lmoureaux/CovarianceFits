import argparse
import itertools as it
import pickle
import sys
import numpy as np
import yaml


def load_data(yaml_data):
    """
    Turns HepData YAML data to bin edges and bin contents.
    """

    # Load the independent variable - the bin edges
    ivals = yaml_data["independent_variables"][0]["values"]
    bins = [ivals[i]["low"] for i in range(len(ivals))]
    bins.append(ivals[-1]["high"])

    # Load the dependent variable - the bin contents
    dvals = yaml_data["dependent_variables"][0]["values"]
    data = [val["value"] for val in dvals]

    return np.array(bins), np.array(data)


def load_covariances(yaml_data):
    """
    Turns HepData YAML covariances to a dictionary of square NumPy arrays, one
    per source of uncertainty.
    """

    covariances = {}
    for dep_var in yaml_data["dependent_variables"]:
        name = dep_var["header"]["name"]
        values = np.array([val["value"] for val in dep_var["values"]])

        bin_count = np.sqrt(len(values))
        assert bin_count % 1 == 0
        bin_count = int(bin_count)

        # This shortcut works when the data are nicely structured, in general
        # bins would need to be deduced from the independent_variables array
        covariances[name] = values.reshape((bin_count, bin_count))
    return covariances


def data2numpy():
    """
    Entry point to turn a measurement to a cached Numpy file.
    """

    parser = argparse.ArgumentParser(
        description="Caches the contents of a 1D histogram for fast lookup.",
        epilog="""
            This program takes two files in HepData YAML format: the first one
            (data) with the measured values, and the second (covs) with the
            covariance matrix of the measurement. It turns them into a single
            Pickle .pkl file that is much faster to load.""",
    )
    parser.add_argument("data", help="The input HepData file with the measured values")
    parser.add_argument(
        "covs",
        help="The input HepData file with the covariance matrix of the measurement",
    )
    parser.add_argument("output", help="Location of the output file")
    args = parser.parse_args()

    with open(args.data) as stream:
        bins, data = load_data(yaml.safe_load(stream))
    with open(args.covs) as stream:
        covs = load_covariances(yaml.safe_load(stream))

    with open(args.output, "wb") as stream:
        pickle.dump({"bins": bins, "data": data, "covs": covs}, stream)


def get_mc_errors(objects, base_name, scales=False):
    """
    Loads the muR and muF scale uncertainties from YODA objects for the given
    histogram. Adds the MC stat uncertainties. Returns them as a dict of covariance matrices.
    """

    hist = objects[base_name]
    central = hist.heights()
    stat_err = hist.yErrs()

    errors = {"MC_stat": np.diag(stat_err**2)}

    if scales:
        variations = []
        for mur, muf in it.product([0.5, 1, 2], [0.5, 1, 2]):
            if mur == 1 / muf:
                continue
            suffix = f"[MUR{mur:.1f}_MUF{muf:.1f}]"
            variations.append(objects[base_name + suffix].heights())

        envelope = np.max(np.abs(variations - central), axis=0)
        errors["MC_scale"] = np.outer(envelope, envelope)

    return errors


def yoda2numpy():
    """
    Entry point to turn a measurement to a cached Numpy file.
    """

    # https://gitlab.com/hepcedar/yoda/-/issues/77
    try:
        import yoda
    except ImportError as e:
        print("Could not import yoda. Make sure it is installed on your system.")
        print("Please see https://yoda.hepforge.org/ for more information.")
        print("Error message:", e)
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Caches the contents of a 1D histogram for fast lookup.",
        epilog="""
            This program takes a file YODA format containing the output of a
            Rivet routine. It extracts one histogram and turns it into a single
            Pickle .pkl file. Scale variations are optionally included as fully
            correlated uncertainties.""",
    )
    parser.add_argument("yoda", help="The input YODA file")
    parser.add_argument(
        "name", help="Name of the histogram to load (/ROUTINE/histogram)"
    )
    parser.add_argument("output", help="Location of the output file")
    parser.add_argument(
        "--scale-unc",
        default=False,
        action="store_true",
        help="Use scale uncertainties",
    )
    args = parser.parse_args()

    with open(args.yoda) as stream:
        objects = yoda.read(stream)

    if not args.name in objects:
        print(f"Could not find object {args.name} in file {args.yoda}")
        print("Available objects with similar names:")
        print()
        keys = list(filter(lambda key: args.name in key, objects.keys()))
        for key in keys:
            print(f"\t{key}")
        print()
        sys.exit(1)

    hist = objects[args.name]
    if not isinstance(hist, yoda.Histo1D):
        print(
            f"Unexpected object type for {args.name}: {type(hist).__name__} (expected Histo1D)"
        )
        sys.exit(1)

    with open(args.output, "wb") as stream:
        pickle.dump(
            {
                "bins": hist.xEdges(),
                "data": hist.heights(),
                "covs": get_mc_errors(objects, args.name, scales=args.scale_unc),
            },
            stream,
        )
