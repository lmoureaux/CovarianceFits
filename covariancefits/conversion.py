import argparse
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
            NumPy .npz file that is much faster to load.""",
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

    np.savez(args.output, bins=bins, data=data, covs=covs)
