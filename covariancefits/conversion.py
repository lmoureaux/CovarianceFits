import argparse
import itertools as it
import pickle
import sys
import numpy as np
import yaml
from pathlib import Path


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


def data2pkl():
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


def load_lhcb_data(table_17, figure_13, table_22, figure_15):
    """
    Turns HepData YAML data to bin edges and bin contents (LHCb 13 TeV measurement).

    Source: https://www.hepdata.net/record/ins1990313
    """

    # Load the independent variable - the bin edges
    ivals = table_17["independent_variables"][0]["values"]
    bins = [ivals[i]["low"] for i in range(len(ivals))]
    bins.append(ivals[-1]["high"])
    bin_widths = np.diff(bins)

    # Load the independent variable - the bin edges. We only return pT.
    ivals = table_17["independent_variables"][0]["values"]

    # Load the dependent variables - the bin contents and some uncertainties
    data = []
    lumi = []
    stat = []
    for depvar in table_17["dependent_variables"]:
        data.append(bin_widths * [val["value"] for val in depvar["values"]])
        lumi.append(bin_widths * [val["errors"][2]["symerror"] for val in depvar["values"]])
        stat.append(bin_widths * [val["errors"][0]["symerror"] for val in depvar["values"]])

    data = np.array(data).flatten()
    lumi = np.array(lumi).flatten()
    stat = np.array(stat).flatten()

    # Now create some covariance matrices
    covs = {}
    covs["lumi"] = np.outer(lumi, lumi)  # We assume full correlation

    # Stat correlation matrix is in Figure 13 (left)
    # It is encoded according to "bin numbers", which are in the same order as
    # our bin indices.
    id1 = [v["value"] for v in figure_13["independent_variables"][0]["values"]]
    id2 = [v["value"] for v in figure_13["independent_variables"][1]["values"]]
    corr = [v["value"] for v in figure_13["dependent_variables"][0]["values"]]

    cov = np.outer(stat, stat)  # start with full correlation...
    for i1, i2, c in zip(id1, id2, corr):
        # ...then multiply with the correlation coefficient
        cov[i1 - 1][i2 - 1] *= c
    covs["stat"] = cov

    # Now let's add other systematics. They come in the same order as our table.
    # Full correlation.
    for depvar in table_22["dependent_variables"]:
        name = depvar["header"]["name"].replace("(%)", "").lower()
        unc = data * np.array([v["value"] for v in depvar["values"]]) / 100
        covs[name] = np.outer(unc, unc)

    # One last plot twist: the efficiency uncertainty isn't fully correlated.
    # Load this from the Figure 15 (left) which contains the correlation matrix.
    # Same logic as for the data.
    id1 = [v["value"] for v in figure_15["independent_variables"][0]["values"]]
    id2 = [v["value"] for v in figure_15["independent_variables"][1]["values"]]
    corr = [v["value"] for v in figure_15["dependent_variables"][0]["values"]]
    for i1, i2, c in zip(id1, id2, corr):
        # ...then multiply with the correlation coefficient
        covs["eff"][i1 - 1][i2 - 1] *= c

    return data, covs


def lhcb2pkl():
    """
    Entry point to turn the LHCb measurement to a cached Pickle file.
    """

    parser = argparse.ArgumentParser(
        description="Imports data from https://www.hepdata.net/record/ins1990313 v2",
    )
    parser.add_argument(
        "folder", type=Path, help="The folder with upacked HepData files"
    )
    parser.add_argument("output", help="Location of the output file")
    args = parser.parse_args()

    with (args.folder / "table_17.yaml").open() as table_17, (
        args.folder / "figure_13_(left).yaml"
    ).open() as figure_13, (
        args.folder / "figure_15_(left).yaml"
    ).open() as figure_15, (
        args.folder / "table_22.yaml"
    ).open() as table_22:
        data, covs = load_lhcb_data(
            yaml.safe_load(table_17),
            yaml.safe_load(figure_13),
            yaml.safe_load(table_22),
            yaml.safe_load(figure_15),
        )

    with open(args.output, "wb") as stream:
        pickle.dump({"bins": None, "data": data, "covs": covs}, stream)


def get_histogram_contents(
    objects, names, rescale=[1], suffix="", func="sumW"
) -> np.ndarray:
    """
    Concatenates bin contents from a set of histograms.
    """

    import yoda

    if len(rescale) == 1:
        rescale *= len(names)
    elif len(rescale) != len(names):
        raise ValueError(
            f"Different number of histogram names ({len(names)}) and rescale factors ({len(rescale)})"
        )

    all_values = []
    for name, scale in zip(names, rescale):
        full_name = name + suffix
        if not full_name in objects:
            raise ValueError(f"No {full_name} object in YODA file")

        hist = objects[full_name]
        if not isinstance(hist, yoda.Histo1D):
            raise ValueError(
                f"Unexpected object type for {full_name}: {type(hist).__name__} (expected Histo1D)"
            )

        all_values += [scale * getattr(b, func)() for b in hist.bins()]

    return np.array(all_values)


def get_mc_errors(objects, base_names, rescale=[1], scales=False, scales_mode="max"):
    """
    Loads the muR and muF scale uncertainties from YODA objects for the given
    histogram. Adds the MC stat uncertainties. Returns them as a dict of covariance matrices.
    """

    central = get_histogram_contents(objects, base_names, rescale=rescale)
    stat_err = get_histogram_contents(objects, base_names, rescale=rescale, func="errW")

    errors = {"MC_stat": np.diag(stat_err**2)}

    if scales:
        variations = []
        for mur, muf in it.product([0.5, 1, 2], [0.5, 1, 2]):
            if mur == 1 / muf:
                continue
            suffix = f"[MUR{mur:.1f}_MUF{muf:.1f}]"
            variations.append(
                get_histogram_contents(
                    objects, [n + suffix for n in base_names], rescale=rescale
                )
            )

        up = np.max(variations, axis=0) - central
        down = np.min(variations, axis=0) - central

        if scales_mode == "max":
            unc = np.max(np.abs([up, down]), axis=0)
        elif scales_mode == "mean":
            unc = np.mean(np.abs([up, down]), axis=0)
        else:
            raise ValueError(f"Unsupported scales mode {scales_mode}")

        errors["MC_scale"] = np.outer(unc, unc)

    return errors


def yoda2pkl():
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
        "names",
        nargs="+",
        help="Name of the histograms to load (/ROUTINE/histogram). Multiple histograms are concatenated.",
    )
    parser.add_argument("output", help="Location of the output file")
    parser.add_argument(
        "--rescale",
        default=[1],
        type=float,
        nargs="+",
        help="Multiply the cross sections by these factors (one number, or one per input histograms)",
    )
    parser.add_argument(
        "--scale-unc",
        default=False,
        action="store_true",
        help="Use scale uncertainties",
    )
    parser.add_argument(
        "--scale-unc-symmetrization",
        default="max",
        choices=["max", "mean"],
        help="How to symmetrize the up and down variations of scale uncertainties",
    )
    args = parser.parse_args()

    with open(args.yoda) as stream:
        objects = yoda.read(stream)

    with open(args.output, "wb") as stream:
        pickle.dump(
            {
                "bins": None,
                "data": get_histogram_contents(
                    objects, args.names, rescale=args.rescale
                ),
                "covs": get_mc_errors(
                    objects,
                    args.names,
                    rescale=args.rescale,
                    scales=args.scale_unc,
                    scales_mode=args.scale_unc_symmetrization,
                ),
            },
            stream,
        )
