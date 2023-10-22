# Fits with covariance matrices

This repository contains Python-based tools to perform chi-squared fits to
particle physics measurements while taking the covariance matrices of the data
into account.

### Data import

All tools work on special (very simple) files, each containing a single
histogram and its uncertainty. The first step to do a calculation is thus to
convert input files to this format. Two tools are provided for this purpose:

* **`data2pkl`** imports HepData YAML. It assumes that the data points and their
  covariance matrices are stored in two distinct YAML files. Each source of
  uncertainty should be a `dependent_variable` in the file that contains the
  covariance matrix. The total covariance should *not* be provided alongside
  individual sources.

  Basic usage:
  ```
  data2pkl file-with-histogram.yaml file-with-covariance.yaml data-histogram.pkl
  ```

* **`yoda2pkl`** imports YODA files. It requires the YODA Python bindings to be
  installed locally. It imports a single histogram and has optional support for
  loading `MUR` and `MUF` variations (it then takes the envelope as the
  uncertainty).

  Basic usage:
  ```
  yoda2pkl prediction.yoda /Analysis/histogram-name pred-histogram.pkl --scale-unc
  ```

### Calculating chi2s

Calculating the chi2s between two histograms is done with the command
**`chi2`**, which takes two files as inputs. This command also supports
restricting the considered range with `--first-bin` and `--last-bin`.

Basic usage:
```
chi2 data-histogram pred-histogram.pkl --first-bin 0 --last-bin 2
```
This will calculate the chi2 using the first 3 bins (0 to 2 inclusive). Negative
values are supported for `--last-bin` and works as usual in Python.

### Conbined chi2

A combined chi2 with multiple histograms is similar to a the normal case, except
that the histograms need to be merged first. The tool for this is called
**`merge-histograms`**. It takes the following inputs:

* A list of histogram files to merge
* Bin ranges to include in the merged output
* The histogram file in which the output should be placed
* A list of uncertainties to consider **uncorrelated** between the inputs. This
  is an essential physics input for the final calculation and will affect the
  results, so think about it! Note that it only makes sense to consider
  uncertainties as correlated between bins if they are correlated within the bin
  in the first place. To make your life easier, this field supports regular
  expressions.

Basic usage:
```
merge-histograms -o merged.pkl input1.pkl input2.pkl --ranges 0:4 0:5 --uncorrelated '.*[sS]tat.*'
```
This will merge the first 5 bins of `input1.pkl` with the first 6 bins of
`input2.pkl`, considering any uncertainty whose name contains `Stat` or `stat`
as uncorrelated between the inputs.

## Histogram data format

The histograms data format is designed to be relatively fast and extremely
simple to use (in Python). It is based on the `pickle` serialization library.
Each file contains a dictionary with the following keys:

* `data`: A 1D Numpy array with the bin contents.
* `bins`: A 1D Numpy array with the bin boundaries.
* `covs`: A Python dictionary. Each item is a source of uncertainty. The name is
  the uncertainty and the value is a 2D Numpy array encoding the uncertainty as
  a covariance matrix.
