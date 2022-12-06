# IDSRED: INT-IDS data-reduction pipeline


## Installation

It is recommended to install it on an anaconda environment:

```code
conda create -n idsred pip
conda activate idsred
```

and install it using pip:

```code
pip install "idsred"
```

To install it in developer mode, use:

```code
pip install -e ".[dev]"
```

## Test

A simple test can be run with:

```code
run test
```

or for developers:

```code
pytest
```

## Acknowledgement

This pipeline is based on the [GROWTH school github repository](https://github.com/growth-astro/growth-school-2020), which nicely explains the entire reduction process for images and spectra, and on [INT-IDS-DataReduction](https://github.com/aayush3009/INT-IDS-DataReduction) for (long-slit) spectra reduction.
