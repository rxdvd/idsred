# IDSRED: INT-IDS data-reduction pipeline

This is a spectroscopic data-reduction pipeline for the Isaac Newton Telescope (INT) Intermediate Dispersion Spectrograph (IDS) instrument. 
It is optimised for the blue detector (EEV10).

[![repo](https://img.shields.io/badge/GitHub-temuller%2Fidsred-blue.svg?style=flat)](https://github.com/temuller/idsred)
[![license](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/temuller/idsred/blob/master/LICENSE)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
[![PyPI](https://img.shields.io/pypi/v/idsred?label=PyPI&logo=pypi&logoColor=white)](https://pypi.org/project/idsred/)
[![DOI](https://zenodo.org/badge/574921206.svg)](https://zenodo.org/badge/latestdoi/574921206)



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

or from source:

```code
git clone https://github.com/temuller/idsred.git
cd idsred
pip install .
```

Developer mode:

```code
pip install -e .
```

## Usage example

A notebook that explains how to use the pipeline is found in this repository [here](https://github.com/temuller/idsred/blob/main/reduction.ipynb).

## Contributing

To contribute, either open an issue or send a pull request (preferred option). You can also contact me directly.

## Citing IDSRED

```code
@software{tomas_e_muller_bravo_2023_7851772,
  author       = {Tomás E. Müller Bravo},
  title        = {temuller/idsred: First Release!},
  month        = apr,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v0.1.0},
  doi          = {10.5281/zenodo.7851772},
  url          = {https://doi.org/10.5281/zenodo.7851772}
}
```

## Acknowledgement

This pipeline is based on the [GROWTH school github repository](https://github.com/growth-astro/growth-school-2020), which nicely explains the entire reduction process for images and spectra, and on [INT-IDS-DataReduction](https://github.com/aayush3009/INT-IDS-DataReduction) for (long-slit) spectra reduction.
