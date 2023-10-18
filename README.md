# Quantum Convolutional Classifier (QCC)

## Installation

> ### Create virtual environment [optional]
>
> It is recommended to install this package in a virtual environment. One good option is [mamba](https://mamba.readthedocs.io/en/latest/), an alternative to [conda](https://docs.conda.io/en/latest/) based on C++.
>
> **After installing, run the following in the main directory of the repo.**
>
> ```bash
> mamba create --prefix $PWD/.env python
> mamba activate $PWD/.env
> ```

**To build and install the `qcc` package, run the following.**

```bash
pip install --upgrade build # for setuptools
pip install -e .[dev] # --editable installation
```

## Execution

```bash
# List the various options for running QCC
qcc --help

# Run the test suite
cd test; qcc load config/
```
