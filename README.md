# Quantum Convolutional Classifier (QCC)

## Installation

> ### Create virtual environment [optional] 
> It is recommended to install this package in a virtual environment. One good option is [mamba](https://mamba.readthedocs.io/en/latest/), an alternative to [conda](https://docs.conda.io/en/latest/) based on C++.
>
> **After installing, run the following in the main directory of the repo:**
> ```sh
> mamba create --prefix $PWD/.env
> mamba activate $PWD/.env
> ```

**To build and install the `qcc` package, run the following:**
```sh
pip install --upgrade build # for setuptools
pip install -e .[development] # --editable installation
```

## Execution

```sh
# List the various options for running QCC
qcc --help
```
