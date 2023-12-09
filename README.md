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
pip install -e .[dev] # --editable installation
```

## Execution

```bash
# List the various options for running QCC
qcc --help

# Run the test suite
cd test; qcc load config/
```

## Todo

- [ ] Write documentation / comments
  - [ ] Sphinx / MKDocs deployment?
- [ ] Rewrite all draw methods to use plotly
- [ ] Rename variables to match publication(s) and report
  - [ ] Shift &rarr stride
  - [ ] Permute &rarr Data Rearrangement
  - [ ] Filter &rarr Kernel
- [ ] Get Pennylane lightning working for faster execution
- [ ] Stride, dilation, padding in convolution
- [ ] Spin-off general methods into seperate KUARQ package
  - [ ] qcc.quantum
  - [ ] Qiskit gates
    - [ ] (Pennylane too?)
  - [ ] Experiment
  - [ ] Logger
  - [ ] Graph (plotly)
