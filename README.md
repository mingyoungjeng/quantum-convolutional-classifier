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

## Code Structure

### Source Code File Tree (`src/qcc`)

```bash
.
├── cli # Command-line interface (start here)
├── experiment # Generic class that runs multiple experimental trials
│   └── logger # Custom logging that stores outputs in DataFrames
├── file # Useful I/O commands
├── filters.py # Classical convolution and common filters
├── graph # Auto-formatted graphs based on plotly - a.k.a. David bot
├── ml # PyTorch implementations of ML models
│   ├── cnn.py # Convolutional neural network implementation
│   ├── data.py # Custom dataset wrapper
│   ├── ml.py # ML-related functions
│   ├── mlp.py # Multi-layer perceptron implementation
│   ├── model.py # Generic class for ML experiments (training + testing)
│   ├── optimize.py # Custom optimizer wrapper
│   ├── quantum.py # MQCC and MQCC Optimized
│   └── quanvolution.py # Quanvolution PyTorch module
└── quantum
    ├── pennylane # Quantum operations in Pennylane
    │   └── ansatz # Ansatz compatible with Pennylane
    ├── qiskit # Qiskit classes and functions 
    │   └── ansatz # Ansatz compatible with Qiskit
    │              # (WARNING: ML / Ansatz related items are likely jank)
    ├── quantum.py # Quantum-related functions
    └── qubits.py # 2D list for representing multidimensional quantum circuits
```

### Docs / Examples File Tree (`docs/`)

TBD

### Test Suite File Tree (`test/`)

TBD

## Todo

- [ ] Write documentation / comments
  - [ ] Sphinx / MKDocs deployment?
- [ ] Use <https://arxiv.org/pdf/quant-ph/0410184.pdf> for shift operation
- [ ] Rewrite all draw methods to use plotly
- [ ] Rename variables to match publication(s) and report
  - [ ] Shift → stride
  - [ ] Permute → Data Rearrangement
  - [ ] Filter → Kernel
- [ ] Get Pennylane lightning working for faster execution
- [ ] Stride, dilation, padding in convolution
- [ ] Spin-off general methods into seperate KUARQ package
  - [ ] qcc.quantum
  - [ ] Qiskit gates
    - [ ] (Pennylane too?)
  - [ ] Experiment
  - [ ] Logger
  - [ ] Graph (plotly)
- [ ] Major project: research method of unifying Qiskit and Pennylane quantum gates
- [ ] There is a good chance I typed some Sequences as Iterables. Check this.
- [ ] Add bias terms using quantum
