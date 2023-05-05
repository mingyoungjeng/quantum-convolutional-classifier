import pennylane as qml
from pennylane.templates.embeddings import AmplitudeEmbedding
import unitary
from itertools import zip_longest, tee
from pennylane import numpy as np

U_params = 15
pool_params = 2


def convolution(U, params, iterable):
    a, b = tee(iterable)
    first = next(b, None)
    lst = list(zip_longest(a, b, fillvalue=first))
    last = lst.pop()[::-1]
    lst = lst[::2] + lst[1::2]
    lst.insert(0, last)

    for wires in lst:
        U(params, wires=wires)


def pooling(V, params, iterable):
    measurements = iterable[1::2]
    controlled = iterable[0::2]
    for wires in zip(measurements, controlled):
        V(params, wires=wires)

    return controlled


def QCNN_structure(U, params, wires):
    while len(wires) > 1:
        convolution(U, params[:U_params], wires)
        wires = pooling(
            unitary.pooling_ansatz, params[U_params : U_params + pool_params], wires
        )
        params = params[U_params + pool_params :]

    return np.array(wires).item()


num_qubits = 10
dev = qml.device("default.qubit", wires=num_qubits)


@qml.qnode(dev)
def QCNN(X, params, cost_fn="cross_entropy"):
    # Data Embedding
    AmplitudeEmbedding(X, wires=range(num_qubits), normalize=True, pad_with=0)

    meas = QCNN_structure(unitary.U_SU4, params, wires=range(num_qubits))

    if cost_fn == "mse":
        result = qml.expval(qml.PauliZ(meas))
    elif cost_fn == "cross_entropy":
        result = qml.probs(wires=meas)
    return result
