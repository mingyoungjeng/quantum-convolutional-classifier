import pennylane as qml
from itertools import zip_longest, tee
from pennylane import numpy as np


# Unitary Ansatz for Convolutional Layer
def U_SU4(params, wires):  # 15 params, 2 qubit
    qml.U3(params[0], params[1], params[2], wires=wires[0])
    qml.U3(params[3], params[4], params[5], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[6], wires=wires[0])
    qml.RZ(params[7], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(params[8], wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.U3(params[9], params[10], params[11], wires=wires[0])
    qml.U3(params[12], params[13], params[14], wires=wires[1])


# Pooling Layer
def pooling_ansatz(params, wires):  # 2 params, 2 qubits
    qml.CRZ(params[0], wires=[wires[0], wires[1]])
    qml.PauliX(wires=wires[0])
    qml.CRX(params[1], wires=[wires[0], wires[1]])


def pooling_ansatz2(params, wires):  # 2 params, 2 qubits
    qml.CRZ(params[0], wires=[wires[0], wires[1]])
    qml.PauliX(wires=wires[0])
    qml.CRX(params[1], wires=[wires[0], wires[1]])
    qml.PauliX(wires=wires[0])


def pooling_ansatz3(params, wires):  # 2 params, 2 qubits
    m_0 = qml.measure(wires[0])
    qml.cond(m_0 == 0, qml.RY)(params[0], wires=wires[1])
    qml.cond(m_0 == 1, qml.RY)(params[1], wires=wires[1])


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


# TODO: work with num_classes > 2
def qcnn_ansatz(params, dims_q, *_, **__):
    wires = range(sum(dims_q))
    while len(wires) > 1:
        convolution(U_SU4, params[:U_params], wires)
        wires = pooling(
            pooling_ansatz, params[U_params : U_params + pool_params], wires
        )
        params = params[U_params + pool_params :]

    return wires  # np.array(wires)  # .item()


# total_params = (15 + 2) * num_layers
# TODO: work with num_classes > 2
def total_params(dims_q, *_, **__):
    num_layers = int(np.ceil(np.log2(sum(dims_q))))
    return (U_params + pool_params) * num_layers
