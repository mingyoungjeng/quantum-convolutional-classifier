# This module contains the set of unitary ansatze that will be used to benchmark the performances of Quantum Convolutional Neural Network (QCNN) in QCNN.ipynb module
import pennylane as qml


# Unitary Ansatze for Convolutional Layer
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
