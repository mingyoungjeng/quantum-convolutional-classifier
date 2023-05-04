from unitary.unitary import Unitary
import pennylane as qml


class BaselineConvolution(Unitary):
    @staticmethod
    def _u(params, wires) -> None:
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

    def total_params(self, *_) -> int:
        return 15


class BaselinePooling1(Unitary):
    @staticmethod
    def _u(params, wires) -> None:
        qml.CRZ(params[0], wires=[wires[0], wires[1]])
        qml.PauliX(wires=wires[0])
        qml.CRX(params[1], wires=[wires[0], wires[1]])

    def total_params(self, *_) -> int:
        return 2


class BaselinePooling2(Unitary):
    @staticmethod
    def _u(params, wires) -> None:
        qml.CRZ(params[0], wires=[wires[0], wires[1]])
        qml.PauliX(wires=wires[0])
        qml.CRX(params[1], wires=[wires[0], wires[1]])
        qml.PauliX(wires=wires[0])

    def total_params(self, *_) -> int:
        return 2


class BaselinePooling3(Unitary):
    @staticmethod
    def _u(params, wires) -> None:
        m_0 = qml.measure(wires[0])
        qml.cond(m_0 == 0, qml.RY)(params[0], wires=wires[1])
        qml.cond(m_0 == 1, qml.RY)(params[1], wires=wires[1])

    def total_params(self, *_) -> int:
        return 2
