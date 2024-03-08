import pennylane as qml
from config import *


class QuantumModel:
    """
    Quantum class which creates quantum feature map and the parametrized quantum circuit
    """
    def __init__(self, qubits) -> None:
        self.qubits = qubits
    def QuantumFeatureMap(self, X):
        lys = 5
        idx = 0
        for layer in range(lys):
            for i in range(self.qubits):
                qml.Rot(phi=X[idx + 0], theta=X[idx + 1], omega=X[idx + 2], wires=i)
                idx += 3
            last_idx = 60
            qml.RX(phi=X[last_idx], wires=0)
            qml.RY(phi=X[last_idx+1], wires=1)
            qml.RZ(phi=X[last_idx+2], wires=2)
            qml.RX(phi=X[last_idx+3], wires=3)
        qml.Barrier(only_visual=True)
    def Ring_like_layer(self, params):
        idx = 0
        for i in range(self.qubits):
            qml.Rot(phi=params[0 + idx], theta=params[1 + idx], omega=params[2 + idx], wires=i)
            idx += 3
        qml.Barrier(only_visual=True)
        for j in range(self.qubits - 1):
            qml.CRot(phi=params[idx + 0], theta=params[idx + 1], omega=params[idx + 2], wires=[j, j + 1])
            idx += 3
        qml.Barrier(only_visual=True)
        for k in range(self.qubits):
            qml.Rot(phi=params[0 + idx], theta=params[1 + idx], omega=params[2 + idx], wires=k)
            idx += 3






