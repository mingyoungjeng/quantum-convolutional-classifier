import machine_learning.data as data
import Training
import QCNN_circuit
import numpy as np
from pathlib import Path


def accuracy_test(predictions, labels, cost_fn):
    if cost_fn == "mse":
        acc = 0
        for l, p in zip(labels, predictions):
            if np.abs(l - p) < 1:
                acc = acc + 1
        return acc / len(labels)

    elif cost_fn == "cross_entropy":
        acc = 0
        for l, p in zip(labels, predictions):
            if p[0] > p[1]:
                P = 0
            else:
                P = 1
            if P == l:
                acc = acc + 1
        return acc / len(labels)


filename = Path("result/result.txt")


def Benchmarking(dataset, classes, cost_fn, binary=True):
    training_dataloader, testing_dataloader = data.data_load_and_process(
        dataset, classes=classes, binary=binary
    )

    print(f"Loss History with {cost_fn}")
    loss_history, trained_params = Training.circuit_training(
        training_dataloader, cost_fn
    )

    predictions, labels = zip(
        *[
            [
                QCNN_circuit.QCNN(np.squeeze(x.numpy()), trained_params, cost_fn),
                np.squeeze(y.numpy()),
            ]
            for x, y in testing_dataloader
        ]
    )

    accuracy = accuracy_test(predictions, labels, cost_fn)
    print(f"Accuracy: {accuracy}")

    with open(filename, "a") as file:
        file.write(f"Loss History with {cost_fn}\n")
        file.write(f"{loss_history}\n")
        file.write(f"Accuracy: {accuracy}\n")
