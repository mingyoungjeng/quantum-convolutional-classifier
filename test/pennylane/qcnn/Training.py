# Implementation of Quantum circuit training procedure
import pennylane as qml
from pennylane import numpy as np
import QCNN_circuit


def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2
    loss = loss / len(labels)
    return loss


def cross_entropy(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        c_entropy = l * (np.log(p[l])) + (1 - l) * np.log(1 - p[1 - l])
        loss = loss + c_entropy
    return -1 * loss


def cost(params, X, Y, cost_fn):
    predictions = [QCNN_circuit.QCNN(x, params, cost_fn) for x in X]

    if cost_fn == "mse":
        loss = square_loss(Y, predictions)
    elif cost_fn == "cross_entropy":
        loss = cross_entropy(Y, predictions)
    return loss


# Circuit training parameters
steps = 200
learning_rate = 0.01
batch_size = 25
rng = np.random.default_rng()


def circuit_training(train_dataloader, cost_fn):
    total_params = (15 + 2) * 4
    params = rng.standard_normal(total_params, requires_grad=True)
    opt = qml.NesterovMomentumOptimizer(stepsize=learning_rate)
    loss_history = []

    for i, (X_batch, Y_batch) in enumerate(train_dataloader):
        params, cost_new = opt.step_and_cost(
            lambda v: cost(v, X_batch.numpy(), Y_batch.numpy(), cost_fn),
            params,
        )
        loss_history.append(cost_new)
        if i % 10 == 0:
            print(f"iteration: {i}/{len(train_dataloader)}, cost: {cost_new}")

    return loss_history, params
