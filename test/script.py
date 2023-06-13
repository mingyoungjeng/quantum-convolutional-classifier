from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss, MSELoss

# from pennylane import NesterovMomentumOptimizer
from qcnn.qcnn import QCNN

from qcnn.ml.data import BinaryData
from qcnn.ml.optimize import Optimizer
from qcnn.ml.data import image_transform
from qcnn.experiment import Experiment
from qcnn.cnn import CNN

from pathlib import Path
from qcnn.file import save_dataframe_as_csv

from qcnn.quantum.operation.ansatz.convolution.v6 import ConvolutionAnsatz as Ansatz

# from qcnn.quantum.operation.ansatz import SimpleAnsatz as Ansatz

if __name__ == "__main__":
    # Meta parameters
    name = "name"
    path = Path(f"results/{name}")
    num_trials = 10
    silent = False
    is_quantum = True

    # Ansatz parameters
    dims = (28, 28)
    num_layers = 4

    # Create model
    cls = QCNN if is_quantum else CNN
    data = BinaryData(FashionMNIST, image_transform(dims, flatten=is_quantum))
    optimizer = Optimizer(Adam)
    loss = CrossEntropyLoss()
    epoch = 1
    model = cls.with_logging(data, optimizer, loss, epoch=epoch)

    # Log circuit ID
    model.logger.info(f"Circuit ID: {name}")

    # Save circuit drawing
    if is_quantum:
        model.ansatz = Ansatz.from_dims(dims, num_layers=num_layers)
        filename = path.with_stem(f"{name}_circuit").with_suffix(".png")
        model.ansatz.draw(filename=filename, decompose=True)

    # Run experiment
    experiment = Experiment(model, num_trials, results_schema=["accuracy"])

    args = (Ansatz,) if is_quantum else ()
    results = experiment(*args, dims, num_layers=num_layers, silent=silent)

    # Save and print accuracy results
    save_dataframe_as_csv(path.with_suffix(".csv"), results)
    acc = results["accuracy"]
    print(acc.median(), acc.mean(), acc.std())

    # Save aggregated loss history figure
    (fig,) = experiment.draw(path.with_suffix(".png"))
