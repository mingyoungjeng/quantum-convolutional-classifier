from pathlib import Path

from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss, MSELoss

from qcnn.ml.data import BinaryData
from qcnn.ml.data import ImageTransform, ImageTransform1D
from qcnn.ml.optimize import Optimizer
from qcnn.experiment import Experiment
from qcnn.qcnn import QCNN
from qcnn.cnn import CNN
from qcnn.file import save_dataframe_as_csv

from qcnn.quantum.operation.ansatz.convolution import ConvolutionPoolingAnsatz as Ansatz

# from qcnn.quantum.operation.ansatz import BaselineAnsatz as Ansatz

if __name__ == "__main__":
    # Parameters
    name = "name"
    num_trials = 10
    dims = (28, 28)
    num_layers = 4
    batch_size = (32, 1000)
    epoch = 1
    options = {}

    # Meta parameters
    path = Path(f"results/{name}")
    silent = False
    is_quantum = True

    # Create model
    cls = QCNN if is_quantum else CNN
    transform = image_transform(dims, flatten=is_quantum)
    data = BinaryData(FashionMNIST, transform, batch_size=batch_size)
    optimizer = Optimizer(Adam)
    loss = CrossEntropyLoss()
    model = cls.with_logging(data, optimizer, loss, epoch=epoch)

    # Log important values
    model.logger.info(f"Circuit ID: {name}")
    model.logger.info(f"{data=}")
    model.logger.info(f"{optimizer=}")
    model.logger.info(f"{loss=}")

    model.logger.info(f"{num_trials=}")
    model.logger.info(f"{dims=}")
    model.logger.info(f"{num_layers=}")
    model.logger.info(f"{epoch=}")
    model.logger.info(f"{options=}")

    # Save circuit drawing
    if is_quantum:
        model.ansatz = Ansatz.from_dims(dims, num_layers=num_layers, **options)
        filename = path.with_stem(f"{name}_circuit").with_suffix(".png")
        model.ansatz.draw(filename=filename, decompose=True)

    # Run experiment
    experiment = Experiment(model, num_trials, results_schema=["accuracy"])

    args = (Ansatz,) if is_quantum else ()
    results = experiment(*args, dims, num_layers=num_layers, silent=silent, **options)

    # Save and print accuracy results
    save_dataframe_as_csv(path.with_suffix(".csv"), results)
    acc = results["accuracy"]
    print(acc.median(), acc.mean(), acc.std())

    # Save aggregated loss history figure
    (fig,) = experiment.draw(path.with_suffix(".png"))
