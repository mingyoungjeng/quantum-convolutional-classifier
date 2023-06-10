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
    name = "skree2"
    path = Path(f"results/{name}")
    num_trials = 10
    silent = False
    is_quantum = True
    
    # Ansatz parameters
    dims = (28, 28)
    num_layers = 4

    # Create model
    cls = QCNN if is_quantum else CNN
    data = BinaryData(
        FashionMNIST, image_transform(dims, flatten=is_quantum)
    )
    optimizer = Optimizer(Adam)
    loss = CrossEntropyLoss()
    epoch = 200
    model = cls.with_logging(data, optimizer, loss, epoch=epoch)
    
    # Log circuit ID
    model.logger.info(f"Circuit ID: {name}")

    # Save circuit drawing
    if is_quantum:
        model.ansatz = Ansatz.from_dims(dims, num_layers=num_layers)
        circuit_drawing = model.ansatz.draw(decompose=True)
        circuit_drawing.savefig(path.with_stem(f"{name}_circuit").with_suffix(".png"))

    # Run experiment
    experiment = Experiment(model, num_trials, results_schema=["accuracy"])
    
    if is_quantum:
        results = experiment(Ansatz, dims, silent=silent, num_layers=num_layers)
    else:
        results = experiment(dims, num_layers, silent=silent)
    
    # Save and print accuracy results
    save_dataframe_as_csv(path.with_suffix(".csv"), results)
    acc = results["accuracy"]
    print(acc.median(), acc.mean(), acc.std())

    # Save aggregated loss history figure
    (fig,) = experiment.draw()
    fig.savefig(path.with_suffix(".png"))
