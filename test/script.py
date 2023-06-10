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

from qcnn.quantum.operation.ansatz.convolution.v6 import ConvolutionAnsatz as A
# from qcnn.quantum.operation.ansatz import SimpleAnsatz as A

if __name__ == "__main__":
    # Meta parameters
    name = "skree2"
    path = Path(f"results/{name}")
    num_trials = 10
    silent = False
    
    # Ansatz parameters
    dims = (28, 28)
    num_layers = 4

    # Create QCNN
    data = BinaryData(
        FashionMNIST, image_transform(dims, flatten=True)
    )
    optimizer = Optimizer(Adam)
    loss = CrossEntropyLoss()
    epoch = 200
    qcnn = QCNN.with_logging(data, optimizer, loss, epoch=epoch)
    
    # Log circuit ID
    qcnn.logger.info(f"Circuit ID: {name}")

    # Save circuit drawing
    qcnn.ansatz = A.from_dims(dims, num_layers=num_layers)
    circuit_drawing = qcnn.ansatz.draw(decompose=True)
    circuit_drawing.savefig(path.with_stem(f"{name}_circuit").with_suffix(".png"))

    # Run experiment
    experiment = Experiment(qcnn, num_trials, results_schema=["accuracy"])
    # results = experiment(dims, num_layers, silent=silent)
    results = experiment(A, dims, silent=silent, num_layers=num_layers)
    
    # Save and print accuracy results
    save_dataframe_as_csv(path.with_suffix(".csv"), results)
    acc = results["accuracy"]
    print(acc.median(), acc.mean(), acc.std())

    # Save aggregated loss history figure
    (fig,) = experiment.draw()
    fig.savefig(path.with_suffix(".png"))
