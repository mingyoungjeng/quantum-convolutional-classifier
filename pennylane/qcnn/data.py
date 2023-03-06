# Loads and Processes the data that will be used in QCNN and Hierarchical Classifier Training
import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

transform = transforms.Compose(
    [
        # transforms.Resize((16, 16)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(torch.squeeze(x).T)),
    ]
)


def data_load_and_process(dataset, classes=None, binary=True, batch_size=4):
    if dataset == "fashion_mnist":
        dataset = datasets.FashionMNIST
    elif dataset == "mnist":
        dataset = datasets.MNIST
    else:
        return

    if classes is None:
        classes = [0, 1]

    for is_train in [True, False]:
        data = dataset(
            root="data",
            train=is_train,
            download=True,
            transform=transform,
        )

        (idx,) = np.isin(data.targets, classes).nonzero()
        subset = Subset(data, idx)
        dataloader = DataLoader(
            subset, batch_size=batch_size if is_train else 1, shuffle=is_train
        )

        if is_train:
            training_dataloader = dataloader
        else:
            testing_dataloader = dataloader

    return training_dataloader, testing_dataloader


# x_train, x_test = (
#     x_train[..., np.newaxis] / 255.0,
#     x_test[..., np.newaxis] / 255.0,
# )  # normalize the data

# if classes == "odd_even":
#     odd = [1, 3, 5, 7, 9]
#     X_train = x_train
#     X_test = x_test
#     if binary == False:
#         Y_train = [1 if y in odd else 0 for y in y_train]
#         Y_test = [1 if y in odd else 0 for y in y_test]
#     elif binary == True:
#         Y_train = [1 if y in odd else -1 for y in y_train]
#         Y_test = [1 if y in odd else -1 for y in y_test]

# elif classes == ">4":
#     greater = [5, 6, 7, 8, 9]
#     X_train = x_train
#     X_test = x_test
#     if binary == False:
#         Y_train = [1 if y in greater else 0 for y in y_train]
#         Y_test = [1 if y in greater else 0 for y in y_test]
#     elif binary == True:
#         Y_train = [1 if y in greater else -1 for y in y_train]
#         Y_test = [1 if y in greater else -1 for y in y_test]
