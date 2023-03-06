# This generates the results of the bechmarking code

import Benchmarking


"""
Here are possible combinations of benchmarking user could try.
dataset: 'mnist' or 'fashion_mnist'
cost_fn: 'mse' or 'cross_entropy'
Note: when using 'mse' as cost_fn binary="True" is recommended, when using 'cross_entropy' as cost_fn must be binary="False".
"""

dataset = "mnist"
classes = [0, 1]
binary = False
cost_fn = "cross_entropy"

Benchmarking.Benchmarking(dataset, classes, cost_fn=cost_fn, binary=binary)
