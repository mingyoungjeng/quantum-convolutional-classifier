import torch


# TODO: detect a plateau
def train(
    fn, optimizer, training_dataloader, cost_fn, initial_parameters=None, total_params=0
):
    params = (
        torch.randn(total_params, requires_grad=True)
        if initial_parameters is None
        else initial_parameters
    )
    opt = optimizer([params], lr=0.01, momentum=0.9, nesterov=True)

    for i, (data, labels) in enumerate(training_dataloader):
        opt.zero_grad()
        predictions = fn(params, data)

        if predictions.dim() == 1:  # Makes sure batch is 2D array
            predictions = predictions.unsqueeze(0)

        loss = cost_fn(predictions, labels)
        loss.backward()
        opt.step()

        if (i % 100 == 0) or i == len(training_dataloader) - 1:
            print(f"iteration: {i}/{len(training_dataloader)}, cost: {loss:.03f}")

    return params


def test(fn, params, testing_dataloader):
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in testing_dataloader:
            predictions = torch.argmax(fn(params, data))
            correct += torch.count_nonzero(predictions == labels)
            total += len(data)

    return correct / total
