def train(fn, opt, train_dataloader, cost_fn, initial_params=None):
    params = initial_params
    for i, batch in enumerate(train_dataloader):
        data, labels = batch
        opt.zero_grad()
        outputs = [fn(params, d) for d in data]
        loss = cost_fn(outputs, labels)
        loss.backward()
        opt.step()
