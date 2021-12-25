import torch
from tqdm.notebook import tqdm
from datetime import datetime


def get_log_name(model, loss_fn, optimizer, epochs, batch_size, lr):
    m = model.__class__.__name__
    l_fn = loss_fn.__class__.__name__
    opt = optimizer.__class__.__name__
    dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f'{m}_ep-{epochs}_bs-{batch_size}_lr-{lr}_l-fn-{l_fn}_opt-{opt}_{dt}'


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    avg_loss = 0
    model.train()
    with tqdm(dataloader) as pbar:
        for (X, _) in pbar:
            X = X.to(device)
            
            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, X)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # sum loss and correct
            avg_loss += loss.item()
            
            # set loss val to pbar
            pbar.set_postfix(loss=f'{loss.item():>7f}')
    
    # cal avg loss and correct
    avg_loss /= num_batches
    print(f"[Train] Avg loss: {avg_loss:>8f} \n")
    return avg_loss


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, X).item()
    test_loss /= num_batches
    print(f"[Test] Avg loss: {test_loss:>8f} \n")
    return test_loss