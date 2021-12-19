import torch
from tqdm.notebook import tqdm


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    avg_loss, avg_correct = 0, 0
    model.train()
    with tqdm(dataloader) as pbar:
        for (X, y) in pbar:
            X, y = X.to(device), y.to(device)
            
            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # sum loss and correct
            avg_loss += loss.item()
            avg_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
            # set loss val to pbar
            pbar.set_postfix(loss=f'{loss.item():>7f}')
    
    # cal avg loss and correct
    avg_loss /= num_batches
    avg_correct /= size
    print(f"[Train] Accuracy: {(100*avg_correct):>0.1f}%, Avg loss: {avg_loss:>8f} \n")
    return avg_loss, avg_correct


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"[Test] Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct