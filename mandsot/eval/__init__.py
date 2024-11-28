import torch
from torch import nn


def evaluate(model, data_loader, device):
    model.eval()
    criterion = nn.MSELoss()

    total_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets, initial in data_loader:
            inputs, targets, initial = inputs.float().to(device), targets.float().to(device), initial.long().to(device)
            outputs = model(inputs, initial)
            loss = criterion(outputs.squeeze(), targets.squeeze())

            mae = torch.abs(outputs.squeeze() - targets.squeeze()).mean()
            mse = ((outputs.squeeze() - targets.squeeze()) ** 2).mean()

            total_loss += loss.item()
            total_mae += mae.item()
            total_mse += mse.item()
            total_samples += len(inputs)

    avg_loss = total_loss / total_samples
    avg_mae = total_mae / total_samples
    avg_mse = total_mse / total_samples

    return avg_loss, avg_mae, avg_mse
