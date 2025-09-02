from ESPCN import *
from ImageDataset import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np

def pnsr(y, y_hat, n, m, d: float = 255.0):
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if isinstance(y_hat, torch.Tensor):
            y_hat = y_hat.detach().cpu().numpy()
        sse = ((y - y_hat) ** 2).sum()
        if sse == 0.0:
            return float('inf')
        return 10.0 * np.log10((n * m * 3 * (d ** 2)) / sse)

def entrainement (model, data, optimizer, criterion, device, epochs):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        currentLoss = 0.0
        size = len(data)

        for y, X, label in tqdm(data):
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            output = model(X)

            currentAcc = pnsr(y, output, 128, 128)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            currentLoss += loss.item()
    
        print(f"Epoch {epoch+1:02d} | Loss: {currentLoss/size:.6f} | Accuracy : {currentAcc/size:.2f}")


if __name__=="__main__":
    trainDataset = ImageDataset('dataset/patches-div2k')
    train = DataLoader(trainDataset, batch_size=10, shuffle=True)
    print(f"{len(train)*10} instances disponibles")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_str = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        device_str = "mps"
    else:
        try:
            import torch_directml
            device = torch_directml.device()
            device_str = "dml"
        except Exception:
            device = torch.device("cpu")
            device_str = "cpu"
    print(f"Using {device_str} device")

    model = ESPCN()
    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    criterion = nn.MSELoss()

    entrainement(model, train, optimizer, criterion, device, 10)
