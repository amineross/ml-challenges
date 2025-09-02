from ESPCN import *
from ImageDataset import *
from torch.utils.data import DataLoader
from tqdm import tqdm



def entrainement (model, data, optimizer, device, epochs):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        currentLoss = 0.0
        size = len(data)

        for X, y, label in tqdm(data):
            y = y.to(device)

            optimizer.zero_grad()
            output = model(X)

            loss = ESPCN.pnsr(y, output, 128, 128)
            loss.backward()
            optimizer.step()
            currentLoss += loss.item()
    
        print(f"Epoch {epoch+1:02d} | Loss: {currentLoss/size:.6f}")


if __name__=="__main__":
    trainDataset = ImageDataset('dataset/patches-div2k')
    train = DataLoader(trainDataset, batch_size=10, shuffle=True)
    print(f"{len(train)*10} instances disponibles")

    device = "cpu"
    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator().type
    print(f"Using {device} device")

    model = ESPCN()
    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)

    entrainement(model, train, optimizer, device, 10)
