from ESPCN import *
from ImageDataset import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
import argparse
import os

"""def pnsr(y, y_hat, n, m, d: float = 255.0):
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if isinstance(y_hat, torch.Tensor):
            y_hat = y_hat.detach().cpu().numpy()
        sse = ((y - y_hat) ** 2).sum()
        if sse == 0.0:
            return float('inf')
        return 10.0 * np.log10((n * m * 3 * (d ** 2)) / sse)
"""


def pnsr(y, y_hat, d=1.0):
    mse = torch.mean((y - y_hat) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(d**2 / mse).item()


def load_model_checkpoint(checkpoint_path, device):
    """Load model and training metadata from checkpoint file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Validate checkpoint structure
    required_keys = ["model_state_dict", "model_class"]
    for key in required_keys:
        if key not in checkpoint:
            raise KeyError(f"Missing required key in checkpoint: {key}")
    
    # Create model instance
    if checkpoint["model_class"] == "ESPCN":
        scale = checkpoint.get("scale", 4)  # Default to 4 if not saved
        model = ESPCN(scale=scale)
    else:
        raise ValueError(f"Unknown model class: {checkpoint['model_class']}")
    
    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Extract metadata
    previous_epochs = checkpoint.get("epochs", 0)
    learning_rate = checkpoint.get("learning_rate", 0.003)  # Default to 0.003 if not saved
    pytorch_version = checkpoint.get("pytorch_version", "unknown")
    
    print(f"Loaded {checkpoint['model_class']} model (scale={scale})")
    print(f"Previous training: {previous_epochs} epochs")
    print(f"Learning rate: {learning_rate}")
    print(f"Saved with PyTorch version: {pytorch_version}")
    
    return model, previous_epochs, learning_rate


def entrainement(model, data, optimizer, criterion, device, epochs, start_epoch=0):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        currentLoss = 0.0
        pnsr_total = 0.0
        size = len(data)

        for y, X, label in tqdm(data):
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            output = model(X)

            pnsr_batch = pnsr(y, output)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            currentLoss += loss.item()

            pnsr_total += pnsr_batch
    
        # Show epoch number including previous training
        epoch_display = start_epoch + epoch + 1
        print(f"Epoch {epoch_display:02d} | Loss: {currentLoss/size:.6f} | PNSR : {pnsr_total/size:.2f}")


if __name__=="__main__":
    # CLI
    parser = argparse.ArgumentParser(description="Train ESPCN and export trained model")
    parser.add_argument("--output-model", type=str, default="artifacts/espcn_state_dict.pt", help="Path to save trained model state_dict")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--resume-from", type=str, help="Path to model checkpoint to resume training from")
    parser.add_argument("--learning-rate", type=float, default=0.003, help="Learning rate for training")
    args = parser.parse_args()

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

    # Initialize or load model
    previous_epochs = 0
    learning_rate = args.learning_rate  # Default from CLI
    
    if args.resume_from:
        try:
            model, previous_epochs, learning_rate = load_model_checkpoint(args.resume_from, device)
            # Override with CLI learning rate if explicitly provided
            if args.learning_rate != 0.003:  # User specified a different LR
                learning_rate = args.learning_rate
                print(f"Overriding checkpoint LR with CLI value: {learning_rate}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting fresh training instead...")
            model = ESPCN()
            previous_epochs = 0
            learning_rate = args.learning_rate
    else:
        model = ESPCN()
    
    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    epochs = args.epochs
    entrainement(model, train, optimizer, criterion, device, epochs, start_epoch=previous_epochs)

    # Calculate total epochs for saving
    total_epochs = previous_epochs + epochs

    # Ensure directory exists and save trained model
    output_path = args.output_model
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    save_payload = {
        "model_state_dict": model.state_dict(),
        "model_class": "ESPCN",
        "scale": getattr(model, "scale", None),
        "pytorch_version": torch.__version__,
        "device": device_str,
        "epochs": total_epochs,
        "learning_rate": learning_rate,
    }
    torch.save(save_payload, output_path)
    print(f"Saved trained model to: {output_path}")
    print(f"Total epochs trained: {total_epochs}")
