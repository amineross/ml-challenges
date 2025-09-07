from ESPCN import *
from FSRCNN import *
from EDSRLITE import *
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
    """charge le modèle et les métadonnées d'entraînement depuis le fichier checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"fichier checkpoint introuvable: {checkpoint_path}")
    
    print(f"chargement du modèle depuis: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # valider la structure du checkpoint
    required_keys = ["model_state_dict", "model_class"]
    for key in required_keys:
        if key not in checkpoint:
            raise KeyError(f"clé requise manquante dans le checkpoint: {key}")
    
    # créer l'instance du modèle
    if checkpoint["model_class"] == "ESPCN":
        #scale = checkpoint.get("scale", 4)  # par défaut 4 si pas sauvegardé
        model = ESPCN()
    elif checkpoint["model_class"] == "FSRCNN":
        model = FSRCNN()
    elif checkpoint["model_class"] == "EDSRLITE":
        model = EDSRLITE()
    else:
        raise ValueError(f"classe de modèle inconnue: {checkpoint['model_class']}")
    
    # charger le state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # extraire les métadonnées
    previous_epochs = checkpoint.get("epochs", 0)
    learning_rate = checkpoint.get("learning_rate", 0.003)  # par défaut 0.003 si pas sauvegardé
    pytorch_version = checkpoint.get("pytorch_version", "unknown")
    
    if checkpoint["model_class"] == "ESPCN":
        print(f"modèle {checkpoint['model_class']} chargé")
    elif checkpoint["model_class"] == "FSRCNN":
        print(f"{checkpoint['model_class']} chargé")
    elif checkpoint["model_class"] == "EDSRLITE":
        print(f"{checkpoint['model_class']} chargé")
    
    print(f"entraînement précédent: {previous_epochs} epochs")
    print(f"taux d'apprentissage: {learning_rate}")
    print(f"sauvegardé avec pytorch version: {pytorch_version}")
    
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
    
        # afficher le numéro d'epoch incluant l'entraînement précédent
        epoch_display = start_epoch + epoch + 1
        print(f"Epoch {epoch_display:02d} | Loss: {currentLoss/size:.6f} | PNSR : {pnsr_total/size:.2f}")


if __name__=="__main__":
    # cli
    parser = argparse.ArgumentParser(description="entraîner espcn et exporter le modèle entraîné")
    parser.add_argument("--model-class", type=str, default="ESPCN", help="classe de modèle à entraîner")
    parser.add_argument("--output-model", type=str, default="artifacts/espcn_state_dict.pt", help="chemin pour sauvegarder le state_dict du modèle entraîné")
    parser.add_argument("--epochs", type=int, default=10, help="nombre d'epochs d'entraînement")
    parser.add_argument("--resume-from", type=str, help="chemin vers le checkpoint de modèle pour reprendre l'entraînement")
    parser.add_argument("--learning-rate", type=float, default=0.003, help="taux d'apprentissage pour l'entraînement")
    parser.add_argument("--batch-size", type=int, default=10, help="taille de batch pour l'entraînement")   
    args = parser.parse_args()

    trainDataset = ImageDataset('dataset/patches-div2k')
    train = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True)
    print(f"{len(train)*10} instances disponibles")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_str = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        device_str = "mps"
    else:
        device = torch.device("cpu")
        device_str = "cpu"
    print(f"utilisation du périphérique {device_str}")

    # initialiser ou charger le modèle
    previous_epochs = 0
    learning_rate = args.learning_rate  # par défaut depuis cli
    
    if args.model_class == "ESPCN":
        model = ESPCN()
    elif args.model_class == "FSRCNN":
        model = FSRCNN()
    elif args.model_class == "EDSRLITE":
        model = EDSRLITE()

    if args.resume_from:
        try:
            model, previous_epochs, learning_rate = load_model_checkpoint(args.resume_from, device)
            # override avec le lr cli si explicitement fourni
            if args.learning_rate != 0.003:  # utilisateur a spécifié un lr différent
                learning_rate = args.learning_rate
                print(f"remplacement du lr checkpoint par la valeur cli: {learning_rate}")
        except Exception as e:
            print(f"erreur lors du chargement du checkpoint: {e}")
            print("démarrage d'un entraînement frais à la place...")
            previous_epochs = 0
            learning_rate = args.learning_rate

    
    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    epochs = args.epochs
    entrainement(model, train, optimizer, criterion, device, epochs, start_epoch=previous_epochs)

    # calculer le total d'epochs pour la sauvegarde
    total_epochs = previous_epochs + epochs

    # s'assurer que le répertoire existe et sauvegarder le modèle entraîné
    output_path = args.output_model
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    save_payload = {
        "model_state_dict": model.state_dict(),
        "model_class": args.model_class,
        "scale": 4,
        "pytorch_version": torch.__version__,
        "device": device_str,
        "epochs": total_epochs,
        "learning_rate": learning_rate,
    }
    
    
    torch.save(save_payload, output_path)
    print(f"modèle entraîné sauvegardé vers: {output_path}")
    print(f"total d'epochs entraînés: {total_epochs}")
