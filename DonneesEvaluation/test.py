import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F
import numpy as np
import sys
import time
from tqdm import tqdm

# importer depuis le répertoire parent
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from FSRCNN import *
from ESPCN import *

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
import numpy as np
import math 
from PIL import Image

class EvalImageDataset(Dataset):
    """classe dataset pour datasets d'évaluation avec répertoires X et Y séparés"""
    def __init__(self, x_dir, y_dir):
        self.x_dir = x_dir
        self.y_dir = y_dir
        self.files = sorted([
            f for f in os.listdir(x_dir) if f.endswith((".png", ".jpg", ".jpeg"))
        ])
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        filename = self.files[idx]
        x_path = os.path.join(self.x_dir, filename)
        y_path = os.path.join(self.y_dir, filename)
        
        # charger l'entrée basse résolution
        x_img = Image.open(x_path).convert('RGB')
        x_array = np.array(x_img)
        t_x = torch.utils.dlpack.from_dlpack(x_array.__dlpack__())
        x_tensor = t_x.permute(2, 0, 1).to(dtype=torch.float32).div(255.0)
        
        # charger la cible haute résolution
        y_img = Image.open(y_path).convert('RGB')
        y_array = np.array(y_img)
        t_y = torch.utils.dlpack.from_dlpack(y_array.__dlpack__())
        y_tensor = t_y.permute(2, 0, 1).to(dtype=torch.float32).div(255.0)
        
        return x_tensor, y_tensor

def compute_metrics(pred, target):
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    psnr_list = []
    for i in range(pred_np.shape[0]):
        p = np.transpose(pred_np[i], (1, 2, 0)) 
        t = np.transpose(target_np[i], (1, 2, 0))
        score = psnr(t, p, data_range=1.0)
        if score>100: score = 100
        psnr_list.append(score)

    return np.mean(psnr_list)

def safe_psnr(pred, target):
    pred_np = pred.detach().cpu()
    target_np = target.detach().cpu()
    psnr_list = []
    n,m = pred_np.shape[2], pred_np.shape[3]
    for i in range(pred_np.shape[0]): # pour chaque instance du batch
        mse = F.mse_loss(pred_np[i], target_np[i], reduction='mean').item()
        if mse==0: psnr_list.append(40)
        else: psnr_list.append(10 * math.log10(1.0 / mse))
    return np.mean(psnr_list)


def eval(model, dataloader, device, sameSize=False):
    model.to(device)
    model.eval()

    total_psnr = 0
    total_psnr2 = 0
    with torch.no_grad():
        for index, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            inputs, targets = batch
            if sameSize:
                inputs = F.interpolate(inputs, scale_factor=2, mode='bicubic', align_corners=False)
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            total_psnr += safe_psnr(outputs, targets)#compute_metrics(val_outputs, val_targets)
            total_psnr2 += compute_metrics(outputs, targets)

    avg_psnr = total_psnr / len(dataloader)
    avg_psnr2 = total_psnr2 / len(dataloader)
    
    print(f"PSNR: {avg_psnr:.2f} (=?{avg_psnr2:.2f})")

def load_model_checkpoint(checkpoint_path, device):
    """charger le modèle depuis le fichier checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"fichier checkpoint introuvable: {checkpoint_path}")
    
    print(f"chargement du modèle depuis: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # gérer l'ancien format (state_dict direct) et le nouveau format (dict checkpoint)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # nouveau format checkpoint
        model_class = checkpoint["model_class"]
        state_dict = checkpoint["model_state_dict"]
    
        if model_class == "ESPCN":
            model = ESPCN()
        elif model_class == "FSRCNN":
            model = FSRCNN()
        else:
            raise ValueError(f"classe de modèle inconnue: {model_class}")
            
        model.load_state_dict(state_dict)
        return model, model_class.lower()
    else:
        # ancien format (state_dict direct) - on doit déduire le type de modèle
        raise ValueError("ancien format checkpoint non supporté. veuillez fournir le type de modèle explicitement.")

if __name__=="__main__":
    datasetSet5 = EvalImageDataset(f'Set5/X', f'Set5/Y')
    datasetSet14 = EvalImageDataset(f'Set14/X', f'Set14/Y')

    # détection appropriée du périphérique
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
    
    set5 = DataLoader(datasetSet5, batch_size=1, shuffle=True)
    set14 = DataLoader(datasetSet14, batch_size=1, shuffle=True)
    
    if len(sys.argv) < 3:
        print("usage: python test.py <type_modèle> <chemin_modèle>")
        print("types de modèles disponibles: fsrcnn, espcn")
        sys.exit(1)
    
    method = sys.argv[1].lower()
    model_path = sys.argv[2]
    
    # charger le modèle depuis le checkpoint
    try:
        model, detected_method = load_model_checkpoint(model_path, device)
        method = detected_method  # utiliser la méthode du checkpoint
    except Exception as e:
        print(f"erreur lors du chargement du checkpoint: {e}")
        print("retour à la création manuelle du modèle...")
        
        # création manuelle du modèle si le chargement du checkpoint échoue
        if method == 'fsrcnn':
            model = FSRCNN(scale=4)
        elif method == 'espcn':
            model = ESPCN(scale=4)
        else:
            print(f"méthode inconnue: {method}")
            sys.exit(1)
        
        # essayer de charger le state dict directement
        try:
            state_dict = torch.load(model_path, map_location=device)
            if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                model.load_state_dict(state_dict["model_state_dict"])
            else:
                model.load_state_dict(state_dict)
        except Exception as e:
            print(f"erreur lors du chargement des poids du modèle: {e}")
            sys.exit(1)
    
    print(f"évaluation de {method} sur set5")
    eval(model, set5, device=device, sameSize=False)

    print(f"évaluation de {method} sur set14")
    eval(model, set14, device=device, sameSize=False)

