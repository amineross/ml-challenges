"""
class MNISTDataset(Dataset):
    def __init__(self, datasetDirectory):
        self.dir = datasetDirectory
        self.instanceFiles = sorted(os.listdir(datasetDirectory))
    
    def __len__(self):
        return len(self.instanceFiles)
    
    def __getitem__(self, idx):
        filename = os.path.join(self.dir, self.instanceFiles[idx])
        
        with open(filename, 'r') as f:
            ligne = f.readlines()[0][:-1]
            ligne = ligne.split(' ')
            y = int(ligne[0])
            X = [float(elt) for elt in ligne[1:]]
        binY = np.zeros(10)
        binY[y] = 1
        return np.array(X).reshape(28, 28), binY, y

if __name__ == '__main__':
    trainDataset = MNISTDataset('Dataset/Train')
    train = DataLoader(trainDataset, batch_size=1, shuffle=True)
    print(f'{len(trainDataset)} instances in train dataset')
    iterateurSurTrain = iter(train)
    batch = next(iterateurSurTrain)
    X, y = batch[0][0].numpy(), batch[2][0].numpy()
    imgplot = plt.imshow(X)
    plt.title(f'Label: {y}')
    plt.show()
 """

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, patches_dir):
        self.patches_dir = patches_dir
        self.patches_files = sorted([
            f for f in os.listdir(patches_dir) if f.endswith(".png")
        ])

    def __len__(self):
        return len(self.patches_files)

    def __getitem__(self, idx):
        filename = self.patches_files[idx]
        img_path = os.path.join(self.patches_dir, filename)

        img_base = Image.open(img_path)

        # High res
        img = img_base
        img = np.array(img)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float() / 255.0
        
        # Low res
        img_low_res_base = img_base.resize((img_base.width // 4, img_base.height // 4))
        img_low_res = np.array(img_low_res_base)
        img_low_res = img_low_res.transpose(2, 0, 1)
        img_low_res = torch.from_numpy(img_low_res).float() / 255.0


        return img, img_low_res, filename
    
if __name__ == '__main__':
    dataset = ImageDataset('dataset/patches-div2k')
    print(f"Nombre de patches : {len(dataset)}")
    img, img_low_res, fname = dataset[0]
    print(f"Shape du patch : {img.shape}, Shape du patch basse résolution : {img_low_res.shape}, nom : {fname}")
    
    # Convert tensors to displayable numpy arrays (H, W, C) in [0, 255] uint8
    img_disp = img.permute(1, 2, 0).clamp(0, 255).to(torch.uint8).numpy()
    img_low_disp = img_low_res.permute(1, 2, 0).clamp(0, 255).to(torch.uint8).numpy()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_disp)
    plt.title('High Res')
    plt.subplot(1, 2, 2)
    plt.imshow(img_low_disp)
    plt.title('Low Res')
    plt.tight_layout()
    plt.show()