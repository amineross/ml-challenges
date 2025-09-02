""" import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt

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

class ImageDataset(Dataset):
    def __init__(self, datasetDirectory):
        self.dir = datasetDirectory
        self.instanceFiles = sorted(os.listdir(datasetDirectory))
    
    def __len__(self):
        return len(self.instanceFiles)
