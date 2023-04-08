from torch.utils.data import Dataset
import random
import h5py
import torch
import numpy as np
from scipy.io import loadmat

from torchvision import transforms
transformations = transforms.Compose([transforms.ToTensor()]) #It also scales to 0-1 by dividing by 255.
        
class MRAdatasetH5NoScale(Dataset): #Scaling: allPatchesOfAllSubjects =/ max(allPatchesOfAllSubjects )

    def __init__(self, filePath, transform = transformations, prefetch = True, dim = 2, device=None ):
        super(Dataset, self).__init__()
        self.h5f = h5py.File(filePath, 'r')
        self.keys = list(self.h5f.keys())
        
        self.prefetch = prefetch
        if device is None:
            device = torch.device('cuda')
        if (self.prefetch):
            self.data = torch.zeros((len(self.keys), 1, *(np.array(self.h5f[self.keys[0]])).shape[-dim:]))    
            for ii in range(len(self.keys)):
                self.data[ii] = torch.tensor(np.array(self.h5f[self.keys[ii]]))
            self.data = self.data.to(device).float() / self.data.float().max()
            self.h5f.close()
        else:
            self.transform = transform
            random.shuffle(self.keys)
        
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        theIndex = index % len(self.keys)
        if self.prefetch:
            return self.data[theIndex]
        else:    
            key = self.keys[theIndex]
            data = np.array(self.h5f[key])
            if self.transform:
                data = self.transform(data)
            return data
    def openFile(self, filePath):
        self.h5f = h5py.File(filePath, 'r')
    def closeFile(self):
        self.h5f.close()

