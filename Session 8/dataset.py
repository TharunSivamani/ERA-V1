import os

import numpy as np
import torch
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

class CIFAR10(object):

    mean = (0.49139968, 0.48215827, 0.44653124)
    std = (0.24703233, 0.24348505, 0.26158768)
    classes = None

    def __init__(self,batch_size = 128):

        self.batch_size = batch_size
        self.loader_kwargs = {'batch_size':batch_size , 'num_workers':os.cpu_count()-1,'pin_memory':True}
        self.train_loaders , self.test_loaders = self.get_loaders()

    def get_train_loader(self):

        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean , self.std)
        ])

        train_data = datasets.CIFAR10('./data' , train = True , download = True  , transform = train_transform)

        if self.classes is None:

            self.classes = {i:c for i,c in enumerate(train_data.classes)}
        
        self.train_loader = torch.utils.data.DataLoader(train_data , shuffle = True , **self.loader_kwargs)

        return self.train_loader
    
    def get_test_loader(self):

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean , self.std)
        ])

        test_data = datasets.CIFAR10('./data' , train = False , download = True  , transform = test_transform)
        
        self.test_loader = torch.utils.data.DataLoader(test_data , shuffle = False , **self.loader_kwargs)

        return self.test_loader

    
    def get_loaders(self):

        return self.get_train_loader() , self.get_test_loader()