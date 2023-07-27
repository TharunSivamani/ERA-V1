import torch                 
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.utils import denormalize
import cv2

import albumentations as A 
from albumentations.pytorch.transforms import ToTensorV2

class a_train_transform():

    def __init__(self):
        self.albumentations_transform = A.Compose([
            A.Normalize(mean=[0.4914, 0.4822, 0.4471],std=[0.2469, 0.2433, 0.2615],always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.PadIfNeeded(40, 40, p=1),
            A.RandomCrop(32, 32, p=1),
            A.PadIfNeeded(64, 64, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
            A.CoarseDropout(max_holes=1, max_height=16, max_width=16, fill_value=0, p=1),
            A.CenterCrop(32, 32, p=1),
            ToTensorV2()
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.albumentations_transform(image=img)['image']
        return img 


class a_test_transform():

    def __init__(self):
        self.albumentations_transform = A.Compose([
            A.Normalize(mean=[0.4914, 0.4822, 0.4471],std=[0.2469, 0.2433, 0.2615],always_apply=True),
            ToTensorV2()
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.albumentations_transform(image=img)['image']
        return img 
    

class CIFAR10(object):

    mean = (0.49139968, 0.48215827, 0.44653124)
    std = (0.24703233, 0.24348505, 0.26158768)
    classes = None

    def __init__(self, batch_size=1):

        self.batch_size = batch_size
        self.loader_kwargs = {'batch_size':batch_size,'num_workers':os.cpu_count()-1,'pin_memory':True}
        self.train_loaders , self.test_loaders = self.get_loaders()
        
    def get_train_loader(self):
        train_data = datasets.CIFAR10('data',train=True,download=True,transform=a_train_transform())

        if self.classes is None:
            self.classes = {i:c for i,c in enumerate(train_data.classes)}
        
        self.train_loader = torch.utils.data.DataLoader(train_data , shuffle = True , **self.loader_kwargs)
        return self.train_loader

    def get_test_loader(self):
        test_data = datasets.CIFAR10('data',train=False,download=True,transform=a_test_transform())

        if self.classes is None:
            self.classes = {i:c for i,c in enumerate(test_data.classes)}
        
        self.test_loader = torch.utils.data.DataLoader(test_data , shuffle = True , **self.loader_kwargs)
        return self.test_loader
    
    def get_loaders(self):
        return self.get_train_loader() , self.get_test_loader()
    
    def denormalise(self, tensor):
        result = torch.tensor(tensor, requires_grad=False)
        
        for t, m, s in zip(result, self.mean, self.std):
            t.mul_(s).add_(m)
        return result
    
    def show_transform(self, img):
        img = self.denormalise(img)
        if len(self.mean) == 3:
            return img.permute(1, 2, 0)
        else:
            return img.squeeze(0)
    
    def return_classes(self):
        return self.classes
    
    





    


