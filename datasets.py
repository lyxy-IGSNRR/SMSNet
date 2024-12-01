import glob
import random
import os
import numpy as np
from os import listdir
from os.path import join
import scipy.io as scio
import h5py,timeit,time

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import math
import random


def read_data(datafolder):
    file_num=len(os.listdir(datafolder))
    names = os.listdir(datafolder)
    #获取矩阵大小
    file=h5py.File(datafolder+'/'+names[0],'r')
    data_x=file['dataX'][:]
    data_y=file['dataY'][:]
    data_x=np.transpose(data_x,(3,2,1,0))
    data_y=np.transpose(data_y,(3,2,1,0))
    
    data_x_all = np.zeros((file_num,data_x.shape[0],data_x.shape[1],data_x.shape[2],data_x.shape[3]))
    data_y_all = np.zeros((file_num,data_y.shape[0],data_y.shape[1],data_y.shape[2],data_y.shape[3]))
    
    data_count=0
    
    for parents,adds,filenames in os.walk(datafolder):
        for filename in filenames:
            file=h5py.File(os.path.join(parents,filename),'r')
            data_x=file['dataX'][:]
            data_y=file['dataY'][:]
            data_x=np.transpose(data_x,(3,2,1,0))
            data_y=np.transpose(data_y,(3,2,1,0))
            data_x_all[data_count,:,:,:,:]=data_x
            data_y_all[data_count,:,:,:,:]=data_y
            data_count=data_count+1

    return (data_x_all,data_y_all)



class ImageDataset_array(Dataset):
    def __init__(self, data_x1_all,data_x2_all,data_y_all):

        self.data_x1_all = data_x1_all
        self.data_x2_all = data_x2_all
        self.data_y_all = data_y_all
        

    def __getitem__(self, index):
        '''
        img = Image.open(self.files[index % len(self.files)])
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)
        '''
        Ix1 = self.data_x1_all[index,:,:,:,:]
        Ix2 = self.data_x2_all[index,:,:,:,:]
        Iy = self.data_y_all[index,:,:,:,:]
        
        
        Ix1 = torch.from_numpy(Ix1).float()
        Ix2 = torch.from_numpy(Ix2).float()
        Iy = torch.from_numpy(Iy).float()

        return {"Ix1": Ix1, "Ix2": Ix2, "Iy": Iy}

    def __len__(self):
        return self.data_y_all.shape[0]
    
    
    
class ImageDataset_array_test(Dataset):
    def __init__(self, data_x1_all,data_x2_all):

        self.data_x1_all = data_x1_all
        self.data_x2_all = data_x2_all
        

    def __getitem__(self, index):
        '''
        img = Image.open(self.files[index % len(self.files)])
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)
        '''
        Ix1 = self.data_x1_all[index,:,:,:,:]
        Ix2 = self.data_x2_all[index,:,:,:,:]
        
        
        Ix1 = torch.from_numpy(Ix1).float()
        Ix2 = torch.from_numpy(Ix2).float()

        return {"Ix1": Ix1, "Ix2": Ix2}

    def __len__(self):
        return self.data_x1_all.shape[0]