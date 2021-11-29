from torch.utils.data import Dataset
from skimage import io
import torch
import os
import numpy as np
class MyDataset(Dataset):
    def __init__(self,input_dir):
        self.input_dir = input_dir
        self.images = [image for image in os.listdir(input_dir) if image.endswith('png')]

    def __getitem__(self, idx):

        image_path = os.path.join(self.input_dir,self.images[idx])
        
        image = io.imread(image_path)
        image_torch= torch.from_numpy(image[np.newaxis,:]/255).type(torch.float32) #adding a dimension of channel
        return image_torch, image_torch

    def __len__(self):
        return len(self.images)