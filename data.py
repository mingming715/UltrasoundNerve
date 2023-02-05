import torch
import os
import glob
import cv2
import natsort
#import numpy as np

class custom_dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.dir_path = "./data/train/"
        file_list = os.listdir(self.dir_path)
        
        self.input_list = []
        self.gt_list = []

        for image in file_list:
            if image[-5] == 'k':
                self.gt_list.append(image)
            else:
                self.input_list.append(image)

        self.gt_list = natsort.natsorted(self.gt_list)
        self.input_list = natsort.natsorted(self.input_list)
        
 #       print(self.input_list[:10])
 #       print(self.gt_list[:10])
    
    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        input_name = self.input_list[idx]
        gt_name = self.gt_list[idx]

        inp = cv2.imread(self.dir_path + input_name)/255    #Normalization 
        gt = cv2.imread(self.dir_path + gt_name)/255        #Normalization

        inp = inp.transpose(2, 0, 1)
        gt = gt.mean(-1)

        
        return inp, gt

if __name__ == "__main__":
    dataset = custom_dataset()
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    for item in trainloader:
        print(item[1].shape)