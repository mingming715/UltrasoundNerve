import torch
import torch.optim as optim
import torch.nn as nn
from model import UNet
from data import custom_dataset
import os
import cv2
import natsort
import PIL
import torchvision.transforms as transforms


if __name__ == '__main__':
    dir_path = "./data/test/"
    file_list = os.listdir(dir_path)

    input_list = []
    for image in file_list:
        input_list.append(image)

    input_list = natsort.natsorted(input_list)

    input_name = input_list[0]
    print(input_name)
    inp = cv2.imread(dir_path + input_name)/255

    inp = inp.transpose(2, 0, 1)
    inputs = torch.from_numpy(inp)
    inputs = inputs.float()
    inputs = inputs.unsqueeze(0)

    net = UNet(3, 1)
    net.load_state_dict(torch.load('./model_epoch1.pth'))
    net.eval()

    with torch.no_grad():
        output = net(inputs)

    output = output.squeeze(0)

    print("Max of Tensor:")
    print(torch.max(output))
    
    print("Min of Tensor:")
    print(torch.min(output))

    print("Shape:")
    print(output.shape)
    
    num_of_arrays, num_of_rows, num_of_cols = output.shape
    for i in range(num_of_arrays):
        for j in range(num_of_rows):
            for k in range(num_of_cols):
                if output[i][j][k] < 0.:
                    output[i][j][k] = 0.
                elif output[i][j][j] > 1.:
                    output[i][j][k] = 1.
                else:
                    output[i][j][k] = 1.

    print(output)



    tf = transforms.ToPILImage()
    output = tf(output)

    output.show()