import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1, 1)
        self.conv2 = nn.Conv2d(6, 6, 3, 1, 1)
        self.conv3 = nn.Conv2d(6, 1, 3, 1, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.sigmoid(self.conv3(x)) #Sigmoid Activation Function

        return x

if __name__ == "__main__":
    print('model')
    net = Net()
    a = torch.zeros([1, 3, 420, 580])
    out = net(a)
    print(out.shape)