import torch.optim as optim
import torch.nn as nn
from model import Net
from data import custom_dataset
import torch

if __name__ == '__main__':
    dataset = custom_dataset()
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    net = Net()

    criterion = nn.L1Loss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    print('Start training ...')
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.float()
            labels = labels.float()
            labels = labels.unsqueeze(1)

            # Gradient parameter을 0으로 만듦
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            print(running_loss)
            running_loss=0.0

    print('Done training ...')