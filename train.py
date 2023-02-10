import torch.optim as optim
import torch.nn as nn
from model import UNet
from data import custom_dataset
import torch

if __name__ == '__main__':
    dataset = custom_dataset()
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    net = UNet(3, 1)

    #criterion = nn.L1Loss()
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss() if net.n_classes>1 else nn.BCEWithLogitsLoss()
    optimizer = optim.RMSprop(net.parameters(), lr=0.001, momentum=0.9)


    print("Model's state_dict:")
    for param_tensor in net.state_dict():
        print(param_tensor, "\t", net.state_dict()[param_tensor].size())

    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    print('Start training ...')
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.float()
            labels = labels.float()
            labels = labels.unsqueeze(1)

            print(inputs.shape)
            # Gradient parameter을 0으로 만듦
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            print("Running Loss: {}".format(running_loss))
            running_loss=0.0
        torch.save(net.state_dict(), './model_epoch{}.pth'.format(epoch))
        print("Saved model at epoch {}.".format(epoch))

    print('Done training ...')

    # 나중에 train할 때 model 로딩 방법:
    '''
        net = UNet(3, 1)
        net.load_state_dict(torch.load('./model_epoch2.pth'))
        net.eval()
    '''