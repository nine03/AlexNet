import torch
from torch import nn
import numpy as np
from torch.autograd import Variable


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()

        # The first layer is 5*5 convolution, the input channels are 3, the output channels are 64, the step size is 1, and there is no padding
        # The first parameter of conv2d is the input channel, the second parameter is the output channel, and the third parameter is the convolution kernel size
        # The parameter of relu is inplace. True means to modify the input directly, and false means to create a new object to modify
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.ReLU()
        )

        # The second layer is 3*3 pooling, with a step size of 2 and no padding
        self.max_pool1 = nn.MaxPool2d(3, 2)

        # The third layer is 5*5 convolution. The input channels are 64 and the output channels are 64. There is no padding
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 5, 1),
            nn.ReLU(True)
        )

        # The fourth layer is 393 pooling, with a step size of 2 and no padding
        self.max_pool2 = nn.MaxPool2d(3, 2)

        # The fifth layer is the full connection layer. The input is 1204 and the output is 384
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 384),
            nn.ReLU(True)
        )

        # The sixth layer is the full connection layer. The input is 384 and the output is 192
        self.fc2 = nn.Sequential(
            nn.Linear(384, 192),
            nn.ReLU(True)
        )

        # The seventh layer is the full connection layer. The input is 192 and the output is 10
        self.fc3 = nn.Linear(192, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)

        # Flatten the picture matrix
        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

alexnet = AlexNet()
print(alexnet)

input_demo = Variable(torch.zeros(1, 3, 32, 32))
output_demo = alexnet(input_demo)
print(output_demo.shape)

