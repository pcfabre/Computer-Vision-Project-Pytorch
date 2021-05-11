## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 34, 5)

        self.pool1 = nn.MaxPool2d(2,2)
        
        self.fc1_dropout = nn.Dropout2d(p=0.1)

        self.conv2 = nn.Conv2d(34,68,5)

        self.pool2 = nn.MaxPool2d(2,2)

        self.fc2_dropout = nn.Dropout2d(p=0.15)
        self.conv3 = nn.Conv2d(68,136,5)

        self.pool3 = nn.MaxPool2d(2,2)

        self.fc3_dropout = nn.Dropout2d(p=0.2)
        self.conv4 = nn.Conv2d(136,272,4)

        self.pool4 = nn.MaxPool2d(2,2)

        self.fc4_dropout = nn.Dropout2d(p=0.25)
        self.conv5 = nn.Conv2d(272,544,4)

        self.pool5 = nn.MaxPool2d(2,2)

        self.fc5_dropout = nn.Dropout2d(p=0.3)
        
        self.fc1 = nn.Linear(4896,2000)
        self.fc6_dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(2000,1000)
        self.fc7_dropout = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(1000, 500)
        self.fc8_dropout = nn.Dropout(p=0.2)
        self.fc4 = nn.Linear(500,136)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model

        x = self.fc1_dropout(self.pool1(F.elu(self.conv1(x))))
        x = self.fc2_dropout(self.pool2(F.elu(self.conv2(x))))
        x = self.fc3_dropout(self.pool3(F.elu(self.conv3(x))))
        x = self.fc4_dropout(self.pool4(F.elu(self.conv4(x))))
        x = self.fc5_dropout(self.pool5(F.elu(self.conv5(x))))
        #print(str(x.shape))
        #flatten
        x = x.view(x.size(0), -1) 
        #fully connected
        x = F.elu(self.fc1(x))
        x = self.fc6_dropout(x)
        x = F.elu(self.fc2(x))
        x = self.fc7_dropout(x)
        x = F.elu(self.fc3(x))
        x = self.fc8_dropout(x)
        x = self.fc4(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x
