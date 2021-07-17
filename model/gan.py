import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

# defining the Generator
class ShapeGenerator(nn.Module):
    def __init__(self):
        super(ShapeGenerator, self).__init__()
        self.linear1 = nn.Linear(100, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, 100)
        self.linear4 = nn.Linear(100, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(100)
        self.bn3 = nn.BatchNorm1d(100)
        self.bn4 = nn.BatchNorm1d(100)

    def forward(self, x):
        x = F.relu(self.bn1(self.linear1(x)))
        x = F.relu(self.bn2(self.linear2(x)))
        x = F.relu(self.bn3(self.linear3(x)))
        x = torch.tanh(self.bn4(self.linear4(x)))

        return  x
        

# defining the Discriminator
class ShapeDiscriminator(nn.Module):
    def __init__(self):
        super(ShapeDiscriminator, self).__init__()
        self.linear1 = nn.Linear(100, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, 100)
        self.linear4 = nn.Linear(100, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(100)
        self.bn3 = nn.BatchNorm1d(100)
        self.bn4 = nn.BatchNorm1d(100)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.linear1(x)))
        f = x       # for intermediate feature
        x = F.leaky_relu(self.bn2(self.linear2(x)))
        x = F.leaky_relu(self.bn3(self.linear3(x)))
        x = torch.sigmoid(self.bn4(self.linear4(x)))

        return  x, f
