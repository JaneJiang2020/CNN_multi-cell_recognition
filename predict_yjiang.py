# %% --------------------------------------- Imports -------------------------------------------------------------------
import numpy as np
import os
import cv2
import torch
import torch.nn as nn


# %% ----------- hyperparameters -----------------#
LR = 5e-2
N_EPOCHS = 30
BATCH_SIZE = 512
DROPOUT = 0.5

class CellCNN(nn.Module):
    def __init__(self):
        super(CellCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, (3, 3))  # output (n_examples, 16, 48, 48)
        self.convnorm1 = nn.BatchNorm2d(20)
        self.pool1 = nn.MaxPool2d((2, 2))  # output (n_examples, 16, 24, 24)
        self.conv2 = nn.Conv2d(20, 40, (3, 3))  # output (n_examples, 32, 22, 22)
        self.convnorm2 = nn.BatchNorm2d(40)
        self.pool2 = nn.AvgPool2d((2, 2))  # output (n_examples, 32, 11, 11)
        self.linear1 = nn.Linear(40*11*11, 250)  # input will be flattened to (n_examples, 32 * 5 * 5)
        self.linear1_bn = nn.BatchNorm1d(250)
        self.drop = nn.Dropout(DROPOUT)
        self.linear2 = nn.Linear(250, 7)
        self.act = torch.relu

    def forward(self, x):
        #x = x.reshape(1,-1)
        x = self.pool1(self.convnorm1(self.act(self.conv1(x.float()))))  #### first layer ###
        x = self.pool2(self.convnorm2(self.act(self.conv2(x.float()))))  #### second layer ###
        x = self.drop(self.linear1_bn(self.act(self.linear1(x.view(len(x), -1)))))  # fully connected layer ###
        return self.linear2(x)

def predict(paths):
    x = []
    for path in paths:
        x.append(cv2.resize(cv2.imread(path), (50, 50)))
    x_test = torch.FloatTensor(np.array(x))
    model=CellCNN()
    model.eval()
    model.load_state_dict(torch.load('model_yjiang.pt'))
    y_pred= model(x_test.permute(0, 3, 1, 2))
    m=nn.Sigmoid()
    y_pred_prob=m(y_pred)
    return y_pred_prob.detach()
