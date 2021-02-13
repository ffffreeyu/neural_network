# 范例
# import torch.nn as nn
# import torch.nn.functional as F
#
# class LeNet(nn.Module): #定义一个类继承于nn.Module
#     def __init__(self): #初始化一些变量
#         super(LeNet, self).__init__()#解决多层继承中可能会出现的一些问题
#         self.conv1 = nn.Conv2d(3, 16, 5)#默认stride=1,padding=0  #输出16x28x28
#         self.pool1 = nn.MaxPool2d(2)#默认stride=kernel_size  #输出6x14x14
#         self.conv2 = nn.Conv2d(16, 32, 5) #输出32x10x10
#         self.pool2 = nn.MaxPool2d(2) #输出32x5x5
#         self.fc1 = nn.Linear(32*5*5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10) #10=分类任务个数
#
#     def forward(self, input):
#         x = self.conv1(input)
#         x = F.relu(x)
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = self.pool2(x)
#         #将特征矩阵展平为一维向量
#         x = x.view(-1, 32*5*5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)  #为什么没用softmax,...什么内部已经实现？
#         return x
#
# import torch
#
# input1 = torch.rand([32, 3, 32, 32])
# model = LeNet()
# output = model(input1)
# print(output)


import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, input):
        x = self.conv1(input)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = x.view(-1, 16*5*5)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

import torch

input1 = torch.rand([32, 3, 32, 32])
model = LeNet()
output = model(input1)
print(output)