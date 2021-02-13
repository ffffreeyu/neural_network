# LeNet

![image-20210212191023914](C:\Users\86185\AppData\Roaming\Typora\typora-user-images\image-20210212191023914.png)

## **简介**

​		$LeNet$是$LeCun$于1998年提出的卷积神经网络，用于解决手写数字识别的视觉任务，定义了CNN的基本组件：卷积层，池化层，全连接层。如今广泛使用的是简化后的$LeNet-5$(5表示有5个层)，使用的激活函数为$ReLu$。

## **结构**

基本结构为

```mermaid
graph LR
a[conv1]-->b[pool1]-->c[conv2]-->d[pool2]-->e[full.connnection]
```



以图示为例解释

1. 输入是单通道的32x32大小的图像，用矩阵表示为[1,32,32]
2. $C_1，conv1$卷积层，6个卷积核，卷积核的大小为5x5，滑动步长为1，则输出6个特征图，特征图大小为($(32-5)/1 +1 =28$) 28x28 ,输出矩阵为[6,28,28]
3. $S_2, pool1$池化层，池化单元为2x2，步长为2，这是没有重叠的max pooling，池化操作后，深度不变，尺寸减半，输出矩阵为[6,14,14]
4. $C_3，conv2$卷积层，16个卷积核，卷积核大小为5x5，滑动步长为1，输出16个特征图，特征图大小为($(14-5)/1+1=10$),10x10,输出矩阵为[16,10,10]
5. $S_4,pool2$池化层，池化单元为2x2，步长为2，没有重叠的max pooling ,池化操作后，深度不变，尺寸减半，输出矩阵为[16,5,5]
6. $fc1$,全连接层，再接ReLu激活函数
7. $fc2$
8. 送入softmax分类

### **代码实现**

**数据集：CIFAR10**

$Cifar-10$ 由60000张32*32的 RGB 彩色图片构成，共10个分类。50000张训练，10000张测试（交叉验证）。这个数据集最大的特点在于将识别迁移到了普适物体，而且应用于多分类（姊妹数据集$Cifar-100$达到100类，$ILSVRC$比赛则是1000类）。

![img](https://img-blog.csdn.net/20170322103646555?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvZGlhbW9uam95X3pvbmU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

tip:

$pytorch$的通道顺序：[batch,channel,height,width]， batch对应输入图片的个数，CIFER-20为RGB图片channel=3

**module**

搭建模型：初始化函数，定义正向传播的过程

```python
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module): #定义一个类继承于nn.Module
    def __init__(self): #初始化一些变量
        super(LeNet, self).__init__()#解决多层继承中可能会出现的一些问题
        self.conv1 = nn.Conv2d(3, 16, 5)#默认stride=1,padding=0  #输出16x28x28
        self.pool1 = nn.MaxPool2d(2)#默认stride=kernel_size  #输出16x14x14
        self.conv2 = nn.Conv2d(6, 32, 5) #输出32x10x10
        self.pool2 = nn.MaxPool2d(2) #输出32x5x5
        self.fc1 = nn.Linear(32*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) #10=分类任务个数

    def forward(self, input):
        x = self.conv1(input)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        #将特征矩阵展平为一维向量
        x = x.view(-1, 32*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  #为什么没用softmax,...什么内部已经实现？
        return x

# import torch
# 
# input1 = torch.rand([32, 3, 32, 32])
# model = LeNet()
# output = model(input1)
# print(output)
```

1. super()可以避免基类的重复调用
2. nn.Conv2D

**train**

```python

import torch
import torchvision
from torchvision.transforms import transforms
from module import LeNet
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


#定义预处理  将图像改为tensor, 并且归一化处理 ；标准化过程
#input[channel] = (input[channel] - mean[channel]) / std[channel]
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(
        [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

# 下载数据集 第一次下载时将download=True,root下载目录，train=True意味着导入训练集，transform进行预处理
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, transform=transform, download=False)

#载入训练集，batch_size为一批次拿出多少张图片训练
# shuffle=True代表随机取出数据训练，windows下num_worker只能取0
train_loader = torch.utils.data.DataLoader(trainset, batch_size=36,
                                           shuffle=True, num_workers=0)
# 同理处理测试集 10000张测试图片
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, transform=transform, download=False)
test_loader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                           shuffle=True, num_workers=0)

# 转化成可迭代的迭代器 ？？ 获取到测试图像和对应标签的一批数据
test_data_iter = iter(test_loader)
test_img, test_label = test_data_iter.next()

# 导入标签 元祖类型
tags = ('plane', 'car', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# #测试一下
# def imshow(img):
#     img = img/2 +0.5  #反标准化处理
#     img_np = img.numpy()
#     plt.imshow(np.transpose(img_np, (1, 2, 0)))
#     plt.show()
# print(' '.join('%5s' % tags[test_label[j]] for j in range(4)))
# imshow(torchvision.utils.make_grid(test_img))

# 实例化模型
lenet = LeNet()
#定义损失函数
loss_function = nn.CrossEntropyLoss() # 这个函数包含了sofmax
#定义优化器 训练参数为lenet.parameters()即将lenet中所有可训练的参数来进行训练
# lr学习率
optimizer = optim.Adam(lenet.parameters(), lr=0.001)


# 训练过程
for epoch in range(5):#将训练集迭代5次
    running_loss = 0.0 #计算训练过程中的损失
    for step, data in enumerate(train_loader, start=0): #循环得到输入???
        inputs, labels = data

        #清除历史梯度 每一个batch 就用一次optimizer
        optimizer.zero_grad()

        # lenet处理
        outputs = lenet(inputs)
        loss = loss_function(outputs, labels)#损失函数
        loss.backward()#反向传播
        optimizer.step()#参数更新

        # 打印数据
        running_loss = loss.item()
        if step % 500 ==499: #每500步打印一次数据
            with torch.no_grad(): #？？with是一个上下文管理器
                outputs = lenet(test_img)
                # 在维度1上寻找最大值,[1]代表只需要index
                predict_y = torch.max(outputs, dim=1)[1]
                # 计算预测准确度，由于一直在tensor中计算，所以要用.item()转化为数值
                accuracy = (predict_y == test_label).sum().item()/test_label.size(0)

                print('[%d, %5d] train_loss: %.3f test_accuracy: %.3f'
                      % (epoch+1, step+1, running_loss/500, accuracy))
                running_loss = 0.0
print("train finished")

save_path = '/lenet.pth'
torch.save(lenet.state_dict(), save_path)
```

![image-20210125125235002](C:\Users\86185\AppData\Roaming\Typora\typora-user-images\image-20210125125235002.png)

测试

```python
import torch
import torchvision.transforms as transforms
from PIL import Image

from module import LeNet


def main():
    #预处理 将图片缩放到32x32,转化为tensor
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    tags = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    #实例化网络
    lenet = LeNet()
    lenet.load_state_dict(torch.load('Lenet.pth'))#载入数据

    img = Image.open('1.jpg')
    img = transform(img)  # [C, H, W]
    img = torch.unsqueeze(img, dim=0)  # [N, C, H, W]

    with torch.no_grad():
        outputs = lenet(img)
        predict = torch.max(outputs, dim=1)[1].data.numpy()
    print(tags[int(predict)])


if __name__ == '__main__':
    main()

```

