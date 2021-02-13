
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
            with torch.no_grad(): #？？with是一个上下文管理器,防止内存爆炸
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