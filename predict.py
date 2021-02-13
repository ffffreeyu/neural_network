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
