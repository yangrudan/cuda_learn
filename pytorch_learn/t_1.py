"""
Copyright (c) Cookie Yang. All right reserved.
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()  # 碾平，将数据碾平为一维
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),  # 全连接层
            nn.ReLU(),  # ReLU激活函数
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)  # 随后x经过linear_relu_stack
        return logits


# 读取训练好的模型，加载训练好的参数
model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))

# 定义所有类别
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# 模型进入验证阶段
model.eval()

x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f"predicted: {predicted}, actual: {actual}")
