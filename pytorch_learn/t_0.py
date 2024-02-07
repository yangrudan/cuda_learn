"""
Copyright (c) Zhejiang Lab. All right reserved.
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


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


model = NeuralNetwork().to(device)  # 调用刚定义的模型，将模型转到GPU（如果可用）
print(model)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    # 从数据加载器中读取batch（一次读取多少张，即批次数），X(图片数据)，y（图片真实标签）。
    for batch, (X, y) in enumerate(dataloader):
        # 将数据存到显卡
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)          # 得到预测的结果pred
        loss = loss_fn(pred, y)

        # Backpropagation 反向传播，更新模型参数
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 每训练100次，输出一次当前信息
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()      # 将模型转为验证模式
    test_loss, correct = 0, 0      # 初始化test_loss 和 correct， 用来统计每次的误差
    with torch.no_grad():      # 测试时模型参数不用更新，所以no_gard()
        for X, y in dataloader:  # 加载数据加载器，得到里面的X（图片数据）和y(真实标签）
            X, y = X.to(device), y.to(device)
            pred = model(X)  # 将图片传入到模型当中就，得到预测的值pred
            test_loss += loss_fn(pred, y).item()  # 计算预测值pred和真实值y的差距
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # 统计预测正确的个数
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


loss_fn = nn.CrossEntropyLoss()  # 定义损失函数，计算相差多少
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)  # 定义优化器，用来训练时候优化模型参数

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

# 保存训练好的模型
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")


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
