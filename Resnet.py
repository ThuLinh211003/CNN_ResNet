import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from fashion_mnist_loader import load_fashion_mnist_csv, get_transform
import time

class SimpleResNet(nn.Module):
    def __init__(self):
        super(SimpleResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_resnet():
    train_loader, _ = load_fashion_mnist_csv(batch_size=64)

    resnet_model = SimpleResNet()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet_model.parameters(), lr=0.001)

    num_epochs = 5
    start_time = time.time()
    for epoch in range(num_epochs):
        resnet_model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = resnet_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total Training Time: {elapsed_time} seconds")
if __name__ == "__main__":
    train_resnet()
