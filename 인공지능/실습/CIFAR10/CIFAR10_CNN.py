# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

#@title 모듈 임포트, 데이터셋 로드 및 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

# %%
#@title 모델 구조 정의

# model1 설계
class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        # Conv 레이어와 MaxPool 레이어 정의
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=5, padding=2)
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv2 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.flat = nn.Flatten()

        # Fully Connected 레이어 정의
        self.fc0 = nn.Linear(1682, 256)  # 입력 크기는 conv 레이어 출력 크기에 따라 계산됨
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool0(F.relu(self.conv0(x)))
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        x = self.flat(x)  # Flatten
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# model2 설계
class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        # Conv 레이어와 MaxPool 레이어 정의
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv1 = nn.Conv2d(in_channels=32, out_channels=24, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv2 = nn.Conv2d(in_channels=24, out_channels=16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.flat = nn.Flatten()

        # Fully Connected 레이어 정의
        self.fc0 = nn.Linear(13456, 256)  # 입력 크기는 conv 레이어 출력 크기에 따라 계산됨
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool0(F.relu(self.conv0(x)))
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        x = self.flat(x)  # Flatten
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# %%
#@title 모델 평가 함수 (정확도 계산)
def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# %%
#@title 모델1 훈련 및 평가

for epochs in tqdm(range(20, 31, 5), desc='Experiment: '):
    model = Model1().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

        for inputs, labels in progress_bar:
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=(running_loss / len(progress_bar)))
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")
            accuracy = evaluate(model, test_loader)
            print(f"Test Accuracy: {accuracy:.2f}%")
    print()

# %%
#@title 모델2 훈련 및 평가

for epochs in tqdm(range(20, 31, 5), desc='Experiment: '):
    model = Model2().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

        for inputs, labels in progress_bar:
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=(running_loss / len(progress_bar)))

        if (epoch+1)% 5 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")
            accuracy = evaluate(model, test_loader)
            print(f"Test Accuracy: {accuracy:.2f}%")
    print()


