import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. MNIST 데이터셋을 로드하고 전처리 변환 설정
transform = transforms.Compose([
    transforms.ToTensor(),             # 이미지를 텐서로 변환
    transforms.Normalize((0.5,), (0.5,))  # 평균과 표준편차로 정규화
])

train = MNIST(root='./data', download=True, train=True, transform=transform)
test = MNIST(root='./data', download=True, train=False, transform=transform)

train_loader = DataLoader(train, batch_size=16, shuffle=True)
test_loader = DataLoader(test, batch_size=16, shuffle=True)

train_loader.batch_size

class CNN_model(nn.Module):
    def __init__(self):
        super(CNN_model, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 6, 5), nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(nn.Conv2d(6, 16, 5), nn.MaxPool2d(2, 2))
        self.flat = nn.Flatten()
        self.linear = nn.Sequential(nn.Linear(256, 120),
                                    nn.Linear(120, 84),
                                    nn.Linear(84, 10))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flat(x)
        x = self.linear(x)
        return x

from tqdm.notebook import tqdm

k = 0
for epochs in tqdm(range(5, 35, 5), desc='Experiments: '):
    model = CNN_model().to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    model.train()
    for epoch in tqdm(range(epochs), desc="Epoch: ", leave=False):
        for batch, (X, y) in tqdm(enumerate(train_loader), desc='Batch: ', leave=False):
            pred = model(X.to(device))
            loss = loss_fn(pred, y.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    k +=1
    print(f'Training Finished for {k} experiment.')
    model.eval()

    total_correct = 0
    class_correct = [0 for _ in range(10)]
    class_total = [0 for _ in range(10)]

    with torch.no_grad():
        for X, y in test_loader:
            x = X.to(device)
            y = y.to(device)
            pred = model(x)

            # 전체 정확도 계산
            total_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            # 클래스별 정확도 계산
            for i in range(10):
                # 실제 레이블이 i인 샘플 개수
                class_total[i] += (y == i).sum().item()
                # 예측과 레이블이 i로 일치하는 경우의 수
                class_correct[i] += ((pred.argmax(1) == y) & (y == i)).sum().item()

        # 총 테스트 데이터 수
        total_samples = len(test_loader.dataset)

        print(f'Total Accuracy: {total_correct / total_samples * 100:.2f}%')

        for i in range(10):
            if class_total[i] > 0:
                accuracy = class_correct[i] / class_total[i] * 100
                print(f'Accuracy of {i}: {accuracy:.2f}%')
            else:
                print(f'Accuracy of {i}: No samples')


