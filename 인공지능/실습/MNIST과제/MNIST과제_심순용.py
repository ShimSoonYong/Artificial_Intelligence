import torch
from torch import nn
from torch import optim
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Current Device: {device}')

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root='./data', train=True, download=True,
                               transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True,
                              transform=transform)

print(f'훈련 데이터의 차원: {train_dataset.data.shape}')

train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

# DataLoader의 첫 번째 배치 출력
first_batch = next(iter(train_loader))
print(type(first_batch))
print(len(first_batch))

# 훈련 루프
for epochs in tqdm(range(10, 40, 5), desc='Experitment'):
    NN_model = nn.Sequential(
    nn.Linear(in_features=28*28, out_features=256),
    nn.ReLU(),
    nn.Linear(in_features=256, out_features=128),
    nn.ReLU(),
    nn.Linear(in_features=128, out_features=64),
    nn.ReLU(),
    nn.Linear(in_features=64, out_features=10)
    ).to(device)

    # 손실함수와 최적화 기법 정의
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(NN_model.parameters(), lr=0.01)

    for epoch in tqdm(range(epochs), desc=f'Epochs = {epochs}'):
        NN_model.train()

        for i, (X, y) in tqdm(enumerate(train_loader), desc=f'Batch', leave=False):
            running_loss = 0
            optimizer.zero_grad()

            X = X.view(-1, 28*28).to(device)

            hypothesis = NN_model(X)
            loss = loss_fn(hypothesis, y.to(device))

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    print(f"Training Finished for {(epochs-5)//5} experiment.")

    # 평가 루프
    with torch.no_grad():
        correct_per_class = torch.zeros(10)
        total_per_class = torch.zeros(10)

        for X, y in test_loader:
            NN_model.eval()

            X = X.view(-1, 28*28).to(device)

            hypothesis = NN_model(X)

            predicted = torch.argmax(hypothesis, dim=1)

            true_labels = y

            for i in range(10):
                class_mask = (true_labels == i)

                # 맞춘 개수
                correct_per_class[i] += (predicted[class_mask] == i).sum().item()

                # 전체 개수
                total_per_class[i] += class_mask.sum().item()

    # 각 클래스에 대한 정확도 출력
    for i in range(10):
        if total_per_class[i] > 0:  # 해당 클래스에 대한 샘플이 있을 경우
            accuracy = correct_per_class[i] / total_per_class[i]
            print(f'Accuracy for class {i}: {accuracy * 100:.2f}%')
        else:
            print(f'No samples for class {i}')

    accuracy = correct_per_class / total_per_class
    # 가장 정확도가 높은 클래스 출력
    acc, label = torch.max(accuracy, dim=0)
    print(f'Max Accuracy: {acc}| The Class: {label}')

    # 가장 정확도가 낮은 클래스 출력
    acc, label = torch.min(accuracy, dim=0)
    print(f'Min Accuracy: {acc}| The Class: {label}')

    # 전체 정확도 출력
    acc = torch.mean(accuracy, dim=0)
    print(f'Total Accuracy: {acc}')
