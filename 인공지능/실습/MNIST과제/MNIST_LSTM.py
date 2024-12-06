# %%
import os
import time
import torch
from torch import nn
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# %%
from torchvision.datasets import MNIST

mnist_train = MNIST(root='./MNIST_data/', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = MNIST(root='./MNIST_data/', train=False, transform=transforms.ToTensor(), download=True)

BATCH_SIZE = 32
NUM_CLASS = 10
NUM_HIDDEN_LAYERS = 2
HIDDEN_STATE_SIZE = 256
num_epochs = 10

# %%
class MNISTLSTM(nn.Module):
    def __init__(self, num_hidden_layers, hidden_state_size):
        super(MNISTLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=49, hidden_size=hidden_state_size,
                            num_layers=num_hidden_layers, batch_first=True,
                            bidirectional=True)
        self.linear1 = nn.Linear(2*hidden_state_size, 128, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, NUM_CLASS, bias=True)

    def forward(self, X):

        x1, (_, _) = self.lstm(X)
        x2 = x1[:, 15]
        x3 = self.linear1(x2)
        x4 = self.relu(x3)
        x5 = self.linear2(x4)

        return x5

# %%
model = MNISTLSTM(NUM_HIDDEN_LAYERS, HIDDEN_STATE_SIZE)
model = model.to(device)

# %%
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# %%
train_loader = DataLoader(dataset=mnist_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=mnist_test, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# %%
train_batch = len(train_loader)
test_batch = len(test_loader)
print('Total number of batches for train and test =', train_batch, ' ', test_batch)

# %%
a_batch = torch.zeros((BATCH_SIZE, 4, 4, 7, 7))
b_batch = torch.zeros((BATCH_SIZE, 16, 49))

# %%
model.train()

from tqdm import tqdm

for epoch in tqdm(range(num_epochs), desc='Epochs'):
    total_loss = 0
    cnt_batch = 0

    for X, y in tqdm(train_loader, desc='Batch', leave=False):

        y = y.to(device)
        X = X.to(device)

        for r in range(4):
            for c in range(4):
                a_batch[:, r, c, :, :] = X[:, 0, r*7:(r+1)*7, c*7:(c+1)*7]

        t_batch = torch.reshape(a_batch, (BATCH_SIZE, 4, 4, 49))

        for r in range(4):
            for c in range(4):
                b_batch[:, r*4+c, :] = t_batch[:, r, c, :]
        b_batch = b_batch.float().to(device)

        optimizer.zero_grad()
        hypothesis = model(b_batch)

        loss = criterion(hypothesis, y)
        total_loss +=loss.item()

        loss.backward()
        optimizer.step()

        cnt_batch += 1

    avg_loss = total_loss/train_batch
    print(f'Epoch: {epoch+1}| Loss: {avg_loss:.4f}')

print('Training Finished')

# %%
model.eval()
with torch.no_grad():
    total_hit = 0
    total_examples = 0

    for X, y in tqdm(test_loader, desc='Batch', leave=False):
        y = y.to(device)
        X = X.to(device)
        for r in range(4):
            for c in range(4):
                a_batch[:, r, c, :, :] = X[:, 0, r*7:(r+1)*7, c*7:(c+1)*7]
        t_batch = torch.reshape(a_batch, (BATCH_SIZE, 4, 4, 49))

        for r in range(4):
            for c in range(4):
                b_batch[:, r*4+c, :] = t_batch[:, r, c, :]
        b_batch = b_batch.float().to(device)

        optimizer.zero_grad()
        prediction = model(b_batch)

        pr_label = torch.argmax(prediction, 1)
        correct = pr_label == y
        hit_cnt = correct.sum()
        total_hit += hit_cnt
        total_examples += BATCH_SIZE

acc = float(total_hit)/total_examples
print(f'Test Accuracy: {acc:.4f}')


