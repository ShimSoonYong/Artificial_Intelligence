# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch import optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'

MSL = 100
NUM_CLASS = 50
SZ_WORD_VOCAB = 51459
DIM_EMBEDDING = 100
SZ_HIDDEN_STATE = 128
NUM_HIDDEN_LAYERS = 2
BATCH_SIZE = 256

# %%
def load_XY(path_index_file, num_line_to_read, MSL):
    list_X = []
    list_Y = []
    list_len = []
    line_cnt = 0

    # 파일 열기 및 오류 처리
    try:
        with open(path_index_file, 'r', encoding='utf-8') as fp:
            while True:
                # 단어와 품사 라인 읽기
                wordline = fp.readline()
                line_len = len(wordline)

                if line_len == 0:  # 파일의 끝
                    break
                elif line_len == 1:  # 빈 줄
                    continue

                posline = fp.readline()
                if not posline:
                    print("경고: 품사 라인이 부족합니다.")
                    break

                # 줄바꿈 제거 및 split 수정
                w_index = wordline.strip().split()
                p_index = posline.strip().split()

                # 오류 핸들링: 단어와 품사 길이 불일치 검사
                if len(w_index) != len(p_index):
                    print(f"경고: {line_cnt+1}번째 줄의 단어와 품사 길이가 일치하지 않습니다.")
                    continue

                line_cnt += 1
                X = []
                Y = []

                leng = len(w_index)
                if leng > MSL - 1:
                    leng = MSL - 1
                for i in range(leng):
                    X.append(int(w_index[i]))
                    Y.append(int(p_index[i]))

                # MSL 크기 맞추기
                for i in range(leng, MSL):
                    X.append(0)
                    Y.append(0)

                list_X.append(X)
                list_Y.append(Y)
                list_len.append(leng)

                if line_cnt >= num_line_to_read:
                    break
    except FileNotFoundError:
        print(f"오류: 파일 '{path_index_file}'을(를) 찾을 수 없습니다.")
        return None, None, None
    except Exception as e:
        print(f"오류 발생: {e}")
        return None, None, None

    return list_X, list_Y, list_len

# %%
print('reading trian data'.upper())
x_train, y_train, leng_train = load_XY('/content/drive/MyDrive/Colab Notebooks/2024_2_인공지능_과제/POSdata/all_index_sentences_train.txt',
                                       20000, MSL)

print('num of sentences:'.upper(), len(x_train), len(y_train), len(leng_train))
print(x_train[0])

# %%
train_X = torch.LongTensor(x_train)
train_Y = torch.LongTensor(y_train)
train_Leng = torch.IntTensor(leng_train)

# %%
print('reading test data'.upper())
x_test, y_test, leng_test = load_XY('/content/drive/MyDrive/Colab Notebooks/2024_2_인공지능_과제/POSdata/all_index_sentences_test.txt',
                                      2000, MSL)
print('reading done'.upper())
test_X = torch.LongTensor(x_test)
test_Y = torch.LongTensor(y_test)
test_Leng = torch.IntTensor(leng_test)

# %%
print('reading validation data'.upper())
x_val, y_val, leng_val = load_XY('/content/drive/MyDrive/Colab Notebooks/2024_2_인공지능_과제/POSdata/all_index_sentences_validation.txt',
                                      4000, MSL)
print('reading done'.upper())
val_X = torch.LongTensor(x_val)
val_Y = torch.LongTensor(y_val)
val_Leng = torch.IntTensor(leng_val)

# %%
# 양방향 LSTM 위에 3개의 FF 층을 올린다.
class POS_model(nn.Module):
    def __init__(self, token_vocab_size, dim_embedding, num_hidden_layers, hidden_state_size):
        super(POS_model, self).__init__()
        self.embedding = nn.Embedding(token_vocab_size, dim_embedding, padding_idx=0)

        self.lstm = nn.LSTM(input_size=dim_embedding, hidden_size=hidden_state_size,
                            num_layers=num_hidden_layers, batch_first=True, bidirectional=True)

        self.linear1 = nn.Linear(2*hidden_state_size, 512, bias=True)
        self.relu = nn.ReLU()

        self.linear2 = nn.Linear(512, 256, bias=True)

        self.linear3 = nn.Linear(256, NUM_CLASS)    # NUM_CLASS is the number of classes of POS

    def forward(self, X):
        # X: shape (batch, msl) where each example is a list of token ids of length msl.
        x1 = self.embedding(X)     # output shape is (batch, msl, dim_embedding)

        x2, _ = self.lstm(x1)      # LSTM을 3계 시퀀스로 통과한 상태
        # x2의 형태는 2계 방향이 추가된 최종 상태: (batch, MSL, hidden_state)

        x3 = self.linear1(x2)      # FF3 : output shape is (batch, msl, 512).
        x4 = self.relu(x3)

        x5 = self.linear2(x4)      # FF3 : output shape is (batch, msl, 256).
        x6 = self.relu(x5)

        x7 = self.linear3(x6)      # FF3 : output shape is (batch, msl, num_class).
        return x7

# %%
train_data = TensorDataset(train_X, train_Y, train_Leng)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler,
                              batch_size=BATCH_SIZE, drop_last=True)

test_data = TensorDataset(test_X, test_Y, test_Leng)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler,
                             batch_size=BATCH_SIZE, drop_last=True)

val_data = TensorDataset(val_X, val_Y, val_Leng)
val_sampler = RandomSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler,
                             batch_size=BATCH_SIZE, drop_last=True)

# %%
from tqdm.notebook import tqdm

def train_eval(num_EPOCHS):
    global hold_acc

    # 매 실험마다 모델 초기화
    model = POS_model(SZ_WORD_VOCAB, DIM_EMBEDDING, NUM_HIDDEN_LAYERS, SZ_HIDDEN_STATE).to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction='mean').to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999),
                            eps=1e-8, weight_decay=0.01)

    # 주어진 에포크에 따라 훈련
    for epoch in tqdm(range(num_EPOCHS), 'Epoch: ', leave=False):
        model.train()
        total_loss = 0

        for i, batch in tqdm(enumerate(train_dataloader), 'Batch: ', leave=False):
            batch = tuple(r.to(device) for r in batch)
            X, Y, Leng = batch

            optimizer.zero_grad()

            logits = model(X)

            loss = loss_fn(logits.view(-1, NUM_CLASS), Y.view(-1))
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 10 == 0:
            # 평균 훈련 손실 계산
            avg_loss = total_loss / len(train_dataloader)
            print(f'Epoch {epoch+1}/{num_EPOCHS}, Loss: {avg_loss:.4f}', end='')

            # 중간 평가
            total_word_cnt = 0
            total_success_cnt = 0
            model.eval()
            with torch.no_grad():
                for k, batch in tqdm(enumerate(val_dataloader), 'Batch: ', leave=False):
                    batch = tuple(r.to(device) for r in batch)
                    X, Y, Leng = batch

                    logits = model(X)

                    pred_label_batch = torch.argmax(logits, dim=2)

                    for i in range(len(Y)):
                        target_label_seq = Y[i]
                        pred_label_seq = pred_label_batch[i]
                        leng = Leng[i].item()

                        match_cnt = 0
                        for j in range(leng):
                            if pred_label_seq[j] == target_label_seq[j]:
                                match_cnt += 1

                        total_word_cnt += leng
                        total_success_cnt += match_cnt
                acc = float(total_success_cnt) / float(total_word_cnt)
                print(f'Validation accuracy after epoch {epoch+1}: {acc}')

                # 성능이 좋은 모델 상태를 10 에포크마다 자동 저장
                if acc > hold_acc:
                    hold_acc = acc
                    torch.save(model.state_dict(), 'model_checkpoint.pth')
                    print('Model Saved')

# %%
hold_acc = 0
for num_EPOCHS in [10, 20, 30, 40]:
    print(f'num_EPOCHS: {num_EPOCHS}'.upper())
    train_eval(num_EPOCHS)
    print()

# %%
model = POS_model(SZ_WORD_VOCAB, DIM_EMBEDDING, NUM_HIDDEN_LAYERS, SZ_HIDDEN_STATE).to(device)
model.load_state_dict(torch.load('model_checkpoint.pth', weights_only=True))


total_word_cnt = 0
total_success_cnt = 0
model.eval()
with torch.no_grad():
    for k, batch in tqdm(enumerate(test_dataloader), 'Batch: ', leave=False):
        batch = tuple(r.to(device) for r in batch)
        X, Y, Leng = batch

        logits = model(X)

        pred_label_batch = torch.argmax(logits, dim=2)

        for i in range(len(Y)):
            target_label_seq = Y[i]
            pred_label_seq = pred_label_batch[i]
            leng = Leng[i].item()

            match_cnt = 0
            for j in range(leng):
                if pred_label_seq[j] == target_label_seq[j]:
                    match_cnt += 1

            total_word_cnt += leng
            total_success_cnt += match_cnt
    acc = float(total_success_cnt) / float(total_word_cnt)
    print(f'Best Test accuracy: {acc}')


