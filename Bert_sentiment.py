import pandas as pd
import numpy as np
import torch
import numpy as np
from torch import nn
from transformers import BertTokenizer, BertModel,get_linear_schedule_with_warmup
from torch.optim import Adam
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained('chinese-bert-wwm-ext')

# def get_balance_corpus(corpus_size, corpus_pos, corpus_neg):
#     sample_size = corpus_size // 2
#     pd_corpus_balance = pd.concat([corpus_pos.sample(sample_size, replace=corpus_pos.shape[0]<sample_size), \
#                                    corpus_neg.sample(sample_size, replace=corpus_neg.shape[0]<sample_size)])
       
#     print('评论数目（总体）：%d' % pd_corpus_balance.shape[0])
#     print('评论数目（正向）：%d' % pd_corpus_balance[pd_corpus_balance.label==1].shape[0])
#     print('评论数目（负向）：%d' % pd_corpus_balance[pd_corpus_balance.label==0].shape[0])    
    
#     return pd_corpus_balance

def get_balance_corpus(corpus_size, corpus1, corpus2,corpus3,corpus4,corpus5,corpus6):
    sample_size = corpus_size // 6
    pd_corpus_balance = pd.concat([corpus1.sample(sample_size, replace=corpus1.shape[0] < sample_size), 
                                   corpus2.sample(sample_size, replace=corpus2.shape[0] < sample_size),
                                   corpus3.sample(sample_size, replace=corpus3.shape[0] < sample_size),
                                   corpus4.sample(sample_size, replace=corpus4.shape[0] < sample_size),
                                   corpus5.sample(sample_size, replace=corpus5.shape[0] < sample_size),
                                   corpus6.sample(sample_size, replace=corpus6.shape[0] < sample_size)])

    return pd_corpus_balance
def predict(text,model):
    input = tokenizer(text, 
                      padding='max_length', 
                      max_length = 512, 
                      truncation=True,
                      return_tensors="pt")

    mask = input['attention_mask']
    input_id = input['input_ids'].squeeze(1)

    output = model(input_id, mask)
    prediction = output.argmax(dim=1).item()
    if model.num_labels == 2:
        if prediction == 1:
            prediction = '正向'
        else:  
            prediction = '负向' 
        return prediction
    elif model.num_labels == 3:
        if prediction == 0:
            prediction = '负向'
        elif prediction == 1:
            prediction = '中性'
        else:
            prediction = '正向'
    else:
        if prediction == 0:
            prediction = '中性'
        elif prediction == 1:
            prediction = '喜好'
        elif prediction == 2:
            prediction = '悲伤'
        elif prediction == 3:
            prediction = '厌恶'
        elif prediction == 4:
            prediction = '愤怒'
        else:
            prediction = '高兴' 
    return prediction

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = df['label'].values
        self.reviews = [tokenizer(review, 
                                padding='max_length', 
                                max_length = 512, 
                                truncation=True,
                                return_tensors="pt") 
                      for review in df['review']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_reviews(self, idx):
        # Fetch a batch of inputs
        return self.reviews[idx]

    def __getitem__(self, idx):
        batch_reviews = self.get_batch_reviews(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_reviews, batch_y


class BertClassifier(nn.Module):
    def __init__(self, dropout=0.4,num_labels=2):
        super(BertClassifier, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('chinese-bert-wwm-ext')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, num_labels)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer


def train(model, train_data, val_data, learning_rate, epochs):
  # 通过Dataset类获取训练和验证集
    train, val = Dataset(train_data), Dataset(val_data)
    # DataLoader根据batch_size获取数据，训练时选择打乱样本
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2,shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)
  # 判断是否使用GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    total_step = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_step, num_training_steps=total_step)

    if use_cuda:
            model = model.cuda()
            criterion = criterion.cuda()
    # 开始进入训练循环
    for epoch_num in range(epochs):
      # 定义两个变量，用于存储训练集的准确率和损失
            total_acc_train = 0
            total_loss_train = 0
            model.train()

      # 进度条函数tqdm    
            for train_input, train_label in train_dataloader:
                train_label = train_label.type(torch.LongTensor).to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)
        # 通过模型得到输出
                optimizer.zero_grad()
                output = model(input_id, mask)
                # 计算损失
                batch_loss = criterion(output, train_label)
                total_loss_train += batch_loss.item()
                # 计算精度
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc
        # 模型更新
                batch_loss.backward()
                optimizer.step()
                scheduler.step()
            # ------ 验证模型 -----------
            # 定义两个变量，用于存储验证集的准确率和损失
            total_acc_val = 0
            total_loss_val = 0
            model.eval()

      # 不需要计算梯度
            with torch.no_grad():
                # 循环获取数据集，并用训练好的模型进行验证
                for val_input, val_label in val_dataloader:
          # 如果有GPU，则使用GPU，接下来的操作同训练
                    val_label = val_label.type(torch.LongTensor).to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)
  
                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label)
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
            
            print(
                f'''Epochs: {epoch_num + 1} 
              | Train Loss: {total_loss_train / len(train_data): .3f} 
              | Train Accuracy: {total_acc_train / len(train_data): .3f} 
              | Val Loss: {total_loss_val / len(val_data): .3f} 
              | Val Accuracy: {total_acc_val / len(val_data): .3f}''')    

if __name__ == '__main__':
    pd_all = pd.read_csv('ChnSentiCorp_htl_all.csv')
    pd_all.dropna(inplace= True)

    pd_positive = pd_all[pd_all.label==1]
    pd_negative = pd_all[pd_all.label==0]

    data = get_balance_corpus(4800, pd_positive, pd_negative)

    data_train, data_valid= np.split(data.sample(frac=1,random_state=42) ,[int(0.8*len(data))],axis=0)

    EPOCHS = 5
    model = BertClassifier()
    LR = 1e-6
    train(model, data_train, data_valid, LR, EPOCHS)
    torch.save(model.state_dict(), 'model.pth')