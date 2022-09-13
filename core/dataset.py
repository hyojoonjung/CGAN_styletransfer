from lib2to3.pgen2 import token
import torch
from sklearn.preprocessing import MinMaxScaler
from utils.config import ModelConfig
from utils.utils import pandora_label

mconfig = ModelConfig()

def read_essay_split(df, dataset):
    texts = []
    labels = []
    for i in range(len(df)):
        label = []
        texts.append(df['TEXT'].iloc[i])

        if dataset == 'essay':
            label.append(float(1) if df['OPN'].iloc[i] == 'y' else float(0))
            label.append(float(1) if df['CON'].iloc[i] == 'y' else float(0))
            label.append(float(1) if df['EXT'].iloc[i] == 'y' else float(0))
            label.append(float(1) if df['AGR'].iloc[i] == 'y' else float(0))
            label.append(float(1) if df['NEU'].iloc[i] == 'y' else float(0))

        elif dataset == 'pandora':
            label.append(float(df['OPN'].iloc[i]))
            label.append(float(df['CON'].iloc[i]))
            label.append(float(df['EXT'].iloc[i]))
            label.append(float(df['AGR'].iloc[i]))
            label.append(float(df['NEU'].iloc[i]))

        labels.append(label)

    return texts, labels

class OCEANDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, dataset):
        self.texts = texts
        self.tokenizer = tokenizer
        if dataset == 'pandora':
            scaler = MinMaxScaler()
            self.labels = pandora_label(scaler.fit_transform(labels))
        
        elif dataset == 'essay':
            self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encodings = self.tokenizer.encode_plus(self.texts[idx], add_special_tokens=True, max_length=mconfig.max_seq_len, padding='max_length', truncation=True)
        item = {key: torch.tensor(val, dtype=torch.long) for key, val in encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item