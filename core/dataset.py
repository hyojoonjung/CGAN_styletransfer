from torch.utils.data import Dataset
import torch
from transformers import T5Tokenizer

class MyDataset(Dataset):
    def __init__(self, data, t5_model_name='t5-base', max_length=512):
        self.texts = data['TEXT'].tolist()
        # labels is assumed to be a list of lists, where each inner list contains 5 values (one for each personality trait)
        self.labels = data[['OPN','CON','EXT','AGR','NEU']].values
        self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'text': encoding['input_ids'].squeeze(),  # removing the batch dimension
            'labels': torch.tensor(labels, dtype=torch.float)  # assuming labels are continuous values between 0 and 1
        }
