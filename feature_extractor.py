import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class TextDataset(Dataset):
    def __init__(self, texts):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(self.texts[idx], padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        return encoded

class FeatureExtractor(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:,0,:]

    def extract_features(self, dataloader):
        features = []
        for batch in dataloader:
            input_ids = batch['input_ids'].squeeze(1)
            attention_mask = batch['attention_mask'].squeeze(1)
            feature = self(input_ids, attention_mask)
            features.append(feature.cpu().numpy())
        return features
