import torch
import numpy as np

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class ToxicCommentsDataset(Dataset):
    def __init__(self, data: np.ndarray, label: np.ndarray, tokenizer: BertTokenizer, max_token_len: int=512):
        self.tokenizer = tokenizer
        self.data = data
        self.label = label
        self.max_token_len = max_token_len
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index: int):
        comment_text = self.data[index]
        label = self.label[index]

        encoding = self.tokenizer.encode_plus(
            comment_text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        return {
            "comment_text": comment_text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.IntTensor(label)
        }


class ToxicCommentDataModule(pl.LightningDataModule):
    def __init__(self, train_data, train_label, test_data, test_label, tokenizer, batch_size=8, max_token_len=512):
        super().__init__()
        self.train = train_data
        self.train_label = train_label
        self.test = test_data
        self.test_label = test_label
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_token_len = max_token_len

    def setup(self, stage):
        self.train_dataset = ToxicCommentsDataset(
            self.train,
            self.train_label,
            self.tokenizer,
            self.max_token_len
        )
        self.test_dataset = ToxicCommentsDataset(
            self.test,
            self.test_label,
            self.tokenizer,
            self.max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.trat_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=2)
