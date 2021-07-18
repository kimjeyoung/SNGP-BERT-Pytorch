import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from models.tokenizer import BertTokenizer


class DatasetLoader(Dataset):
    def __init__(self,
                 data_dir: str,
                 vocab_path: str,
                 max_len: int = 32,
                 train_or_test: str = "train",
                 model_type: str = "bert-base-uncased"):
        if train_or_test == 'train':
            df = pd.read_csv(os.path.join(data_dir, "train.csv"))
        elif train_or_test == 'val':
            df = pd.read_csv(os.path.join(data_dir, "val.csv"))
        else:
            df = pd.read_csv(os.path.join(data_dir, "test.csv"))
            df_ood = pd.read_csv(os.path.join(data_dir, "test_ood.csv"))
            df = df.append(df_ood)

        self.max_len = max_len
        self.x_data = df['text'].values
        self.y_data = df['intent'].values
        self.tokenizer = BertTokenizer(vocab_path, do_lower_case=False if 'uncased' in model_type else True)

        # num_classes is number of valid intents plus out-of-scope intent
        self.num_classes = len(np.unique(self.y_data)) + 1 if train_or_test in ['train', 'val'] else len(np.unique(self.y_data))

    @staticmethod
    def list2tensor(x):
        return torch.tensor(x).to(torch.long)

    def convert_text2tensor(self, x_sample):
        tokens = self.tokenizer.tokenize(x_sample)

        # considering [CLS] and [SEP]
        if len(tokens) > self.max_len - 2:
            tokens = tokens[:(self.max_len - 2)]

        input_tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        input_attns = [1] * len(input_ids)
        input_segs = [0] * len(input_ids)

        # Zero-pad up to the sequence length.
        pad_len = self.max_len - len(input_ids)
        padding = [0] * pad_len
        input_ids += padding
        input_attns += padding
        input_segs += padding

        input_ids = self.list2tensor(input_ids)
        input_attns = self.list2tensor(input_attns)
        input_segs = self.list2tensor(input_segs)

        return input_ids, input_segs, input_attns

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x_sample = self.x_data[idx]
        x_ids, x_segs, x_attns = self.convert_text2tensor(x_sample)
        y_sample = torch.tensor(self.y_data[idx]).to(torch.long)
        return x_ids, x_segs, x_attns, y_sample
