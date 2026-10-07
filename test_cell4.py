import os, re, random, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Reproducibility
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

MAXLEN    = 50
BATCH     = 64

URL = ("https://raw.githubusercontent.com/mohitgupta-omg/"
       "Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv")
df = pd.read_csv(URL, encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "text"]
df = df.dropna().reset_index(drop=True)

def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()

df["tokens"]    = df["text"].apply(clean)
df["label_idx"] = df["label"].map({"ham": 0, "spam": 1})

n   = len(df)
idx = list(range(n)); random.shuffle(idx)
split = int(0.8 * n)
train_df = df.iloc[idx[:split]].reset_index(drop=True)
test_df  = df.iloc[idx[split:]].reset_index(drop=True)

counter   = Counter(tok for toks in train_df["tokens"] for tok in toks)
vocab     = {"<PAD>": 0, "<UNK>": 1}
for word, _ in counter.most_common():
    vocab[word] = len(vocab)
VOCAB_SIZE = len(vocab)

def encode(tokens):
    return [vocab.get(t, 1) for t in tokens[:MAXLEN]]

class SpamDataset(Dataset):
    def __init__(self, dataframe):
        self.texts  = [torch.tensor(encode(t), dtype=torch.long)
                       for t in dataframe["tokens"]]
        self.labels = torch.tensor(dataframe["label_idx"].values, dtype=torch.long)
    def __len__(self):        return len(self.labels)
    def __getitem__(self, i): return self.texts[i], self.labels[i]

def collate_fn(batch):
    texts, labels = zip(*batch)
    padded = pad_sequence(texts, batch_first=True, padding_value=0)
    return padded, torch.stack(labels)

train_ds = SpamDataset(train_df)
test_ds  = SpamDataset(test_df)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  collate_fn=collate_fn)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, collate_fn=collate_fn)

xb, yb = next(iter(train_loader))
print(f"Input batch shape  : {tuple(xb.shape)}")
print(f"Labels batch shape : {tuple(yb.shape)}")
