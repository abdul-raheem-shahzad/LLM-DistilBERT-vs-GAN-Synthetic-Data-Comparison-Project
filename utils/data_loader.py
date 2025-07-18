"""
Data loading utilities for IMDB dataset
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab
import numpy as np
from typing import List, Tuple, Iterator
import sys
import os
from transformers import AutoTokenizer
from datasets import load_dataset
from functools import partial
from utils.config import *

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class IMDBDataset(Dataset):
    """Custom Dataset for IMDB reviews"""
    
    def __init__(self, data_iter, vocab, max_length=MAX_SEQUENCE_LENGTH):
        self.data = []
        self.vocab = vocab
        self.max_length = max_length
        self.tokenizer = get_tokenizer('basic_english')
        
        # Convert iterator to list
        for label, text in data_iter:
            # Convert label to binary (0 for negative, 1 for positive)
            label = 1 if label == 'pos' else 0
            self.data.append((text, label))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text, label = self.data[idx]
        
        # Tokenize and convert to indices
        tokens = self.tokenizer(text.lower())
        indices = [self.vocab[token] for token in tokens if token in self.vocab]
        
        # Pad or truncate to max_length
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        else:
            indices = indices + [self.vocab['<pad>']] * (self.max_length - len(indices))
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def yield_tokens(data_iter: Iterator) -> Iterator[List[str]]:
    """Yield tokens from dataset iterator"""
    tokenizer = get_tokenizer('basic_english')
    for _, text in data_iter:
        yield tokenizer(text.lower())

def build_vocab(data_iter: Iterator, vocab_size: int = VOCAB_SIZE) -> Vocab:
    """Build vocabulary from dataset"""
    vocab = build_vocab_from_iterator(
        yield_tokens(data_iter),
        max_tokens=vocab_size,
        specials=['<unk>', '<pad>', '<sos>', '<eos>']
    )
    vocab.set_default_index(vocab['<unk>'])
    return vocab

def load_imdb_data(batch_size: int = BATCH_SIZE, max_length: int = MAX_SEQUENCE_LENGTH):
    """Load and preprocess IMDB dataset"""
    
    # Load train and test datasets
    train_iter = IMDB(split='train')
    test_iter = IMDB(split='test')
    
    # Build vocabulary from training data
    print("Building vocabulary...")
    vocab = build_vocab(train_iter, VOCAB_SIZE)
    
    # Reload iterators after building vocab
    train_iter = IMDB(split='train')
    test_iter = IMDB(split='test')
    
    # Create datasets
    train_dataset = IMDBDataset(train_iter, vocab, max_length)
    test_dataset = IMDBDataset(test_iter, vocab, max_length)
    
    # Split train into train and validation
    train_size = int(TRAIN_SPLIT * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"Vocabulary size: {len(vocab)}")
    
    return train_loader, val_loader, test_loader, vocab

def save_synthetic_data(data: List[Tuple[str, int]], filepath: str):
    """Save synthetic data to file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for text, label in data:
            label_str = 'pos' if label == 1 else 'neg'
            f.write(f"{label_str}\t{text}\n")

def load_synthetic_data(filepath: str, tokenizer=None, max_length: int = MAX_SEQUENCE_LENGTH):
    """Load synthetic data and create a list of (text, label) pairs for HuggingFace pipeline"""
    dataset = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            label_str, text = line.strip().split('\t', 1)
            label = 1 if label_str == 'pos' else 0
            dataset.append((text, label))
    return dataset

MODEL_NAME = "AirrStorm/DistilBERT-SST2-Yelp"

# Collate function for tokenization and padding
def collate_batch(tokenizer, batch, max_length, text_column):
    texts = [item[text_column] for item in batch]
    labels = [item["label"] for item in batch]
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    labels = torch.tensor(labels, dtype=torch.long)
    return encodings, labels

def load_hf_data(batch_size=BATCH_SIZE, max_length=MAX_SEQUENCE_LENGTH):
    # Load the dataset (try Yelp, fallback to SST2 if needed)
    text_column = "text"
    try:
        dataset = load_dataset(MODEL_NAME)
    except Exception:
        # Fallback to SST2
        dataset = load_dataset("glue", "sst2")
        text_column = "sentence"

    # Use 'train', 'validation', 'test' splits if available
    splits = list(dataset.keys())
    train_split = splits[0]
    val_split = splits[1] if len(splits) > 2 else None
    test_split = splits[-1]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Prepare DataLoaders
    def get_loader(split):
        if split is None:
            return None
        data = dataset[split]
        return DataLoader(
            data,
            batch_size=batch_size,
            shuffle=(split == train_split),
            collate_fn=partial(collate_batch, tokenizer, max_length=max_length, text_column=text_column)
        )

    train_loader = get_loader(train_split)
    val_loader = get_loader(val_split)
    test_loader = get_loader(test_split)

    # Number of classes (assume binary)
    num_classes = 2

    return train_loader, val_loader, test_loader, tokenizer, num_classes 