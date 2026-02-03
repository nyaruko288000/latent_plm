import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from typing import Optional, Dict, Union
from pathlib import Path
import random


def get_tokenizer(
    tokenizer_path: Optional[str] = None,
    fallback: str = "gpt2"
) -> PreTrainedTokenizerFast:
    if tokenizer_path and Path(tokenizer_path).exists():
        from build_tokenizer import load_custom_tokenizer
        print(f"Loading custom tokenizer from {tokenizer_path}")
        tokenizer = load_custom_tokenizer(tokenizer_path)
    else:
        print(f"Using {fallback} tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(fallback)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    print(f"  Vocab size: {tokenizer.vocab_size}")
    return tokenizer


class TinyStoriesDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        tokenizer: PreTrainedTokenizerFast = None,
        tokenizer_path: Optional[str] = None,
        max_prefix_len: int = 256,
        max_target_len: int = 256,
        chunk_size: int = 8,
        split_ratio: float = 0.5,
        cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = get_tokenizer(tokenizer_path)
        
        self.max_prefix_len = max_prefix_len
        self.max_target_len = max_target_len
        self.chunk_size = chunk_size
        self.split_ratio = split_ratio
        self.pad_token_id = self.tokenizer.pad_token_id
        
        print(f"Loading TinyStories {split} split...")
        dataset = load_dataset(
            "roneneldan/TinyStories",
            split=split,
            cache_dir=cache_dir
        )
        
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        self.data = dataset
        print(f"  Loaded {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.data[idx]["text"]
        
        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_prefix_len + self.max_target_len
        )
        
        total_len = len(tokens)
        split_point = int(total_len * self.split_ratio)
        split_point = max(16, min(split_point, total_len - 16))
        
        prefix_tokens = tokens[:split_point]
        target_tokens = tokens[split_point:]
        
        # 确保 target 是 chunk_size 的倍数
        target_len = (len(target_tokens) // self.chunk_size) * self.chunk_size
        target_len = max(target_len, self.chunk_size)
        target_len = min(target_len, self.max_target_len)  # 限制最大长度
        target_tokens = target_tokens[:target_len]
        
        # Pad target 到固定长度
        if len(target_tokens) < self.max_target_len:
            # 先确保是 chunk_size 的倍数
            padded_len = ((self.max_target_len // self.chunk_size)) * self.chunk_size
            pad_len = padded_len - len(target_tokens)
            target_tokens = target_tokens + [self.pad_token_id] * pad_len
        
        # Pad prefix
        prefix_len = len(prefix_tokens)
        if prefix_len < self.max_prefix_len:
            pad_len = self.max_prefix_len - prefix_len
            prefix_mask = [1] * prefix_len + [0] * pad_len
            prefix_tokens = prefix_tokens + [self.pad_token_id] * pad_len
        else:
            prefix_tokens = prefix_tokens[:self.max_prefix_len]
            prefix_mask = [1] * self.max_prefix_len
        
        return {
            "prefix": torch.tensor(prefix_tokens, dtype=torch.long),
            "prefix_mask": torch.tensor(prefix_mask, dtype=torch.long),
            "target": torch.tensor(target_tokens, dtype=torch.long),
        }


class AutoencoderDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        tokenizer: PreTrainedTokenizerFast = None,
        tokenizer_path: Optional[str] = None,
        chunk_size: int = 8,
        max_length: int = 512,
        cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = get_tokenizer(tokenizer_path)
        
        self.chunk_size = chunk_size
        self.max_length = max_length
        self.pad_token_id = self.tokenizer.pad_token_id
        
        print(f"Loading TinyStories {split} for autoencoder...")
        dataset = load_dataset(
            "roneneldan/TinyStories",
            split=split,
            cache_dir=cache_dir
        )
        
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        self.data = dataset
        print(f"  Loaded {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.data[idx]["text"]
        
        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length
        )
        
        if len(tokens) >= self.chunk_size:
            start = random.randint(0, len(tokens) - self.chunk_size)
            chunk = tokens[start:start + self.chunk_size]
        else:
            chunk = tokens + [self.pad_token_id] * (self.chunk_size - len(tokens))
        
        return {
            "tokens": torch.tensor(chunk, dtype=torch.long)
        }


def get_dataloaders(
    batch_size: int = 32,
    max_prefix_len: int = 256,
    max_target_len: int = 256,
    chunk_size: int = 8,
    num_workers: int = 4,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = 10000,
    tokenizer_path: Optional[str] = None,
):
    tokenizer = get_tokenizer(tokenizer_path)
    
    # 确保 max_target_len 是 chunk_size 的倍数
    max_target_len = (max_target_len // chunk_size) * chunk_size
    
    train_dataset = TinyStoriesDataset(
        split="train",
        tokenizer=tokenizer,
        max_prefix_len=max_prefix_len,
        max_target_len=max_target_len,
        chunk_size=chunk_size,
        max_samples=max_train_samples,
    )
    
    val_dataset = TinyStoriesDataset(
        split="validation",
        tokenizer=tokenizer,
        max_prefix_len=max_prefix_len,
        max_target_len=max_target_len,
        chunk_size=chunk_size,
        max_samples=max_val_samples,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, tokenizer


def get_ae_dataloader(
    batch_size: int = 128,
    chunk_size: int = 8,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    tokenizer_path: Optional[str] = None,
):
    tokenizer = get_tokenizer(tokenizer_path)
    
    dataset = AutoencoderDataset(
        split="train",
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        max_samples=max_samples,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    return loader, tokenizer