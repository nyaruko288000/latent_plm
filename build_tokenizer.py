"""
为 TinyStories 训练 3K 词表
- 字节级 BPE（无 OOV）
- 不需要 unk token
"""

import json
from pathlib import Path
from datasets import load_dataset
from tokenizers import (
    Tokenizer,
    models,
    trainers,
    pre_tokenizers,
    decoders,
    processors,
)
from tokenizers.normalizers import NFD, StripAccents, Sequence
from transformers import PreTrainedTokenizerFast
import argparse


def build_byte_level_bpe(
    vocab_size: int = 3000,
    save_path: str = "tokenizer_3k",
    min_frequency: int = 2,
    max_samples: int = None,
):
    """
    训练字节级 BPE tokenizer
    
    字节回退原理：
    - 所有文本先转为 UTF-8 字节 (0-255)
    - 在字节序列上做 BPE
    - 任何 Unicode 字符都可以表示，无需 unk
    """
    
    print(f"Building byte-level BPE tokenizer...")
    print(f"  vocab_size: {vocab_size}")
    print(f"  save_path: {save_path}")
    
    # 加载数据
    print("\nLoading TinyStories...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"  samples: {len(dataset)}")
    
    def batch_iterator(batch_size=1000):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i:i + batch_size]["text"]
    
    # 创建字节级 BPE tokenizer
    tokenizer = Tokenizer(models.BPE())
    
    # 字节级预处理（关键！）
    # 这会把所有字符转换为字节表示
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # 字节级解码
    tokenizer.decoder = decoders.ByteLevel()
    
    # 后处理：添加特殊 token
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    
    # 特殊 token（不需要 unk！）
    special_tokens = ["<pad>", "<bos>", "<eos>"]
    
    # 训练
    print("\nTraining tokenizer...")
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # 256 字节作为初始字母表
    )
    
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    
    # 设置特殊 token ID
    tokenizer.add_special_tokens(special_tokens)
    
    # 保存
    save_dir = Path(save_path)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    tokenizer_path = save_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    
    # 保存配置
    config = {
        "vocab_size": tokenizer.get_vocab_size(),
        "pad_token": "<pad>",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token_id": tokenizer.token_to_id("<pad>"),
        "bos_token_id": tokenizer.token_to_id("<bos>"),
        "eos_token_id": tokenizer.token_to_id("<eos>"),
        "model_type": "byte_level_bpe",
    }
    
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Saved to {save_dir}")
    print(f"  Vocab size: {tokenizer.get_vocab_size()}")
    
    # 测试
    print("\n" + "=" * 50)
    print("Testing tokenizer:")
    print("=" * 50)
    
    test_texts = [
        "Once upon a time, there was a little girl.",
        "The quick brown fox jumps over the lazy dog.",
        "Hello! How are you? 你好！",  # 测试非 ASCII
        "Numbers: 12345, symbols: @#$%",
    ]
    
    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)
        print(f"\nOriginal: {text}")
        print(f"Tokens:   {encoded.tokens[:20]}{'...' if len(encoded.tokens) > 20 else ''}")
        print(f"Length:   {len(encoded.ids)}")
        print(f"Decoded:  {decoded}")
        
        # 验证无损
        assert decoded == text, f"Decoding mismatch! Got: {decoded}"
    
    print("\n✓ All tests passed (lossless encoding)")
    
    return tokenizer


def load_custom_tokenizer(tokenizer_path: str) -> PreTrainedTokenizerFast:
    """加载自定义 tokenizer 并包装为 HuggingFace 格式"""
    
    tokenizer_path = Path(tokenizer_path)
    
    # 加载配置
    with open(tokenizer_path / "config.json") as f:
        config = json.load(f)
    
    # 加载 tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path / "tokenizer.json"))
    
    # 包装为 PreTrainedTokenizerFast
    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token=config["pad_token"],
        bos_token=config["bos_token"],
        eos_token=config["eos_token"],
    )
    
    return wrapped


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=3000)
    parser.add_argument("--save_path", type=str, default="tokenizer_3k")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    
    build_byte_level_bpe(
        vocab_size=args.vocab_size,
        save_path=args.save_path,
        max_samples=args.max_samples,
    )