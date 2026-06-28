"""
Train a BPE tokenizer on multilingual data (top-20 languages).
Saves both formats:
  tokenizer/tokenizer.pkl   — tiktoken (used during training)
  tokenizer/tokenizer.json  — HuggingFace format (used for HF Hub upload)

Run from the repo root:
    python -m scripts.tok_train_multilingual
"""

import os
import time
import argparse
import torch

from nanochat.tokenizer import RustBPETokenizer, HuggingFaceTokenizer, SPLIT_PATTERN
from nanochat.common import get_base_dir
from nanochat.multilingual_dataset import get_mix_dir

import pyarrow.parquet as pq

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train multilingual BPE tokenizer")
parser.add_argument("--max-chars", type=int, default=3_000_000_000,
                    help="Max characters to train on (default: 3B for better multilingual coverage)")
parser.add_argument("--doc-cap", type=int, default=5_000,
                    help="Max characters per document (default: 5,000)")
parser.add_argument("--vocab-size", type=int, default=65536,
                    help="Vocabulary size (default: 65536 = 2^16, larger for multilingual)")
args = parser.parse_args()

print(f"max_chars:  {args.max_chars:,}")
print(f"doc_cap:    {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")

# -----------------------------------------------------------------------------
# Build a character iterator that samples from the multilingual mix directory
# (50% English, 50% non-English as created by multilingual_dataset.py)

def _iter_parquet_dir(parquet_dir):
    """Yield text documents from a parquet directory in sorted order."""
    if not os.path.isdir(parquet_dir):
        return
    files = sorted(f for f in os.listdir(parquet_dir) if f.endswith(".parquet"))
    for fname in files:
        fp = os.path.join(parquet_dir, fname)
        try:
            pf = pq.ParquetFile(fp)
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                for text in rg.column("text").to_pylist():
                    if text:
                        yield text
        except Exception as e:
            print(f"Warning: skipping {fp}: {e}")


def text_iterator():
    """
    Yield documents for tokenizer training.
    Reads from the multilingual mix directory (created by multilingual_dataset.py).
    Falls back to ClimbMix if mix directory doesn't exist.
    """
    mix_dir = get_mix_dir()
    if not os.path.isdir(mix_dir):
        base_dir = get_base_dir()
        fallback = os.environ.get("NANOCHAT_DATA_DIR", os.path.join(base_dir, "base_data_climbmix"))
        print(f"Warning: mix dir {mix_dir} not found, using {fallback}")
        mix_dir = fallback

    nchars = 0
    for text in _iter_parquet_dir(mix_dir):
        doc = text[:args.doc_cap]
        nchars += len(doc)
        yield doc
        if nchars >= args.max_chars:
            return

text_iter = text_iterator()

# -----------------------------------------------------------------------------
# Train the RustBPE tokenizer (used during training for efficiency)
print("\nTraining RustBPE tokenizer...")
t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
t1 = time.time()
print(f"RustBPE training time: {t1 - t0:.2f}s")

# Save the RustBPE tokenizer
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
tokenizer.save(tokenizer_dir)
print(f"Saved RustBPE tokenizer to {tokenizer_dir}/tokenizer.pkl")

# -----------------------------------------------------------------------------
# ALSO train a HuggingFace tokenizer with identical settings for HF Hub export
print("\nTraining HuggingFace tokenizer (for export)...")

# Reset the text iterator for a second pass
text_iter2 = text_iterator()
t0 = time.time()
hf_tokenizer = HuggingFaceTokenizer.train_from_iterator(text_iter2, args.vocab_size)
t1 = time.time()
print(f"HuggingFace tokenizer training time: {t1 - t0:.2f}s")

hf_tokenizer.save(tokenizer_dir)
print(f"Saved HuggingFace tokenizer to {tokenizer_dir}/tokenizer.json")

# -----------------------------------------------------------------------------
# Sanity check: both tokenizers should produce similar output lengths
test_texts = [
    "Hello world! This is a test.",  # English
    "你好世界，这是一个测试。",            # Chinese
    "Hola mundo! Esta es una prueba.",  # Spanish
    "مرحبا بالعالم! هذا اختبار.",       # Arabic
    "こんにちは世界！これはテストです。", # Japanese
]
print("\nSanity check:")
for text in test_texts:
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    assert decoded == text, f"Roundtrip failed for: {repr(text)}"
    print(f"  '{text[:30]}...' → {len(ids)} tokens")

# -----------------------------------------------------------------------------
# Save token_bytes tensor (for bits-per-byte evaluation)
vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_bytes = []
for token_id in range(vocab_size):
    token_str = tokenizer.decode([token_id])
    if token_str in special_set:
        token_bytes.append(0)
    else:
        token_bytes.append(len(token_str.encode("utf-8")))
token_bytes = torch.tensor(token_bytes, dtype=torch.int32)
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
torch.save(token_bytes, token_bytes_path)
print(f"\nSaved token_bytes to {token_bytes_path}")
print(f"Vocab size: {vocab_size:,}")
print(f"Avg bytes per token: {token_bytes[token_bytes > 0].float().mean():.2f}")
