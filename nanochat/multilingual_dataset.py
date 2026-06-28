"""
Multilingual dataset manager for top-20 languages.

Downloads from mC4 (Google's Multilingual Common Crawl) via HuggingFace datasets.
Stores as parquet files compatible with nanochat's dataloader.

Creates:
  multilingual_data/<lang>/shard_NNNNN.parquet  — per-language shards
  base_data_multilingual_mix/                    — symlinked 50/50 English+multilingual

Usage:
    python -m nanochat.multilingual_dataset --shards-per-lang 20 -w 4
    python -m nanochat.multilingual_dataset --create-mix
    python -m nanochat.multilingual_dataset --shards-per-lang 20 -w 4 --create-mix
"""

import os
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
from multiprocessing import Pool

from nanochat.common import get_base_dir
from nanochat.dataset import list_parquet_files

# Top-19 non-English languages (English comes from ClimbMix)
LANGUAGES = [
    "zh", "es", "fr", "de", "ja", "ru", "pt", "ar", "ko", "it",
    "nl", "pl", "tr", "vi", "hi", "id", "cs", "uk", "sv",
]

DOCS_PER_SHARD = 50_000   # ~50K documents per parquet shard


def get_multilingual_dir():
    return os.path.join(get_base_dir(), "multilingual_data")


def get_mix_dir():
    return os.path.join(get_base_dir(), "base_data_multilingual_mix")


def _download_language(task):
    lang, n_shards, output_dir = task
    lang_dir = os.path.join(output_dir, lang)
    os.makedirs(lang_dir, exist_ok=True)

    already = sorted(f for f in os.listdir(lang_dir) if f.endswith(".parquet"))
    done_shards = len(already)
    if done_shards >= n_shards:
        print(f"  {lang}: already have {done_shards} shards, skipping")
        return lang, done_shards

    print(f"  {lang}: downloading shards {done_shards}..{n_shards - 1}")
    try:
        from datasets import load_dataset
        ds = load_dataset("mc4", lang, split="train", streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"  {lang}: WARNING — load_dataset failed ({e}), skipping")
        return lang, done_shards

    docs = []
    shard_idx = done_shards

    for item in ds:
        text = item.get("text", "")
        if not text:
            continue
        docs.append(text)
        if len(docs) >= DOCS_PER_SHARD:
            _save_shard(lang_dir, shard_idx, docs)
            docs = []
            shard_idx += 1
            if shard_idx >= n_shards:
                break

    if docs and shard_idx < n_shards:
        _save_shard(lang_dir, shard_idx, docs)
        shard_idx += 1

    print(f"  {lang}: done ({shard_idx} shards total)")
    return lang, shard_idx


def _save_shard(lang_dir, shard_idx, docs):
    shard_path = os.path.join(lang_dir, f"shard_{shard_idx:05d}.parquet")
    table = pa.table({"text": pa.array(docs, type=pa.string())})
    pq.write_table(table, shard_path, compression="zstd")
    print(f"    saved {shard_path} ({len(docs)} docs)")


def create_mix_directory():
    """
    Create base_data_multilingual_mix/ containing symlinks to both ClimbMix (English)
    and mC4 (non-English) shards in a 50/50 ratio.

    For each group of 2*N shards: N English + 1 per non-English language.
    With N=19 languages: 19 English + 19 non-English = 38 per group → 50% English.
    """
    mix_dir = get_mix_dir()
    multilingual_dir = get_multilingual_dir()
    os.makedirs(mix_dir, exist_ok=True)

    # Gather English shards (all but last which is val)
    # Use explicit ClimbMix path to avoid NANOCHAT_DATA_DIR env var interference
    climbmix_dir = os.path.join(get_base_dir(), "base_data_climbmix")
    english_all = sorted(list_parquet_files(data_dir=climbmix_dir))
    if not english_all:
        print("ERROR: No ClimbMix shards found. Run: python -m nanochat.dataset -n 170")
        return False
    val_shard = english_all[-1]          # last shard is reserved for val
    english_train = english_all[:-1]

    # Gather non-English shards per language
    lang_paths = {}
    for lang in LANGUAGES:
        lang_dir = os.path.join(multilingual_dir, lang)
        if os.path.isdir(lang_dir):
            paths = sorted(
                os.path.join(lang_dir, f)
                for f in os.listdir(lang_dir)
                if f.endswith(".parquet")
            )
            if paths:
                lang_paths[lang] = paths

    if not lang_paths:
        print(f"ERROR: No multilingual shards in {multilingual_dir}")
        print("Run: python -m nanochat.multilingual_dataset --shards-per-lang 20")
        return False

    n_langs = len(lang_paths)
    min_lang_shards = min(len(v) for v in lang_paths.values())
    # Number of groups: each group needs n_langs English shards + 1 per language
    n_groups = min(min_lang_shards, len(english_train) // n_langs)
    if n_groups == 0:
        print(f"ERROR: Not enough shards to create mix (min_lang={min_lang_shards}, english={len(english_train)}, n_langs={n_langs})")
        return False

    print(f"Creating mix: {n_groups} groups × ({n_langs} EN + {n_langs} multilingual) = {n_groups * 2 * n_langs} shards")

    # Clear stale symlinks
    for f in os.listdir(mix_dir):
        fp = os.path.join(mix_dir, f)
        if os.path.islink(fp):
            os.unlink(fp)

    en_idx = 0
    lang_idx = {lang: 0 for lang in lang_paths}
    sym_idx = 0

    for _ in range(n_groups):
        # n_langs English shards
        for _ in range(n_langs):
            _make_symlink(mix_dir, sym_idx, english_train[en_idx])
            en_idx += 1
            sym_idx += 1
        # 1 shard per non-English language
        for lang in sorted(lang_paths.keys()):
            _make_symlink(mix_dir, sym_idx, lang_paths[lang][lang_idx[lang]])
            lang_idx[lang] += 1
            sym_idx += 1

    # Validation shard: use English val shard (last in sort order)
    _make_symlink(mix_dir, sym_idx, val_shard)
    sym_idx += 1

    total = sym_idx
    en_count = n_groups * n_langs
    non_en_count = n_groups * n_langs
    print(f"Mix directory: {mix_dir}")
    print(f"  {total} total shards: {en_count} EN ({100*en_count/total:.0f}%) + {non_en_count} multilingual ({100*non_en_count/total:.0f}%)")
    return True


def _make_symlink(mix_dir, idx, target):
    link = os.path.join(mix_dir, f"shard_{idx:06d}.parquet")
    os.symlink(os.path.abspath(target), link)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download multilingual mC4 data")
    parser.add_argument("--shards-per-lang", type=int, default=20,
                        help="Shards to download per language (default: 20, ~1M docs each)")
    parser.add_argument("--langs", nargs="+", default=LANGUAGES,
                        help="Languages to download (default: all 19)")
    parser.add_argument("-w", "--workers", type=int, default=4,
                        help="Parallel download workers (default: 4)")
    parser.add_argument("--create-mix", action="store_true",
                        help="Create the symlinked mix directory after downloading")
    args = parser.parse_args()

    output_dir = get_multilingual_dir()
    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading {args.shards_per_lang} shards × {len(args.langs)} languages → {output_dir}")

    tasks = [(lang, args.shards_per_lang, output_dir) for lang in args.langs]
    with Pool(processes=min(args.workers, len(tasks))) as pool:
        results = pool.map(_download_language, tasks)

    print("\nDownload summary:")
    for lang, count in sorted(results):
        print(f"  {lang}: {count} shards")

    if args.create_mix:
        print("\nCreating mix directory...")
        create_mix_directory()
