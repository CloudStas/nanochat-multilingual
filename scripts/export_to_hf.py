"""
Export a nanochat checkpoint to HuggingFace Hub in LlamaForCausalLM format.

The conversion is approximate in the MLP:
  nanochat: h = fc2(relu²(fc1(x)))               (2-matrix, relu² activation)
  Llama:    h = down(silu(gate(x)) * up(x))       (3-matrix, SiLU gated MLP)

Approximation: gate_proj = up_proj = fc1,  down_proj = fc2.
Result: down(silu(fc1(x)) * fc1(x)) ≈ fc2(relu²(fc1(x)))  since silu(t)*t ≈ relu²(t).

Features that are NOT converted (minor quality impact):
  - value_embeds (ResFormer-style residual in attention values)
  - smear / backout scalars
  - per-layer resid/x0 lambdas

RoPE, RMSNorm (as identity weights=1), attention, and embeddings convert exactly.

Usage:
    python -m scripts.export_to_hf --model-tag multilingual_d28 --hf-repo your-org/nanochat-multilingual
    python -m scripts.export_to_hf --source sft --hf-repo your-org/nanochat-multilingual-sft

After uploading, convert to GGUF with llama.cpp:
    git clone https://github.com/ggerganov/llama.cpp
    pip install -r llama.cpp/requirements/requirements-convert-hf-to-gguf.txt
    python llama.cpp/convert_hf_to_gguf.py /path/to/hf-model --outtype bf16
    ./llama.cpp/build/bin/llama-quantize model.gguf model-Q4_K_M.gguf Q4_K_M
"""

import os
import json
import pickle
import argparse
import shutil
import torch
import numpy as np

# -------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Export nanochat checkpoint to HuggingFace")
parser.add_argument("--source", type=str, default="sft", choices=["base", "sft"],
                    help="Checkpoint source: 'base' (pretrained) or 'sft' (chat-tuned)")
parser.add_argument("--model-tag", type=str, default=None, help="Model tag (e.g. multilingual_d28)")
parser.add_argument("--model-step", type=int, default=None, help="Checkpoint step (default: latest)")
parser.add_argument("--hf-repo", type=str, required=True,
                    help="HuggingFace repo id, e.g. myorg/nanochat-multilingual")
parser.add_argument("--hf-token", type=str, default=None,
                    help="HuggingFace API token (or set HF_TOKEN env var)")
parser.add_argument("--output-dir", type=str, default=None,
                    help="Local output directory (default: /tmp/nanochat_hf_export)")
parser.add_argument("--private", action="store_true", help="Create private HuggingFace repo")
parser.add_argument("--push", action="store_true", help="Push to HuggingFace Hub after export")
args = parser.parse_args()

device = torch.device("cpu")

# -------------------------------------------------------------------------
print(f"Loading nanochat checkpoint (source={args.source}, tag={args.model_tag}, step={args.model_step})")
from nanochat.checkpoint_manager import load_model
model, tokenizer, meta = load_model(args.source, device=device, phase="eval",
                                    model_tag=args.model_tag, step=args.model_step)
config = model.config
print(f"Model config: {config}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# -------------------------------------------------------------------------
# Output directory
output_dir = args.output_dir or "/tmp/nanochat_hf_export"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)
print(f"Writing HF model to {output_dir}")

# -------------------------------------------------------------------------
# 1. Build the Llama config.json

vocab_size = config.vocab_size
n_embd = config.n_embd
n_layer = config.n_layer
n_head = config.n_head
n_kv_head = config.n_kv_head
seq_len = config.sequence_len
intermediate_size = 4 * n_embd  # nanochat MLP uses 4× expansion

llama_config = {
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": n_embd,
    "intermediate_size": intermediate_size,
    "num_hidden_layers": n_layer,
    "num_attention_heads": n_head,
    "num_key_value_heads": n_kv_head,
    "max_position_embeddings": seq_len,
    "rope_theta": 100000.0,
    "vocab_size": vocab_size,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-5,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    # Sliding window: use full context for simplicity (sliding window varies per layer in nanochat)
    "attention_bias": False,
    "mlp_bias": False,
    "bos_token_id": tokenizer.get_bos_token_id(),
    "eos_token_id": tokenizer.get_bos_token_id(),  # nanochat uses BOS as delimiter
    "transformers_version": "4.46.0",
    # Note: MLP activation is an approximation — nanochat uses relu², Llama uses silu
    # Note: value_embeds, smear, backout, resid/x0 lambdas are not converted
    "_nanochat_note": (
        "Approximate Llama conversion of nanochat model. "
        "MLP: relu² → silu (gate=up=fc1, down=fc2). "
        "RMSNorm weights are 1 (nanochat uses unparameterized RMSNorm). "
        "value_embeds and per-layer scalars are dropped."
    ),
}
with open(os.path.join(output_dir, "config.json"), "w") as f:
    json.dump(llama_config, f, indent=2)
print("Wrote config.json")

# -------------------------------------------------------------------------
# 2. Convert and save model weights

state_dict = model.state_dict()

# nanochat pads vocab to a multiple of 64; crop back to vocab_size
def crop_vocab(weight):
    if weight.shape[0] > vocab_size:
        return weight[:vocab_size].contiguous()
    if weight.shape[-1] > vocab_size:
        return weight[..., :vocab_size].contiguous()
    return weight

hf_weights = {}

# Token embeddings
hf_weights["model.embed_tokens.weight"] = crop_vocab(state_dict["transformer.wte.weight"]).to(torch.bfloat16)

# Final norm (unparameterized in nanochat → ones)
hf_weights["model.norm.weight"] = torch.ones(n_embd, dtype=torch.bfloat16)

# LM head
hf_weights["lm_head.weight"] = crop_vocab(state_dict["lm_head.weight"]).to(torch.bfloat16)

# Per-layer weights
for i in range(n_layer):
    prefix = f"transformer.h.{i}"
    out = f"model.layers.{i}"

    # Input layernorm (unparameterized in nanochat → ones)
    hf_weights[f"{out}.input_layernorm.weight"] = torch.ones(n_embd, dtype=torch.bfloat16)

    # Attention
    hf_weights[f"{out}.self_attn.q_proj.weight"] = state_dict[f"{prefix}.attn.c_q.weight"].to(torch.bfloat16)
    hf_weights[f"{out}.self_attn.k_proj.weight"] = state_dict[f"{prefix}.attn.c_k.weight"].to(torch.bfloat16)
    hf_weights[f"{out}.self_attn.v_proj.weight"] = state_dict[f"{prefix}.attn.c_v.weight"].to(torch.bfloat16)
    hf_weights[f"{out}.self_attn.o_proj.weight"] = state_dict[f"{prefix}.attn.c_proj.weight"].to(torch.bfloat16)

    # Post-attention layernorm (unparameterized in nanochat → ones)
    hf_weights[f"{out}.post_attention_layernorm.weight"] = torch.ones(n_embd, dtype=torch.bfloat16)

    # MLP — approximate relu² with silu:  gate=up=fc1, down=fc2
    fc1 = state_dict[f"{prefix}.mlp.c_fc.weight"].to(torch.bfloat16)    # [4d, d]
    fc2 = state_dict[f"{prefix}.mlp.c_proj.weight"].to(torch.bfloat16)  # [d, 4d]
    hf_weights[f"{out}.mlp.gate_proj.weight"] = fc1
    hf_weights[f"{out}.mlp.up_proj.weight"]   = fc1.clone()  # duplicate — see module docstring
    hf_weights[f"{out}.mlp.down_proj.weight"] = fc2

print(f"Converted {len(hf_weights)} weight tensors")

# Save as safetensors (preferred by HF)
try:
    from safetensors.torch import save_file
    save_file(hf_weights, os.path.join(output_dir, "model.safetensors"))
    print("Saved model.safetensors")
except ImportError:
    # Fall back to pytorch format
    torch.save(hf_weights, os.path.join(output_dir, "pytorch_model.bin"))
    print("Saved pytorch_model.bin (install safetensors for the preferred format)")

# -------------------------------------------------------------------------
# 3. Convert tokenizer to HuggingFace format

def _export_tokenizer(tokenizer, tokenizer_dir, output_dir):
    """
    Export nanochat tokenizer to HF format.
    Prefers tokenizer.json (saved by tok_train_multilingual.py).
    Falls back to reconstructing from tiktoken pickle.
    """
    hf_json = os.path.join(tokenizer_dir, "tokenizer.json")
    if os.path.exists(hf_json):
        # Direct copy — already in HF format
        shutil.copy(hf_json, os.path.join(output_dir, "tokenizer.json"))
        print("Copied existing tokenizer.json")
        return

    print("tokenizer.json not found; reconstructing from tiktoken pickle...")
    pkl_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
    if not os.path.exists(pkl_path):
        print("ERROR: no tokenizer.pkl found either — cannot export tokenizer")
        return

    with open(pkl_path, "rb") as f:
        enc = pickle.load(f)

    _reconstruct_hf_tokenizer(enc, output_dir)


def _reconstruct_hf_tokenizer(enc, output_dir):
    """Reconstruct HuggingFace tokenizer from tiktoken Encoding object."""
    from tokenizers import Tokenizer, AddedToken
    from tokenizers.models import BPE
    from tokenizers import pre_tokenizers, decoders, Regex
    from nanochat.tokenizer import SPLIT_PATTERN, SPECIAL_TOKENS

    mergeable_ranks = enc.mergeable_ranks  # dict[bytes, int]

    # Build vocab: encode bytes as latin-1 strings (bijection bytes↔chars)
    vocab = {}
    for tok_bytes, rank in mergeable_ranks.items():
        vocab[tok_bytes.decode("latin-1")] = rank

    # Add special tokens after the base vocabulary
    special_offset = len(mergeable_ranks)
    for i, st in enumerate(SPECIAL_TOKENS):
        vocab[st] = special_offset + i

    # Reconstruct BPE merges from mergeable_ranks
    merges = _reconstruct_merges(mergeable_ranks)
    print(f"  Reconstructed {len(merges)} BPE merges")

    # Build the HF BPE tokenizer
    hf_tok = Tokenizer(BPE(vocab=vocab, merges=merges, byte_fallback=True, unk_token=None, fuse_unk=False))
    hf_tok.normalizer = None
    gpt4_regex = Regex(SPLIT_PATTERN)
    hf_tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=gpt4_regex, behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    hf_tok.decoder = decoders.ByteLevel()

    # Add special tokens
    added = [AddedToken(st, special=True) for st in SPECIAL_TOKENS]
    hf_tok.add_special_tokens(added)

    # Save
    hf_tok.save(os.path.join(output_dir, "tokenizer.json"))
    print("Saved reconstructed tokenizer.json")


def _reconstruct_merges(mergeable_ranks):
    """Reconstruct BPE merge rules from tiktoken's mergeable_ranks dict."""
    merges = []
    # Sort all tokens by rank (ascending = order they were created)
    sorted_tokens = sorted(mergeable_ranks.items(), key=lambda x: x[1])
    for tok_bytes, rank in sorted_tokens:
        if len(tok_bytes) == 1:
            continue  # base byte token, not a merge result
        # Find the split into (left, right) that minimizes max(rank_left, rank_right)
        # This reconstructs the merge that created this token
        best_pair = None
        best_max_rank = rank  # both sub-tokens must have rank < current
        for split in range(1, len(tok_bytes)):
            left, right = tok_bytes[:split], tok_bytes[split:]
            if left in mergeable_ranks and right in mergeable_ranks:
                lr, rr = mergeable_ranks[left], mergeable_ranks[right]
                if lr < rank and rr < rank:
                    max_r = max(lr, rr)
                    if max_r < best_max_rank:
                        best_max_rank = max_r
                        best_pair = (left, right)
        if best_pair is not None:
            merges.append((best_pair[0].decode("latin-1"), best_pair[1].decode("latin-1")))
    return merges


from nanochat.common import get_base_dir
tokenizer_dir = os.path.join(get_base_dir(), "tokenizer")
_export_tokenizer(tokenizer, tokenizer_dir, output_dir)

# Write tokenizer_config.json
bos_id = tokenizer.get_bos_token_id()
tok_config = {
    "bos_token": "<|bos|>",
    "eos_token": "<|bos|>",
    "model_max_length": seq_len,
    "tokenizer_class": "PreTrainedTokenizerFast",
    "clean_up_tokenization_spaces": False,
    "added_tokens_decoder": {
        str(bos_id): {"content": "<|bos|>", "special": True},
    },
}
with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
    json.dump(tok_config, f, indent=2)

# Write special_tokens_map.json
with open(os.path.join(output_dir, "special_tokens_map.json"), "w") as f:
    json.dump({"bos_token": "<|bos|>", "eos_token": "<|bos|>"}, f, indent=2)

print("Wrote tokenizer config files")

# -------------------------------------------------------------------------
# 4. Write generation_config.json
gen_config = {
    "bos_token_id": bos_id,
    "eos_token_id": bos_id,
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
}
with open(os.path.join(output_dir, "generation_config.json"), "w") as f:
    json.dump(gen_config, f, indent=2)

# -------------------------------------------------------------------------
# 5. Push to HuggingFace Hub
print(f"\nExport complete: {output_dir}")
print(f"Contents: {os.listdir(output_dir)}")

if args.push:
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("Set --hf-token or HF_TOKEN env var to push to HuggingFace Hub")

    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)

    print(f"\nCreating/updating HF repo: {args.hf_repo}")
    api.create_repo(repo_id=args.hf_repo, private=args.private, exist_ok=True)

    print(f"Uploading files to {args.hf_repo}...")
    api.upload_folder(
        folder_path=output_dir,
        repo_id=args.hf_repo,
        repo_type="model",
        commit_message=f"Upload nanochat multilingual model (depth={n_layer}, n_embd={n_embd})",
    )
    print(f"\nModel uploaded to: https://huggingface.co/{args.hf_repo}")
    print("\nTo convert to GGUF for llama.cpp:")
    print(f"  git clone https://github.com/ggerganov/llama.cpp")
    print(f"  pip install -r llama.cpp/requirements/requirements-convert-hf-to-gguf.txt")
    print(f"  huggingface-cli download {args.hf_repo} --local-dir /tmp/hf-model")
    print(f"  python llama.cpp/convert_hf_to_gguf.py /tmp/hf-model --outtype bf16 --outfile model.gguf")
    print(f"  ./llama.cpp/build/bin/llama-quantize model.gguf model-Q4_K_M.gguf Q4_K_M")
else:
    print(f"\nDry run complete (use --push to upload to HuggingFace Hub)")
    print(f"To push: python -m scripts.export_to_hf --hf-repo {args.hf_repo} --push")
