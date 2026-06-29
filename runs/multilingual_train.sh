#!/bin/bash
# =============================================================================
# Full multilingual LLM training pipeline for a non-interruptible 1x H100 (12.8 hours).
#
# Strategy: TINY MODEL + EXTREME OVERTRAINING (inference-optimal, phi/SmolLM style)
#   depth=12 → d=768, 85M transformer params, 186M total (incl. 65536-vocab embeddings)
#   Budget breakdown (12.8h total):
#     ~1h   data download + tokenizer
#     ~9.5h pretrain  →  9.5h × ~900K tok/s × 3600 = ~30B tokens  →  ~350× Chinchilla
#     ~2h   SFT
#     ~0.2h export + HF upload
#   Depth=12 is ~4.6× fewer FLOPs/token vs depth=20 → faster training AND inference
#   Result: tiny fast model trained on far more tokens than Chinchilla-optimal
#
# Languages: EN + ZH ES FR DE JA RU PT AR KO IT NL PL TR VI HI ID CS UK SV (top-20)
#
# Usage:
#   HF_REPO=your-org/nanochat-multilingual HF_TOKEN=hf_xxx bash runs/multilingual_train.sh
#   WANDB_RUN=multi20 HF_REPO=... HF_TOKEN=... bash runs/multilingual_train.sh
#   TEST_RUN=1 bash runs/multilingual_train.sh   # CPU smoke-test
# =============================================================================

set -euo pipefail

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

WANDB_RUN="${WANDB_RUN:-dummy}"
HF_REPO="${HF_REPO:-}"
HF_TOKEN="${HF_TOKEN:-}"
MODEL_TAG="multilingual_d12"

# ── Model & training knobs ────────────────────────────────────────────────────
# depth=12   → d=768, 85M transformer params, 186M total (incl. embeddings)
# BATCH=64   → 64 × 2048 = 131K tokens/rank; 8 grad-accum for 1M total batch
# TOTAL_BS   → 1,048,576 tokens (1M, divisible by 64×2048=131072, 8 accum steps)
# RATIO=300  → 300 × 85M ≈ 25.5B tokens → ~8.9h pretrain on H100 at ~800K tok/s
#              (depth=12 is ~4.6× faster than depth=20 → ~300× Chinchilla vs 26×)
# Data       → 120 EN shards + 8 per non-EN lang (dataloader cycles ~2-5×)
# ─────────────────────────────────────────────────────────────────────────────
TEST_RUN="${TEST_RUN:-}"
if [ -n "$TEST_RUN" ]; then
    DEPTH=4; SEQ_LEN=256; BATCH=1; TOTAL_BS=512; RATIO=5; DEVICE_ARGS="--device-type=cpu"
    echo "TEST MODE: tiny model on CPU"
else
    DEPTH=12; SEQ_LEN=2048; BATCH=64; TOTAL_BS=1048576; RATIO=300; DEVICE_ARGS=""
fi

# =============================================================================
# Python venv

command -v uv &>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu 2>/dev/null || uv sync --extra cpu
source .venv/bin/activate

# =============================================================================
# Helper: find latest pretrain checkpoint step
find_pretrain_step() {
    local dir="$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"
    if [ ! -d "$dir" ]; then echo ""; return; fi
    ls "$dir"/model_*.pt 2>/dev/null | sed 's/.*model_0*//' | sed 's/.pt//' | sort -n | tail -1
}

# =============================================================================
# Phase detection: figure out which phase to (re)start from

PRETRAIN_DONE_FLAG="$NANOCHAT_BASE_DIR/pretrain_done_$MODEL_TAG"
SFT_DONE_FLAG="$NANOCHAT_BASE_DIR/sft_done_$MODEL_TAG"

if [ -f "$SFT_DONE_FLAG" ]; then
    echo "SFT already complete. Proceeding to HF upload."
    PHASE="upload"
elif [ -f "$PRETRAIN_DONE_FLAG" ]; then
    echo "Pretraining complete. Starting SFT."
    PHASE="sft"
else
    PRETRAIN_STEP=$(find_pretrain_step)
    if [ -n "$PRETRAIN_STEP" ]; then
        echo "Resuming pretraining from step $PRETRAIN_STEP"
        PHASE="pretrain_resume"
    else
        echo "Starting fresh pretraining"
        PHASE="pretrain_fresh"
    fi
fi

# =============================================================================
# PHASE 1: Data preparation (only runs once)

if [ "$PHASE" = "pretrain_fresh" ]; then
    python -m nanochat.report reset

    echo "=== Phase 1: Download English data (ClimbMix) ==="
    # RATIO=300 → ~25.5B tokens, 50% English = ~12.75B English tokens
    # Each ClimbMix shard ≈ 60M tokens. 120 shards = 7.2B tokens → cycles ~1.8×
    python -m nanochat.dataset -n 120 &
    EN_DOWNLOAD_PID=$!

    echo "=== Phase 1: Download multilingual data (mC4) ==="
    # 8 shards × 19 languages × ~25M tokens/shard = ~3.8B non-EN tokens → cycles ~3.4×
    python -m nanochat.multilingual_dataset --shards-per-lang 8 -w 8 &
    ML_DOWNLOAD_PID=$!

    echo "=== Phase 1: Waiting for all downloads... ==="
    wait $EN_DOWNLOAD_PID
    wait $ML_DOWNLOAD_PID
    echo "All data downloads complete."

    echo "=== Phase 1: Create mixed training directory ==="
    python -m nanochat.multilingual_dataset --create-mix

    echo "=== Phase 1: Train multilingual tokenizer (vocab=65536, 3B chars) ==="
    NANOCHAT_DATA_DIR="$NANOCHAT_BASE_DIR/base_data_multilingual_mix" \
        python -m scripts.tok_train_multilingual

    python -m scripts.tok_eval
fi

# =============================================================================
# PHASE 2: Pretraining

if [ "$PHASE" = "pretrain_fresh" ] || [ "$PHASE" = "pretrain_resume" ]; then
    echo "=== Phase 2: Pretraining (depth=$DEPTH, ~85M transformer params, ~${RATIO}x Chinchilla) ==="

    RESUME_ARG=""
    PRETRAIN_STEP=$(find_pretrain_step)
    if [ -n "$PRETRAIN_STEP" ] && [ "$PHASE" = "pretrain_resume" ]; then
        RESUME_ARG="--resume-from-step=$PRETRAIN_STEP"
        echo "  Resuming from step $PRETRAIN_STEP"
    fi

    # H100 throughput: depth=12 FP8 → ~800K-1M tok/s (4.6× faster than depth=20)
    # RATIO=300 → 300 × 85M / 1M batch = ~25,470 steps × ~1.25s/step ≈ 8.9h ✓
    # TOTAL_BS=1M: grad_accum = 1M / (64×2048) = 8 steps
    NANOCHAT_DATA_DIR="$NANOCHAT_BASE_DIR/base_data_multilingual_mix" \
    torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
        --depth=$DEPTH \
        --max-seq-len=$SEQ_LEN \
        --device-batch-size=$BATCH \
        --total-batch-size=$TOTAL_BS \
        --target-param-data-ratio=$RATIO \
        --fp8 \
        --fp8-recipe=tensorwise \
        --model-tag=$MODEL_TAG \
        --save-every=500 \
        --run=$WANDB_RUN \
        $RESUME_ARG \
        $DEVICE_ARGS

    touch "$PRETRAIN_DONE_FLAG"
    echo "Pretraining complete."
fi

# =============================================================================
# PHASE 3: SFT

if [ "$PHASE" != "upload" ]; then
    echo "=== Phase 3: SFT (Supervised Fine-Tuning with multilingual Aya + SmolTalk) ==="
    # chat_sft.py auto-detects SFT checkpoints in chatsft_checkpoints/$MODEL_TAG/
    # and resumes from the latest one — no manual resume flag needed.

    IDENTITY_FILE="$NANOCHAT_BASE_DIR/identity_conversations.jsonl"
    if [ ! -f "$IDENTITY_FILE" ]; then
        echo "  Skipping identity conversations (file not found: $IDENTITY_FILE)"
    fi

    SFT_RUN="${WANDB_RUN}_sft"
    torchrun --standalone --nproc_per_node=1 -m scripts.chat_sft -- \
        --model-tag=$MODEL_TAG \
        --device-batch-size=$BATCH \
        --save-every=200 \
        --run=$SFT_RUN \
        $DEVICE_ARGS

    touch "$SFT_DONE_FLAG"
    echo "SFT complete."
fi

# =============================================================================
# PHASE 4: Export to HuggingFace

echo "=== Phase 4: Export to HuggingFace ==="
python -m scripts.export_to_hf \
    --source sft \
    --model-tag=$MODEL_TAG \
    --hf-repo="${HF_REPO:-your-org/nanochat-multilingual}" \
    ${HF_TOKEN:+--hf-token=$HF_TOKEN} \
    ${HF_REPO:+--push}

echo ""
echo "=== All phases complete! ==="
if [ -n "$HF_REPO" ]; then
    echo "Model available at: https://huggingface.co/$HF_REPO"
    echo ""
    echo "To run with llama.cpp:"
    echo "  git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && cmake -B build && cmake --build build"
    echo "  pip install -r requirements/requirements-convert-hf-to-gguf.txt"
    echo "  huggingface-cli download $HF_REPO --local-dir ./hf-model"
    echo "  python convert_hf_to_gguf.py ./hf-model --outtype bf16 --outfile model.gguf"
    echo "  ./build/bin/llama-quantize model.gguf model-Q4_K_M.gguf Q4_K_M"
    echo "  ./build/bin/llama-cli -m model-Q4_K_M.gguf -p 'Hello!' -n 200"
fi
