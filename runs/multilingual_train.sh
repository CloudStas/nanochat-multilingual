#!/bin/bash
# =============================================================================
# Full multilingual LLM training pipeline for a spot 1x H100 (50 hours).
#
# Strategy: SMALL MODEL + HEAVY OVERTRAINING (inference-optimal, Llama-3.2 style)
#   depth=20 → ~393M params
#   ~52B tokens in 50h with FP8  →  ~132× Chinchilla  (vs ~107× for Llama-3.2-1B)
#   Result: a small, fast-to-serve multilingual model that trains fully in budget
#
# Languages: EN + ZH ES FR DE JA RU PT AR KO IT NL PL TR VI HI ID CS UK SV (top-20)
# Spot resilience: auto-resumes from latest checkpoint on restart
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
MODEL_TAG="multilingual_d20"

# ── Model & training knobs ────────────────────────────────────────────────────
# depth=20   → model_dim=1280, ~393M params (fast inference, fits any GPU)
# BATCH=64   → 64 × 2048 = 131K tokens/rank; 8 grad-accum for 1M total batch
# TOTAL_BS   → 1,048,576 tokens  (1M, divisible by 64×2048=131072, 8 accum steps)
# RATIO=150  → 150 × 393M ≈ 59B token budget; spot preemption will stop earlier;
#              resume + retrain on the next spot session to consume remaining tokens
# ─────────────────────────────────────────────────────────────────────────────
TEST_RUN="${TEST_RUN:-}"
if [ -n "$TEST_RUN" ]; then
    DEPTH=4; SEQ_LEN=256; BATCH=1; TOTAL_BS=512; RATIO=5; DEVICE_ARGS="--device-type=cpu"
    echo "TEST MODE: tiny model"
else
    DEPTH=20; SEQ_LEN=2048; BATCH=64; TOTAL_BS=1048576; RATIO=150; DEVICE_ARGS=""
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
    echo "Pretraining complete. Starting/resuming SFT."
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
    # With depth=20 + RATIO=150: ~59B total tokens, 50% English = ~29B English tokens
    # Each ClimbMix shard ≈ 60M tokens → need ~480 shards English
    # But mix is 50/50 by shard count and we have 19 non-English × 20 shards = 380 non-EN
    # → need 380 English shards. Predownload 8 for tokenizer, rest in background.
    python -m nanochat.dataset -n 8
    python -m nanochat.dataset -n 400 &
    EN_DOWNLOAD_PID=$!

    echo "=== Phase 1: Download multilingual data (mC4) ==="
    # 20 shards × 19 languages = 380 shards, each ~50K docs / ~25M tokens
    # → ~475B non-English tokens (more than enough; dataloader cycles as needed)
    python -m nanochat.multilingual_dataset --shards-per-lang 20 -w 4 &
    ML_DOWNLOAD_PID=$!

    echo "=== Phase 1: Waiting for initial data (English 8 shards) ==="
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
    echo "=== Phase 2: Pretraining (depth=$DEPTH, ~393M params, overtrained ~${RATIO}x Chinchilla) ==="

    RESUME_ARG=""
    PRETRAIN_STEP=$(find_pretrain_step)
    if [ -n "$PRETRAIN_STEP" ] && [ "$PHASE" = "pretrain_resume" ]; then
        RESUME_ARG="--resume-from-step=$PRETRAIN_STEP"
        echo "  Resuming from step $PRETRAIN_STEP"
    fi

    # H100 throughput notes:
    #   depth=20, FP8 BF16 → ~280-320K tok/sec → 52B tokens in ~50h ✓
    #   TOTAL_BS=1M, BATCH=64: grad_accum = 1M / (64×2048) = 8 steps → good utilization
    #   --target-param-data-ratio=$RATIO: training stops at RATIO × params tokens
    #   If spot is preempted before that, SIGTERM saves checkpoint; resume next session
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
