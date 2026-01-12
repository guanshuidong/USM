#!/bin/bash

# ==========================================================
# Configuration for TEMPO-based Column Matching
# ==========================================================

# 1. Model Selection
# Choose which script to run: gpt2.py, llama.py, or bert.py
TARGET_SCRIPT="gpt2.py"

# 2. Path Setup (GPT-2 Example)
BASE="gpt2"
FT="gpt2_sst2"
CACHE="models"

# 3. Target Weight Configuration
LAYER=11
MODULE="attn.c_attn"
HEAD=0                # TEMPO is often applied head-by-head

# 4. Algorithm Settings
VOTES=7               # Majority voting candidates
DTYPE="float32"       # float32 is usually sufficient for ratio distributions
GPU="cuda:0"

# ==========================================================
# Execution
# ==========================================================

echo "-------------------------------------------------------"
echo "Launching TEMPO Attack: $TARGET_SCRIPT"
echo "Target: Layer $LAYER | Module $MODULE | Head $HEAD"
echo "-------------------------------------------------------"

python "$TARGET_SCRIPT" \
    --base_model "$BASE" \
    --finetuned_path "$FT" \
    --cache_dir "$CACHE" \
    --layer_id "$LAYER" \
    --module_name "$MODULE" \
    --head_idx "$HEAD" \
    --top_n "$VOTES" \
    --dtype "$DTYPE" \
    --device "$GPU"

echo "[*] TEMPO Analysis Complete."