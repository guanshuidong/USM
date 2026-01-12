#!/bin/bash

# ==========================================================
# Configuration for ALS-based Column Matching
# ==========================================================

# 1. Model Selection
# Choose which script to run: gpt2.py, llama.py, or bert.py
TARGET_SCRIPT="gpt2.py"

# 2. Path Setup (GPT-2 Example)
BASE_MODEL="gpt2"
FT_PATH="gpt2_sst2"
CACHE="models"

# 3. Module Configuration
# GPT-2 Modules: attn.c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj
LAYER=11
MODULE="attn.c_proj"

# 4. Hyperparameters
ATTACKS=10
DTYPE="float64"  # float64 is recommended for ALS convergence
GPU="cuda:0"

# ==========================================================
# Execution
# ==========================================================

echo "-------------------------------------------------------"
echo "Starting ALS Attack using $TARGET_SCRIPT"
echo "Layer: $LAYER | Module: $MODULE"
echo "-------------------------------------------------------"

python "$TARGET_SCRIPT" \
    --base_model "$BASE_MODEL" \
    --finetuned_path "$FT_PATH" \
    --cache_dir "$CACHE" \
    --layer_id "$LAYER" \
    --module_name "$MODULE" \
    --attack_number "$ATTACKS" \
    --dtype "$DTYPE" \
    --device "$GPU"

echo "[*] ALS Analysis Complete."