"""
Configuration file for the GSM8K Task.

This file acts as the single source of truth for:
1. Model Identifiers (Student/Teacher)
2. Generation Hyperparameters (Temperature, Tokens)
3. Dataset Split Sizes (Train, Val, Test)
4. Optimization/Training Loop Parameters

Usage:
    from src.tasks.gsm8k.config import STUDENT_MODEL_NAME, TRAIN_SIZE, ...
"""

# -----------------------------------------------------------------------------
# OUTPUT PATHS
# -----------------------------------------------------------------------------
import os

# Base directory for all GSM8K artifacts
OUTPUT_DIR = "outputs/gsm8k"

# Directory specifically for AdalFlow checkpoints
# This makes them visible in the project folder instead of hidden in /root/.adalflow
CKPT_DIR = os.path.join(OUTPUT_DIR, "ckpt")

# -----------------------------------------------------------------------------
# GLOBAL SETTINGS
# -----------------------------------------------------------------------------
# Setting a seed ensures reproducibility for dataset shuffling and splitting.
SEED = 42

# -----------------------------------------------------------------------------
# DATASET CONFIGURATION
# -----------------------------------------------------------------------------
# We define three distinct splits to ensure rigorous evaluation:
# 1. TRAIN: Used by the Optimizer to generate gradients and propose prompts.
# 2. VAL:   Used internally by the Trainer to validate proposals (Early Stopping/Selection).
# 3. TEST:  Used strictly for final evaluation (Baseline vs. Optimized).

# NOTE: Keep these numbers small for Google Colab Free Tier (T4 GPU).
# Increase them if running on stronger hardware (e.g., A100).
TRAIN_SIZE = 25   # Number of samples for the optimization loop
VAL_SIZE = 25      # Number of samples for validating new prompts during training
TEST_SIZE = 50    # Number of samples for the final 'evaluate.py' report

# -----------------------------------------------------------------------------
# MODEL CONFIGURATION
# -----------------------------------------------------------------------------
# Student: The model attempting to solve the math problems.
# We use a lightweight 1.5B model for faster iteration and low VRAM usage.
STUDENT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Teacher: The "Backward Engine" and "Optimizer".
# We use a stronger 7B model to provide high-quality feedback and prompt edits.
TEACHER_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# -----------------------------------------------------------------------------
# GENERATION PARAMETERS
# -----------------------------------------------------------------------------
# Student Parameters:
# - Temperature 0.5: Allows for some creativity in Chain-of-Thought reasoning.
# - max_new_tokens 1024: Enough space for step-by-step logic.
STUDENT_MODEL_KWARGS = {
    "temperature": 0.5,
    "max_new_tokens": 1024,
}

# Teacher Parameters:
# - Temperature 0.4: Slightly lower to reduce hallucinations/leakage.
# - max_new_tokens 8192: Needs space to explain the error (gradient) and rewrite the prompts.
TEACHER_MODEL_KWARGS = {
    "temperature": 0.4,
    "max_new_tokens": 8192,
}

# -----------------------------------------------------------------------------
# TRAINING / OPTIMIZATION HYPERPARAMETERS
# -----------------------------------------------------------------------------
# Max Steps: How many optimization iterations (generations) to run.
MAX_STEPS = 10  

# Train Batch Size (Optimization Logic):
# Determines how many training samples are accumulated before the Optimizer performs an update.
# - If set to 1 (Stochastic Mode): The Teacher critiques a single sample. If incorrect, 
# it proposes an immediate prompt update. Best for fast, granular adaptation.
# - If set > 1 (Mini-Batch Mode): The Teacher aggregates feedback from N samples 
# and proposes one holistic update. Best for stability, but requires a larger context window.
TRAIN_BATCH_SIZE = 2

# Validation Batch Size (Hardware Execution):
# Strictly controls inference throughput and VRAM usage during the evaluation phase.
# - Unlike Train Batch Size, this does NOT affect the optimization logic or results.
# - It simply defines how many validation queries are processed in parallel on the GPU.
# Keep it low for T4 GPUs to prevent Out-Of-Memory (OOM) errors.
VAL_BATCH_SIZE = 4