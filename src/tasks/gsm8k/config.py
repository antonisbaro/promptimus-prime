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

import os
import torch

# -----------------------------------------------------------------------------
# OUTPUT PATHS
# -----------------------------------------------------------------------------

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

# Set to True to enable detailed logging of Teacher (Optimizer/Backward) and Student prompts
PROMPTS_VERBOSITY = False

# -----------------------------------------------------------------------------
# DATASET CONFIGURATION
# -----------------------------------------------------------------------------
# We define three distinct splits to ensure rigorous evaluation:
# 1. TRAIN: Used by the Optimizer to generate gradients and propose prompts.
# 2. VAL:   Used internally by the Trainer to validate proposals (Early Stopping/Selection).
# 3. TEST:  Used strictly for final evaluation (Baseline vs. Optimized).

# NOTE: Keep these numbers small for Google Colab Free Tier (T4 GPU).
# Increase them if running on stronger hardware (e.g., A100).
TRAIN_SIZE = 200   # Number of samples for the optimization loop
VAL_SIZE = 100      # Number of samples for validating new prompts during training
TEST_SIZE = 200    # Number of samples for the final 'evaluate.py' report

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
# MODEL LOADING PARAMETERS
# -----------------------------------------------------------------------------
# This dictionary contains special arguments passed to the .from_pretrained()
# method. It is used to handle model-specific workarounds or optimizations.

MODEL_LOAD_KWARGS = {
    "microsoft/Phi-3-mini-4k-instruct": {
        # This resolves a known compatibility issue between Phi-3 and certain
        # versions of transformers/accelerate when using `device_map="auto"`.
        "attn_implementation": "eager",
    },
    # Add other model-specific loading flags here if needed in the future.
    # "some/other-model": { "some_flag": True }
}

# -----------------------------------------------------------------------------
# GENERATION PARAMETERS
# -----------------------------------------------------------------------------
# Student Parameters:
# - Temperature 0.0: "Deterministic" output
# - max_new_tokens 1024: Enough space for step-by-step logic.
STUDENT_MODEL_KWARGS = {
    "temperature": 0.0,
    "max_new_tokens": 1024,
}

# Teacher Parameters:
# - Temperature 0.8: Allows for some creativity
# - max_new_tokens 8192: Needs space to explain the error (gradient) and rewrite the prompts.
TEACHER_MODEL_KWARGS = {
    "temperature": 0.8,
    "max_new_tokens": 8192,
}

# In src/tasks/gsm8k/config.py

# -----------------------------------------------------------------------------
# TRAINING / OPTIMIZATION HYPERPARAMETERS
# -----------------------------------------------------------------------------
# Max Steps: How many optimization iterations (generations) to run.
MAX_STEPS = 8

# -----------------------------------------------------------------------------
# STUDENT FORWARD PASS CONFIGURATION (Per Step)
# -----------------------------------------------------------------------------
# Train Batch Size: How many NEW samples the Student processes in each step.
# This determines how quickly the pool of observed errors ("the reservoir") fills up.
# A larger size fills the reservoir faster but increases step time.
TRAIN_BATCH_SIZE = 8

# Number of Workers for parallel processing during inference.
# Keep this aligned with your hardware capabilities to avoid deadlocks.
NUM_WORKERS = 4

# -----------------------------------------------------------------------------
# TEACHER BACKWARD PASS CONFIGURATION (Per Step)
# -----------------------------------------------------------------------------
# The Teacher's feedback is NOT based on the current train_batch. Instead, it's
# based on a small, focused subset sampled from ALL previously seen examples.
# These parameters control the composition of that feedback subset.

# Max Error Samples: The maximum number of FAILED examples to randomly sample
# from the reservoir to construct the feedback prompt for the Teacher.
# This is the most critical signal for improvement.
MAX_ERROR_SAMPLES = 2

# Max Correct Samples: The maximum number of CORRECT examples to randomly sample.
# These are included to give the Teacher context on what success looks like and
# to prevent it from making changes that break already working patterns.
MAX_CORRECT_SAMPLES = 2