"""
GSM8K Training Script.

This script executes the Text-Grad optimization loop. It:
1. Initializes the Student and Teacher models using the custom LocalLLMClient.
2. Loads the GSM8K dataset and performs a deterministic split (Train/Val).
3. Configures the AdalFlow Trainer with the TGD Optimizer.
4. Runs the training process.
5. Saves the optimized system prompt to a file for later evaluation.

Usage:
    Run this script from the project root:
    python -m src.tasks.gsm8k.train
"""

import logging
import os
import glob
import adalflow as adal
from adalflow.datasets.gsm8k import GSM8K
from adalflow.optim.text_grad.tgd_optimizer import TGDOptimizer

# Import Core Infrastructure
from src.core.client import LocalLLMClient

# Import Task-Specific Logic & Configuration
from src.tasks.gsm8k.config import (
    STUDENT_MODEL_NAME, TEACHER_MODEL_NAME, 
    STUDENT_MODEL_KWARGS, TEACHER_MODEL_KWARGS,
    TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, MAX_STEPS, 
    TRAIN_SIZE, VAL_SIZE,
    CKPT_DIR, OUTPUT_DIR
)
from src.tasks.gsm8k.task import GSM8KStudent
from src.tasks.gsm8k.pipeline import GSM8KTrainingPipeline

# Configure Logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("transformers").setLevel(logging.ERROR)

def get_latest_checkpoint():
    """
    Checks the CUSTOM checkpoint directory defined in config.
    """
    # Look in our project folder: outputs/gsm8k/ckpt/  
    if not os.path.exists(CKPT_DIR):
        return None
        
    # Search for all json files in the checkpoint tree
    files = glob.glob(os.path.join(CKPT_DIR, "**", "*.json"), recursive=True)
    
    if not files:
        return None
        
    # Get the most recent file
    latest_file = max(files, key=os.path.getctime)
    print(f"🔄 Found checkpoint to resume: {latest_file}")
    return latest_file

def run_training():
    """
    Main function to setup and run the optimization experiment.
    """
    # -------------------------------------------------------------------------
    # 1. INITIALIZE MODELS
    # -------------------------------------------------------------------------
    # These clients handle 4-bit loading and strict chat templating.
    print("👨‍🎓 Initializing Student Client...")
    student_client = LocalLLMClient(model_name=STUDENT_MODEL_NAME)
    print("👨‍🏫 Initializing Teacher Client...")
    teacher_client = LocalLLMClient(model_name=TEACHER_MODEL_NAME)

    # -------------------------------------------------------------------------
    # 2. SETUP COMPONENTS
    # -------------------------------------------------------------------------
    print(f"🧮 Initializing Student Task...")
    student_task = GSM8KStudent(
        student_client=student_client,
        model_kwargs=STUDENT_MODEL_KWARGS
    )
    
    # Capture initial state for comparison
    initial_prompt = student_task.system_prompt.data

    print(f"🛠️  Building Training Pipeline...")
    pipeline = GSM8KTrainingPipeline(
        student_task=student_task,
        teacher_client=teacher_client,
        teacher_model_kwargs=TEACHER_MODEL_KWARGS
    )

    print(f"🧠 Setting up Optimizer...")
    optimizer = TGDOptimizer(
        params=student_task.parameters(), # Target the 'system_prompt'
        model_client=teacher_client,      # The Teacher generates the updates
        model_kwargs=TEACHER_MODEL_KWARGS
    )

    # -------------------------------------------------------------------------
    # 3. DATA LOADING 
    # -------------------------------------------------------------------------
    print(f"📚 Loading Datasets...")
    
    # Load strict Train split
    train_data = GSM8K(split="train", size=TRAIN_SIZE)
    
    # Load strict Val split
    val_data = GSM8K(split="val", size=VAL_SIZE)

    print(f"📊 Splits Loaded:")
    print(f"   - Train Set: {len(train_data)} samples")
    print(f"   - Val Set:   {len(val_data)} samples")

    # -------------------------------------------------------------------------
    # 4. TRAINER SETUP & EXECUTION
    # -------------------------------------------------------------------------
    # Ensure directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    trainer = adal.Trainer(
        adaltask=pipeline,
        optimizer=optimizer,
        strategy="random", 
        max_steps=MAX_STEPS,       
        batch_size=VAL_BATCH_SIZE,
        train_batch_size=TRAIN_BATCH_SIZE,
        ckpt_path=CKPT_DIR
    )

    # Resume Logic
    resume_ckpt = get_latest_checkpoint()

    print("\n🏁 STARTING TRAINING (Steps: {MAX_STEPS})...")
    print(f"📂 Checkpoints will be saved to: {CKPT_DIR}")

    if resume_ckpt:
        print(f"⏩ Resuming from checkpoint...")
    else:
        print(f"📜 INITIAL PROMPT:\n{initial_prompt}\n")
    
    # Start the optimization loop.
    # This modifies student_task.system_prompt in-place.
    trainer.fit(
        train_dataset=train_data, 
        val_dataset=val_data, 
        debug=False
    )

    # -------------------------------------------------------------------------
    # 5. SAVE ARTIFACTS
    # -------------------------------------------------------------------------
    final_prompt = student_task.system_prompt.data
    print("\n✅ TRAINING COMPLETE!")
    print(f"📜 FINAL OPTIMIZED PROMPT:\n{final_prompt}")
    
    output_file = os.path.join(OUTPUT_DIR, "optimized_prompt.txt")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_prompt)
    
    print(f"\n💾 Saved optimized prompt to: {output_file}")

if __name__ == "__main__":
    run_training()