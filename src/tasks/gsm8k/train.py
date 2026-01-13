"""
GSM8K Training Execution Script.

This module implements the end-to-end Textual Gradient Descent (TGD) optimization loop 
for the GSM8K reasoning task. It utilizes the LLM-AutoDiff architecture with **Peer Nodes**, 
optimizing three distinct components simultaneously:
1. Task Instruction
2. Few-Shot Demonstrations
3. Output Format

Key Features:
- **Resumable Training:** Automatically detects and loads the latest checkpoint from `outputs/gsm8k/ckpt`.
- **Deterministic Splits:** Enforces strict Train/Validation separation based on configuration.
- **Artifact Persistence:** Saves the final optimized state of all peer nodes to individual text files.

Usage:
    Run as a module from the project root:
    $ python -m src.tasks.gsm8k.train
"""

import logging
import os
import glob

import adalflow.optim.text_grad.tgd_optimizer as tgd_optimizer_module
from src.core.custom_components import RobustXMLParser

# We replace the original, brittle parser class in the library's
# module with our new, robust class.
tgd_optimizer_module.CustomizedXMLParser = RobustXMLParser

import adalflow as adal
from adalflow.datasets.gsm8k import GSM8K
from adalflow.optim.text_grad.tgd_optimizer import TGDOptimizer
from sklearn.model_selection import train_test_split


# Import Core Infrastructure
from src.core.client import LocalLLMClient

# Import Task-Specific Logic & Configuration
from src.tasks.gsm8k.config import (
    STUDENT_MODEL_NAME, TEACHER_MODEL_NAME, 
    STUDENT_MODEL_KWARGS, TEACHER_MODEL_KWARGS,
    TRAIN_BATCH_SIZE, NUM_WORKERS, MAX_STEPS, 
    TRAIN_SIZE, VAL_SIZE, TEST_SIZE,
    MAX_ERROR_SAMPLES, MAX_CORRECT_SAMPLES,
    CKPT_DIR, OUTPUT_DIR, SEED
)
from src.tasks.gsm8k.task import GSM8KStudent
from src.tasks.gsm8k.pipeline import GSM8KTrainingPipeline

# Configure Logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("transformers").setLevel(logging.ERROR)

def get_latest_checkpoint():
    """
    Retrieves the most recent checkpoint file to enable auto-resumption of training.

    It recursively searches the configured `CKPT_DIR` for JSON files and selects 
    the one with the latest creation timestamp. This allows the process to recover 
    gracefully from interruptions (e.g., Colab timeouts).

    Returns:
        str or None: The absolute path to the latest checkpoint file, or None if no checkpoints exist.
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
    Orchestrates the complete optimization experiment.

    Workflow:
    1. **Initialization:** Sets up 4-bit Quantized Student and Teacher models.
    2. **Component Assembly:** Initializes the `GSM8KStudent` with Peer Nodes and connects it 
       to the `GSM8KTrainingPipeline` and `TGDOptimizer`.
    3. **Data Preparation:** Loads and splits the dataset into strict Train and Validation sets.
    4. **Execution:** Instantiates the `adal.Trainer` with custom checkpoint paths and executes 
       the training loop (with resume capability).
    5. **Serialization:** Exports the final optimized prompts (Instruction, Demos, Format) 
       to the `outputs/gsm8k` directory for evaluation.
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
    initial_instruction = student_task.instruction.data
    initial_demos = student_task.demos.data
    initial_format = student_task.output_format.data

    print(f"🛠️  Building Training Pipeline...")
    pipeline = GSM8KTrainingPipeline(
        student_task=student_task,
        teacher_client=teacher_client,
        teacher_model_kwargs=TEACHER_MODEL_KWARGS
    )

    print(f"🧠 Setting up Optimizer...")
    optimizer = TGDOptimizer(
        params=student_task.parameters(), 
        model_client=teacher_client,      # The Teacher generates the updates
        model_kwargs=TEACHER_MODEL_KWARGS
    )

    # -------------------------------------------------------------------------
    # 3. DATA LOADING 
    # -------------------------------------------------------------------------
    # This section implements the most robust data splitting strategy to
    # prevent any form of data leakage and ensure a scientifically sound experiment.

    # PROBLEM: Pre-trained models (like the Student) have likely seen the
    # official GSM8K 'train' split. Using it for generating feedback (training)
    # can result in too few errors, starving the Teacher of useful signals.

    # SOLUTION: We derive ALL our splits (train, validation, and test) from the
    # official 'test' split, which is guaranteed to be unseen by the models.
    # This ensures that the Student is consistently challenged, generating a
    # rich pool of errors for the Teacher to learn from.
    print(f"📚 Loading Datasets...")
    
    # Load a large, single pool of unseen data from the official 'test' split.
    # The size is the sum of all sets we intend to create.
    unseen_data = GSM8K(split="test", size=TRAIN_SIZE+VAL_SIZE+TEST_SIZE)

    # First, create the final, held-out Test Set.
    # We deterministically split off `TEST_SIZE` samples. These will not be touched
    # until the final evaluation. `random_state=SEED` ensures reproducibility.
    temp_data, test_data = train_test_split(
        unseen_data,
        test_size=TEST_SIZE,
        random_state=SEED
    )

    # Second, split the remaining pool into our new, clean Train and Validation sets.
    # We again use the same SEED to ensure this inner split is also reproducible.
    train_data, val_data = train_test_split(
        temp_data,
        test_size=VAL_SIZE, 
        random_state=SEED
    )

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
        num_workers=NUM_WORKERS,
        train_batch_size=TRAIN_BATCH_SIZE,
        max_error_samples=MAX_ERROR_SAMPLES,
        max_correct_samples=MAX_CORRECT_SAMPLES,
        ckpt_path=CKPT_DIR
    )

    # Resume Logic
    resume_ckpt = get_latest_checkpoint()

    print(f"\n🏁 STARTING TRAINING (Steps: {MAX_STEPS})...")
    print(f"📂 Checkpoints will be saved to: {CKPT_DIR}")

    if resume_ckpt:
        print(f"⏩ Resuming from checkpoint...")
    else:
        print(f"📜 INITIAL INSTRUCTION:\n{initial_instruction}\n")
        print(f"🔢 INITIAL DEMOS:\n{initial_demos}\n")
        print(f"✍️ INITIAL FORMAT:\n{initial_format}\n")
    
    # Start the optimization loop.
    # This modifies student_task.system_prompt in-place.
    trainer.fit(
        train_dataset=train_data, 
        val_dataset=val_data, 
        resume_from_ckpt=resume_ckpt,
        debug=False
    )

    # -------------------------------------------------------------------------
    # 5. SAVE ARTIFACTS
    # -------------------------------------------------------------------------
    print("\n✅ TRAINING COMPLETE!")
    print(f"📜 FINAL OPTIMIZED INSTRUCTION:\n{student_task.instruction.data}\n")
    print(f"🔢 FINAL OPTIMIZED DEMOS:\n{student_task.demos.data}\n")
    print(f"✍️ FINAL OPTIMIZED FORMAT:\n{student_task.output_format.data}\n")

    # Dictionary mapping filenames to parameter data
    artifacts = {
        "optimized_instruction.txt": student_task.instruction.data,
        "optimized_demos.txt": student_task.demos.data,
        "optimized_format.txt": student_task.output_format.data
    }
    
    for filename, content in artifacts.items():
        file_path = os.path.join(OUTPUT_DIR, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"\n💾 Saved: {file_path}")

if __name__ == "__main__":
    run_training()