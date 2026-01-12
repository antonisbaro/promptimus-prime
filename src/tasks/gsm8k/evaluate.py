"""
GSM8K Comparative Evaluation Script.

This module executes a rigorous A/B test between the Baseline (Zero-Shot) configuration 
and the Optimized (Text-Grad) configuration on the held-out Test Set.

It evaluates the performance of the full Peer Node system (Instruction + Demos + Format)
and generates a detailed CSV report for qualitative analysis of the improvements.

Usage:
    Run as a module from the project root:
    $ python -m src.tasks.gsm8k.evaluate
"""

import logging
import os
from typing import Dict
import pandas as pd
from typing import Tuple
from tqdm import tqdm

import adalflow as adal
from adalflow.datasets.gsm8k import GSM8K
from adalflow.eval.answer_match_acc import AnswerMatchAcc

from src.core.client import LocalLLMClient
from src.tasks.gsm8k.config import (
    STUDENT_MODEL_NAME, 
    STUDENT_MODEL_KWARGS, 
    TEST_SIZE, OUTPUT_DIR
)
from src.tasks.gsm8k.task import GSM8KStudent

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("transformers").setLevel(logging.ERROR)

def evaluate_prompt(client: LocalLLMClient, 
                    dataset: GSM8K, 
                    prompt_state: Dict[str, str], 
                    run_name: str = "Run") -> Tuple[float, pd.DataFrame]:
    """
    Executes inference on the dataset using a specific configuration of Peer Nodes.

    It injects the provided state (instructions, demos, format) into the Student task, 
    runs the forward pass, computes accuracy, and logs detailed execution traces.

    Args:
        client (LocalLLMClient): The initialized LLM client wrapper.
        dataset (GSM8K): The dataset split (Test Set) to evaluate on.
        prompt_state (Dict[str, str]): A dictionary mapping parameter names ('instruction', 
                                       'demos', 'output_format') to their text values.
        run_name (str): Label for this evaluation run (e.g., 'Baseline', 'Optimized').

    Returns:
        Tuple[float, pd.DataFrame]: 
            - The accuracy score (0.0 to 1.0).
            - A Pandas DataFrame containing detailed I/O logs (Question, Prediction, GT) for every sample.
    """
    print(f"\n📊 EVALUATING: {run_name}")
    print(f"ℹ️  Set Size: {len(dataset)}")
    
    # Initialize the Task Component with the specific prompt parameters
    task = GSM8KStudent(student_client=client, model_kwargs=STUDENT_MODEL_KWARGS)

    # Inject State (Injecting the specific prompts we want to test)
    if 'instruction' in prompt_state:
        task.instruction.data = prompt_state['instruction']
    if 'demos' in prompt_state:
        task.demos.data = prompt_state['demos']
    if 'output_format' in prompt_state:
        task.output_format.data = prompt_state['output_format']
    
    # Initialize Metric
    eval_fn = AnswerMatchAcc(type="exact_match").compute_single_item
    
    correct_count = 0
    total_count = len(dataset)
    results = []

    # Iterate through the dataset
    for sample in tqdm(dataset, desc=f"Inference {run_name}"):
        try:
            # 1. Forward Pass
            output = task.call(sample.question)
            
            # 2. Extract Data
            parsed_answer = output.data
            ground_truth = sample.answer
            raw_reasoning = output.raw_response
            
            # 3. Evaluate
            score = eval_fn(parsed_answer, ground_truth)
            correct_count += int(score)
            
            # 4. Record Detailed Results
            results.append({
                "question": sample.question,
                "ground_truth": ground_truth,
                "prediction": parsed_answer,
                "reasoning": raw_reasoning,
                "is_correct": bool(score)
            })
            
        except Exception as e:
            logging.error(f"Error processing sample ID {sample.id}: {e}")
            # Append a failure record to keep DataFrame alignment
            results.append({
                "question": sample.question,
                "ground_truth": sample.answer,
                "prediction": "ERROR",
                "reasoning": str(e),
                "is_correct": False
            })

    # Calculate metrics
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    print(f"📈 Accuracy: {accuracy:.2%} ({correct_count}/{total_count})")
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    return accuracy, df_results

def run_evaluation():
    """
    Orchestrates the comparative evaluation workflow.

    Workflow:
    1. **Initialization:** Sets up the Student model client.
    2. **Data Loading:** Loads the strict Test split (unseen during training).
    3. **Baseline Eval:** Loads the initial prompts from `src/tasks/gsm8k/prompts/` and evaluates performance.
    4. **Optimized Eval:** Loads the trained prompts from `outputs/gsm8k/` and evaluates performance.
    5. **Reporting:** 
       - Calculates and prints the accuracy delta (Improvement).
       - Merges detailed logs into `comparison_results.csv` for side-by-side analysis.
    """
    print(f"🚀 Initializing Evaluation Client...")
    client = LocalLLMClient(model_name=STUDENT_MODEL_NAME)

    # -------------------------------------------------------------------------
    # DATA LOADING (TEST SPLIT)
    # -------------------------------------------------------------------------
    print(f"📚 Loading TEST Dataset...")
    # We use the strict 'test' split for final reporting to ensure no data leakage.
    test_data = GSM8K(split="test", size=TEST_SIZE)

    # -------------------------------------------------------------------------
    # 1. BASELINE EVALUATION (Default State)
    # -------------------------------------------------------------------------
    # Helper to load source
    def load_src(name):
        with open(f"src/tasks/gsm8k/prompts/{name}", "r") as f: return f.read().strip()

    baseline_state = {
        "instruction": load_src("instruction.txt"),
        "demos": load_src("demos.txt"),
        "output_format": load_src("output_format.txt")
    }
    
    acc_baseline, df_baseline = evaluate_prompt(
        client, test_data, baseline_state, run_name="Baseline"
    )

    # -------------------------------------------------------------------------
    # 2. OPTIMIZED EVALUATION (Optimized State)
    # -------------------------------------------------------------------------
    
    # Check if files exist
    if os.path.exists(os.path.join(OUTPUT_DIR, "optimized_instruction.txt")):
        
        def load_opt(name):
            with open(os.path.join(OUTPUT_DIR, name), "r") as f: return f.read().strip()

        optimized_state = {
            "instruction": load_opt("optimized_instruction.txt"),
            "demos": load_opt("optimized_demos.txt"),
            "output_format": load_opt("optimized_format.txt")
        }
        
        acc_optimized, df_optimized = evaluate_prompt(
            client, test_data, optimized_state, run_name="Optimized"
        )
        
        # ---------------------------------------------------------------------
        # 3. SAVE COMPARISON REPORT
        # ---------------------------------------------------------------------
        # The 'question' and 'ground_truth' columns are identical for both runs.
        # We can take them from the baseline DataFrame and use them as our base.
        # We select only the columns that are common to both evaluations.
        base_info_df = df_baseline[['question', 'ground_truth']].copy()

        # Rename the columns that are specific to each run BEFORE merging.
        df_baseline_renamed = df_baseline.rename(columns={
            'prediction': 'base_prediction',
            'reasoning': 'base_reasoning',
            'is_correct': 'base_is_correct'
        })
        
        df_optimized_renamed = df_optimized.rename(columns={
            'prediction': 'opt_prediction',
            'reasoning': 'opt_reasoning',
            'is_correct': 'opt_is_correct'
        })
        
        # Now, merge everything into a single, clean DataFrame.
        # We merge based on the index, as the order of samples is the same.
        comparison_df = pd.concat([
            base_info_df,
            df_baseline_renamed[['base_prediction', 'base_reasoning', 'base_is_correct']],
            df_optimized_renamed[['opt_prediction', 'opt_reasoning', 'opt_is_correct']]
        ], axis=1)
        
        output_csv = "outputs/gsm8k/comparison_results.csv"
        comparison_df.to_csv(output_csv, index=False)
        print(f"\n💾 Saved detailed comparison report to: {output_csv}")

        # ---------------------------------------------------------------------
        # 4. FINAL SUMMARY
        # ---------------------------------------------------------------------
        print("\n" + "█"*50)
        print("               RESULTS SUMMARY               ")
        print("█"*50)
        print(f"Test Set Size:      {TEST_SIZE}")
        print(f"Baseline Accuracy:  {acc_baseline:.2%}")
        print(f"Optimized Accuracy: {acc_optimized:.2%}")
        diff = acc_optimized - acc_baseline
        print(f"Improvement:        {'+' if diff >= 0 else ''}{diff:.2%}")
        print("█"*50)
    else:
        print(f"\n⚠️ Optimized artifacts not found.")
        print("   Please run 'train.py' first to generate the optimized artifacts.")

if __name__ == "__main__":
    run_evaluation()