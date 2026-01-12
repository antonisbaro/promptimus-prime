"""
GSM8K Visualization Module.

This module renders a comprehensive analysis of the LLM-AutoDiff optimization
results by comparing the performance of a baseline and an optimized prompt set.
It is designed to work with the Peer Nodes architecture.

The generated report includes:

1.  **Prompt Component Diffs:**
    - Renders a word level HTML diff for each prompt component (Instruction,
      Demos, Output Format) to visualize the exact changes made by the
      optimizer.

2.  **Performance Impact Analysis:**
    - **Success Stories:** Displays interactive cards for test cases that the
      baseline model failed but the optimized model answered correctly.
    - **Failure Stories (Regressions):** Displays interactive cards for test
      cases that the baseline model answered correctly but the optimized
      model failed, highlighting potential negative impacts of the new prompt.
    - **Collapsible Reasoning:** Each card includes a collapsible section to
      show and compare the step-by-step reasoning (`Chain-of-Thought`) of
      both models, allowing for in-depth qualitative analysis.
"""

import os
import difflib
import html
import pandas as pd
from IPython.display import display, HTML, Markdown
from src.tasks.gsm8k.config import OUTPUT_DIR

# -----------------------------------------------------------------------------
# CONSTANTS & PATHS
# -----------------------------------------------------------------------------
# Base directories
BASE_SRC_DIR = "src/tasks/gsm8k/prompts"
BASE_OUT_DIR = OUTPUT_DIR
RESULTS_CSV_PATH = os.path.join(BASE_OUT_DIR, "comparison_results.csv")

# Mapping of Logical Name -> (Source Filename, Output Filename)
PROMPT_COMPONENTS = {
    "1. Task Instruction": ("instruction.txt", "optimized_instruction.txt"),
    "2. Few-Shot Demonstrations": ("demos.txt", "optimized_demos.txt"),
    "3. Output Format": ("output_format.txt", "optimized_format.txt"),
}

def render_inline_diff(text1: str, text2: str) -> str:
    """
    Generates an HTML string representing a word-level diff.
    
    Styling:
    - Insertions: Green background with bold text.
    - Deletions: Red background with strikethrough.
    """
    # Split by whitespace to compare words
    a = text1.split()
    b = text2.split()
    
    matcher = difflib.SequenceMatcher(None, a, b)
    
    html_parts = []
    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
        if opcode == 'equal':
            html_parts.append(" ".join(a[a0:a1]))
        elif opcode == 'insert':
            inserted = " ".join(b[b0:b1])
            html_parts.append(f'<span style="background-color: #d4fcbc; color: #376e37; padding: 2px; border-radius: 3px;"><strong>{inserted}</strong></span>')
        elif opcode == 'delete':
            deleted = " ".join(a[a0:a1])
            html_parts.append(f'<span style="background-color: #fbb6c2; color: #a30000; text-decoration: line-through; padding: 2px; border-radius: 3px;">{deleted}</span>')
        elif opcode == 'replace':
            deleted = " ".join(a[a0:a1])
            inserted = " ".join(b[b0:b1])
            html_parts.append(f'<span style="background-color: #fbb6c2; color: #a30000; text-decoration: line-through; padding: 2px; border-radius: 3px;">{deleted}</span>')
            html_parts.append(f'<span style="background-color: #d4fcbc; color: #376e37; padding: 2px; border-radius: 3px;"><strong>{inserted}</strong></span>')
            
    return " ".join(html_parts)

def display_component_diff(title: str, base_filename: str, opt_filename: str):
    """
    Loads files and displays a diff box for a specific prompt component.
    """
    base_path = os.path.join(BASE_SRC_DIR, base_filename)
    opt_path = os.path.join(BASE_OUT_DIR, opt_filename)
    
    # Load content safely
    baseline_text = "[File not found]"
    optimized_text = "[File not found]"
    
    if os.path.exists(base_path):
        with open(base_path, "r", encoding="utf-8") as f: baseline_text = f.read().strip()
        
    if os.path.exists(opt_path):
        with open(opt_path, "r", encoding="utf-8") as f: optimized_text = f.read().strip()

    # Skip visualization if content is identical (optional, but keeps notebook clean)
    if baseline_text == optimized_text:
        display(Markdown(f"#### {title}: *No Changes*"))
        return

    display(Markdown(f"#### {title}"))
    
    diff_html = f"""
    <div style="
        font-family: 'Liberation Mono', monospace; 
        font-size: 1.0em; 
        line-height: 1.6; 
        border: 1px solid #ddd; 
        padding: 15px; 
        border-radius: 6px; 
        background-color: #f9f9f9;
        max-width: 850px; 
        margin-bottom: 20px;
    ">
        {render_inline_diff(baseline_text, optimized_text)}
    </div>
    """
    display(HTML(diff_html))

def run_visualization():
    """
    Main execution function to display the analysis in a Notebook.
    """
    try:
        display(Markdown("## 📝 Prompt Evolution (Peer Nodes)"))
        display(Markdown("_**Green**: Content added by the Optimizer | **Red**: Content removed_"))
        print("-" * 80)

        # ---------------------------------------------------------------------
        # 1. ITERATE AND DISPLAY DIFFS FOR ALL PEERS
        # ---------------------------------------------------------------------
        for title, (base_file, opt_file) in PROMPT_COMPONENTS.items():
            display_component_diff(title, base_file, opt_file)

        # ---------------------------------------------------------------------
        # 2. ANALYZE CSV RESULTS
        # ---------------------------------------------------------------------
        display(Markdown("---")) # Separator
        display(Markdown("### Performance Impact on Test Set"))

        if os.path.exists(RESULTS_CSV_PATH):
            df = pd.read_csv(RESULTS_CSV_PATH)
            
            # --- Success Stories (Wins) ---
            wins = df[(df['base_is_correct'] == False) & (df['opt_is_correct'] == True)]
            display(Markdown(f"#### 🏆 Success Stories: {len(wins)} examples fixed by optimization"))
            
            if len(wins) > 0:
                for idx, row in wins.head(5).iterrows():
                    # Sanitize HTML content before displaying
                    question = html.escape(str(row['question']))
                    base_reasoning = html.escape(str(row['base_reasoning'])).replace('\n', '<br>')
                    opt_reasoning = html.escape(str(row['opt_reasoning'])).replace('\n', '<br>')
                    
                    # Unique ID for the collapsible element
                    details_id = f"win-{idx}"

                    html_card = f"""
                    <div style="border-left: 5px solid #4CAF50; background-color: #f1f8e9; padding: 15px; margin-bottom: 15px; max-width: 850px; font-family: sans-serif;">
                        <div style="font-weight: bold; margin-bottom: 5px; color: #2e7d32;">📘 Question:</div>
                        <div style="margin-bottom: 10px; font-style: italic;">{question}</div>
                        <div style="margin-bottom: 5px;">
                            <span style="color: #c62828;">❌ <strong>Baseline Answer:</strong></span> {row['base_prediction']}
                        </div>
                        <div>
                            <span style="color: #2e7d32;">✅ <strong>Optimized Answer:</strong></span> {row['opt_prediction']} 
                            <span style="color: #555;">(Correct: {row['ground_truth']})</span>
                        </div>
                        <details style="margin-top: 10px;">
                            <summary style="cursor: pointer; color: #555; font-size: 0.9em;">Show/Hide Reasoning</summary>
                            <div style="background-color: #e8f5e9; padding: 10px; margin-top: 5px; border-radius: 4px; font-family: monospace; font-size: 0.9em;">
                                <strong>Baseline Reasoning:</strong><br>{base_reasoning}<br><br>
                                <strong>Optimized Reasoning:</strong><br>{opt_reasoning}
                            </div>
                        </details>
                    </div>
                    """
                    display(HTML(html_card))
            else:
                print("No examples were fixed by the optimization.")

            # --- Failure Stories (Regressions) ---
            regressions = df[(df['base_is_correct'] == True) & (df['opt_is_correct'] == False)]
            display(Markdown(f"#### 📉 Failure Stories (Regressions): {len(regressions)} examples broken by optimization"))
            
            if len(regressions) > 0:
                for idx, row in regressions.head(5).iterrows():
                    question = html.escape(str(row['question']))
                    base_reasoning = html.escape(str(row['base_reasoning'])).replace('\n', '<br>')
                    opt_reasoning = html.escape(str(row['opt_reasoning'])).replace('\n', '<br>')
                    
                    html_card = f"""
                    <div style="border-left: 5px solid #f44336; background-color: #ffebee; padding: 15px; margin-bottom: 15px; max-width: 850px; font-family: sans-serif;">
                        <div style="font-weight: bold; margin-bottom: 5px; color: #c62828;">📘 Question:</div>
                        <div style="margin-bottom: 10px; font-style: italic;">{question}</div>
                        <div style="margin-bottom: 5px;">
                            <span style="color: #2e7d32;">✅ <strong>Baseline Answer:</strong></span> {row['base_prediction']}
                        </div>
                        <div>
                            <span style="color: #c62828;">❌ <strong>Optimized Answer:</strong></span> {row['opt_prediction']} 
                            <span style="color: #555;">(Correct: {row['ground_truth']})</span>
                        </div>
                         <details style="margin-top: 10px;">
                            <summary style="cursor: pointer; color: #555; font-size: 0.9em;">Show/Hide Reasoning</summary>
                            <div style="background-color: #ffcdd2; padding: 10px; margin-top: 5px; border-radius: 4px; font-family: monospace; font-size: 0.9em;">
                                <strong>Baseline Reasoning:</strong><br>{base_reasoning}<br><br>
                                <strong>Optimized Reasoning:</strong><br>{opt_reasoning}
                            </div>
                        </details>
                    </div>
                    """
                    display(HTML(html_card))
            else:
                print("No correct examples were broken by the optimization. Great!")

        else:
            print(f"⚠️ Results CSV not found at '{RESULTS_CSV_PATH}'. Please run the evaluation script first.")

    except Exception as e:
        print(f"Visualization Error: {e}")

if __name__ == "__main__":
    run_visualization()