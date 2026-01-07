"""
GSM8K Visualization Module.

This module handles the rendering of results within Jupyter Notebooks.
It generates:
1. A Word-Level Diff (HTML) comparing the Baseline vs. Optimized prompt.
2. A table of "Success Stories" (examples fixed by the optimization).
"""

import os
import difflib
import pandas as pd
from IPython.display import display, HTML, Markdown

# -----------------------------------------------------------------------------
# CONSTANTS & PATHS
# -----------------------------------------------------------------------------
# Paths are relative to the project root
BASELINE_PROMPT_PATH = "src/tasks/gsm8k/initial_prompt.txt"
OPTIMIZED_PROMPT_PATH = "outputs/gsm8k/optimized_prompt.txt"
RESULTS_CSV_PATH = "outputs/gsm8k/comparison_results.csv"

def render_inline_diff(text1: str, text2: str) -> str:
    """
    Generates an HTML string representing a word-level diff.
    
    Styling:
    - Insertions: Green background with bold text.
    - Deletions: Red background with strikethrough.
    """
    # Split by whitespace to compare words, not characters or lines
    a = text1.split()
    b = text2.split()
    
    matcher = difflib.SequenceMatcher(None, a, b)
    
    html_parts = []
    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
        if opcode == 'equal':
            # Unchanged text
            html_parts.append(" ".join(a[a0:a1]))
        elif opcode == 'insert':
            # Added text (Green)
            inserted = " ".join(b[b0:b1])
            html_parts.append(f'<span style="background-color: #d4fcbc; color: #376e37; padding: 2px; border-radius: 3px;"><strong>{inserted}</strong></span>')
        elif opcode == 'delete':
            # Deleted text (Red)
            deleted = " ".join(a[a0:a1])
            html_parts.append(f'<span style="background-color: #fbb6c2; color: #a30000; text-decoration: line-through; padding: 2px; border-radius: 3px;">{deleted}</span>')
        elif opcode == 'replace':
            # Replaced text (Visualized as Delete -> Insert)
            deleted = " ".join(a[a0:a1])
            inserted = " ".join(b[b0:b1])
            html_parts.append(f'<span style="background-color: #fbb6c2; color: #a30000; text-decoration: line-through; padding: 2px; border-radius: 3px;">{deleted}</span>')
            html_parts.append(f'<span style="background-color: #d4fcbc; color: #376e37; padding: 2px; border-radius: 3px;"><strong>{inserted}</strong></span>')
            
    return " ".join(html_parts)

def run_visualization():
    """
    Main execution function to display the analysis in a Notebook.
    """
    try:
        # ---------------------------------------------------------------------
        # 1. LOAD PROMPTS
        # ---------------------------------------------------------------------
        baseline_text = "[Error: Baseline file missing]"
        optimized_text = "[Error: Optimized file missing]"
        
        if os.path.exists(BASELINE_PROMPT_PATH):
            with open(BASELINE_PROMPT_PATH, "r", encoding="utf-8") as f:
                baseline_text = f.read().strip()
                
        if os.path.exists(OPTIMIZED_PROMPT_PATH):
            with open(OPTIMIZED_PROMPT_PATH, "r", encoding="utf-8") as f:
                optimized_text = f.read().strip()

        # ---------------------------------------------------------------------
        # 2. DISPLAY PROMPT DIFF
        # ---------------------------------------------------------------------
        display(Markdown("### 📝 Prompt Evolution (Word-Level Diff)"))
        display(Markdown("_**Green**: Content added by the Optimizer | **Red**: Content removed_"))
        
        diff_html = f"""
        <div style="
            font-family: 'Liberation Mono', monospace; 
            font-size: 1.1em; 
            line-height: 1.6; 
            border: 1px solid #ddd; 
            padding: 20px; 
            border-radius: 8px; 
            background-color: #f9f9f9;
            max-width: 850px; 
            margin-bottom: 20px;
        ">
            {render_inline_diff(baseline_text, optimized_text)}
        </div>
        """
        display(HTML(diff_html))

        # ---------------------------------------------------------------------
        # 3. DISPLAY SUCCESS STORIES (WINS)
        # ---------------------------------------------------------------------
        if os.path.exists(RESULTS_CSV_PATH):
            df = pd.read_csv(RESULTS_CSV_PATH)
            
            # Filter for rows where Baseline failed but Optimized succeeded
            wins = df[(df['base_is_correct'] == False) & (df['opt_is_correct'] == True)]
            
            display(Markdown(f"### 🏆 Success Stories: {len(wins)} examples fixed"))
            
            if len(wins) > 0:
                # Iterate and display top 3 wins with nice formatting
                for idx, row in wins.head(3).iterrows():
                    html_card = f"""
                    <div style="
                        border-left: 5px solid #4CAF50; 
                        background-color: #f1f8e9; 
                        padding: 15px; 
                        margin-bottom: 15px; 
                        max-width: 850px;
                        font-family: sans-serif;
                    ">
                        <div style="font-weight: bold; margin-bottom: 5px; color: #2e7d32;">📘 Question:</div>
                        <div style="margin-bottom: 10px; font-style: italic;">{row['question']}</div>
                        
                        <div style="margin-bottom: 5px;">
                            <span style="color: #c62828;">❌ <strong>Baseline Answer:</strong></span> {row['base_prediction']}
                        </div>
                        <div>
                            <span style="color: #2e7d32;">✅ <strong>Optimized Answer:</strong></span> {row['opt_prediction']} 
                            <span style="color: #555;">(GT: {row['ground_truth']})</span>
                        </div>
                    </div>
                    """
                    display(HTML(html_card))
            else:
                print("No direct flips found in the test set comparison.")
        else:
            print("⚠️ Results CSV not found. Please run evaluation first.")

    except Exception as e:
        print(f"Visualization Error: {e}")

if __name__ == "__main__":
    run_visualization()