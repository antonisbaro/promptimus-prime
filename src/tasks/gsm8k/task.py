"""
GSM8K Task Definition with Peer Nodes Architecture.

This module implements the full LLM-AutoDiff architecture by decomposing the 
system prompt into three distinct, optimizable peer nodes:
1.  **Instruction:** The core task definition.
2.  **Demos:** Few-shot demonstrations (in-context learning).
3.  **Output Format:** Specific formatting constraints.

This modular approach allows the optimizer to target specific aspects of the 
prompt generation independently. The module also defines the output parser 
and the Student Component that orchestrates these parameters.
"""

import re
import os
from typing import Any, Dict, Optional

import adalflow as adal
from adalflow.core.types import GeneratorOutput

# -----------------------------------------------------------------------------
# OUTPUT PARSER
# -----------------------------------------------------------------------------
@adal.func_to_data_component
def parse_gsm8k_answer(output: Any) -> str:
    """
    Robust parser optimized for English Math Datasets (GSM8K).
    
    It extracts the final numerical answer from the model's output, handling:
    - Integers inside LaTeX (e.g., \boxed{15})
    - Thousands separators (e.g., 1,000).
    - Trailing punctuation (e.g., 16.).
    - AdalFlow GeneratorOutput objects or raw strings.

    Args:
        output (Any): The raw output from the generator (string or GeneratorOutput).

    Returns:
        str: The extracted number as a string, or empty string if not found.
    """
    # Get Text
    text = ""
    if hasattr(output, "data"):
        text = output.data if output.data else ""
    else:
        text = str(output)

    if not text:
        return ""

    # Clean Commas
    clean_text = text.replace(",", "")

    # Find ALL numbers
    # Matches optional negative sign, digits, and optional decimal part
    numbers = re.findall(r"-?\d+\.?\d*", clean_text)

    # Return the last one found
    if numbers:
        return numbers[-1].rstrip(".")
    
    return ""

# -----------------------------------------------------------------------------
# STUDENT COMPONENT
# -----------------------------------------------------------------------------
class GSM8KStudent(adal.Component):
    """
    The Student Component implementing the Peer Nodes architecture for GSM8K.

    This component wraps the LLM generator and manages three distinct, 
    trainable parameters (Instruction, Demos, Output Format). 
    
    It handles:
    1.  **Initialization:** Loading initial prompt states from external text files 
        located in the `src/tasks/gsm8k/prompts/` directory.
    2.  **Optimization:** Exposing these parameters to the AdalFlow optimizer via `requires_opt=True`.
    3.  **Generation:** Assembling the peer nodes and user input into a structured 
        XML-like template for the forward pass.
    """
    def __init__(self, student_client: adal.ModelClient, model_kwargs: Dict):
        """
        Initialize the Student with Peer Nodes.

        Sets up the three optimizable parameters (`instruction`, `demos`, `output_format`)
        by reading their initial content from the file system. If files are missing,
        safe defaults are provided.

        Args:
            student_client (ModelClient): The backend client (e.g., LocalLLMClient).
            model_kwargs (Dict): Generation parameters (temperature, max_tokens, etc.).
        """
        super().__init__()

        # Helper to load text files safely
        def load_prompt_file(filename: str, default: str) -> str:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(current_dir, "prompts", filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except FileNotFoundError:
                print(f"⚠️ Warning: {filename} not found. Using default.")
                return default

        # Define Peer Parameters
        # Peer 1: Core Instruction
        self.instruction = adal.Parameter(
            data=load_prompt_file("instruction.txt", "You are a helpful math assistant. Solve the problem step by step."),
            role_desc="Task Instruction",
            requires_opt=True,
            param_type=adal.ParameterType.PROMPT,
            name="instruction"
        )

        # Peer 2: Few-Shot Demonstrations (The most important for bigger gains)
        self.demos = adal.Parameter(
            data=load_prompt_file("demos.txt", ""),
            role_desc="Few-shot Examples",
            requires_opt=True,
            param_type=adal.ParameterType.PROMPT,
            name="demos"
        )

        # Peer 3: Output Formatting
        self.output_format = adal.Parameter(
            data=load_prompt_file("output_format.txt", "Finish your answer with exactly: 'Answer: X' where X is the number."),
            role_desc="Output Format formatting requirements",
            requires_opt=True,
            param_type=adal.ParameterType.PROMPT,
            name="output_format"
        )

        # Initialize Generator with Compound Template
        # We explicitly structure the prompt using the three peers.
        self.generator = adal.Generator(
            model_client=student_client,
            model_kwargs=model_kwargs,
            template="""<INSTRUCTION>
{{instruction}}
</INSTRUCTION>

<EXAMPLES>
{{demos}}
</EXAMPLES>

<FORMAT>
{{output_format}}
</FORMAT>

<USER_INPUT>
{{input_str}}
</USER_INPUT>""",
            output_processors=parse_gsm8k_answer,
            use_cache=False
        )

    def call(self, question: str, id: str = None) -> GeneratorOutput:
        """
        Executes the Forward Pass.

        Injects the current state of all three peer parameters (Instruction, Demos, Format)
        along with the user question into the generator template.

        Args:
            question (str): The math problem to solve.
            id (str, optional): The unique sample ID.

        Returns:
            GeneratorOutput: The model's raw response and parsed data.
        """
        return self.generator(
            prompt_kwargs={
                "instruction": self.instruction.data,
                "demos": self.demos.data,
                "output_format": self.output_format.data,
                "input_str": question
            },
            id=id
        )