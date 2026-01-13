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
            # General description of the parameter's role.
            role_desc="Defines the agent's high-level persona and core mission (e.g., 'You are a math expert').",
            # Specific, high-priority command for the optimizer.
            instruction_to_optimizer=(
                "Your goal is to refine the core instruction for the agent. "
                "Focus on defining a clear persona and a robust, high-level reasoning strategy (like 'think step-by-step'). "
                "This parameter's content **MUST BE PURE INSTRUCTION AND REASONING STRATEGY**."
            ),
            # This helps the Critic to focus its feedback correctly.
            instruction_to_backward_engine=(
                "When assessing this parameter's fault, focus ONLY on its clarity, persona, and strategic direction. "
            ),
            requires_opt=True,
            param_type=adal.ParameterType.PROMPT,
            name="instruction"
        )

        # Peer 2: Few-Shot Demonstrations (The most important for bigger gains)
        self.demos = adal.Parameter(
            data=load_prompt_file("demos.txt", ""),
            # General description of what this parameter is.
            role_desc="Provides a list of Question-Reasoning-Answer examples for in-context learning.",
            # The actionable command for the optimizer.
            instruction_to_optimizer=(
                "Your goal is to improve the list of few-shot examples. "
                "Your response MUST be a complete, self-contained list of examples. "
                "You must REPRODUCE any existing examples you wish to keep, and then ADD your completely NEW AND DIVERSE, "
                "well-crafted, Chain-of-Thought style examples. "
                "The new examples should target reasoning failures from the feedback and introduce **DIVERSE AND NOVEL REASONING PATTERNS**. "
                "Ensure the final list starts with '--- Example 1 ---', followed by '--- Example 2 ---', and so on. "
                "This parameter's content MUST BE ONLY EXAMPLES."
            ),
            # This helps the Critic to focus its feedback correctly.
            instruction_to_backward_engine=(
                "When assessing this parameter's fault, focus ONLY on the quality, relevance, and sufficiency of the provided examples. "
                "Consider if a different or entirely new and diverse example was needed to prevent the error."
            ),
            requires_opt=True,
            param_type=adal.ParameterType.PROMPT,
            name="demos"
        )

        # Peer 3: Output Formatting
        self.output_format = adal.Parameter(
            data=load_prompt_file("output_format.txt", "Finish your answer with exactly: 'Answer: X' where X is the number."),
            role_desc="A fixed, non-trainable parameter that defines the mandatory output syntax for the final answer.",
            # We add this for completeness and clarity, even though it's non-trainable.
            # It helps the Teacher understand the system's constraints.
            instruction_to_optimizer=(
                "This parameter is a fixed, non-trainable rule. You cannot change it. "
                "You must ensure that any changes you propose to other parameters "
                "still result in an output that respects this final formatting constraint."
            ),
            # This helps the Critic to focus its feedback correctly.
            instruction_to_backward_engine=(
                "When assessing this parameter's fault, check ONLY if the Student's final output failed to adhere to this strict formatting rule. "
            ),
            requires_opt=False,
            param_type=adal.ParameterType.PROMPT,
            name="output_format"
        )

        # Initialize Generator with Compound Template
        # We explicitly structure the prompt using the three peers.
        self.generator = adal.Generator(
            model_client=student_client,
            model_kwargs=model_kwargs,
            template="""<START_OF_SYSTEM_PROMPT>
<INSTRUCTION>
{{instruction}}
</INSTRUCTION>
<FORMAT>
{{output_format}}
</FORMAT>
<EXAMPLES>
{{demos}}
</EXAMPLES>
<END_OF_SYSTEM_PROMPT>

<START_OF_USER>
<USER_INPUT>
{{input_str}}
</USER_INPUT>
<END_OF_USER>""",
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