"""
GSM8K Task Definition.

This module defines:
1. The output parser (`parse_gsm8k_answer`) which extracts numbers from text.
2. The Student Component (`GSM8KStudent`) which holds the trainable system prompt
   and the generator logic.
"""

import re
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
    The Student Component responsible for solving math problems.
    
    This component wraps the LLM generator and holds the 'system_prompt' 
    as a trainable Parameter. This allows the AdalFlow optimizer to update 
    the instructions based on feedback.
    """
    def __init__(self, student_client: adal.ModelClient, model_kwargs: Dict):
        """
        Initialize the Student.

        Args:
            student_client (ModelClient): The backend client (e.g., LocalLLMClient).
            model_kwargs (Dict): Generation parameters (temp, max_tokens, etc.).
        """
        super().__init__()

        # The System Prompt is the trainable parameter.
        # We set requires_opt=True to tell AdalFlow this text should be optimized.
        self.system_prompt = adal.Parameter(
            data="You are a helpful math assistant. Solve the problem step by step. Finish your answer with exactly: 'Answer: X' where X is the number.",
            role_desc="Math Instructions",
            requires_opt=True,
            param_type=adal.ParameterType.PROMPT,
            name="system_prompt"
        )

        # Initialize Generator
        # We pass the wrapped 'parse_gsm8k_answer' component here.
        # The template explicitly combines the system prompt and the user input
        # into a single string for the client.
        self.generator = adal.Generator(
            model_client=student_client,
            model_kwargs=model_kwargs,
            template="{{system_prompt}}\n\n{{input_str}}",
            output_processors=parse_gsm8k_answer,
            use_cache=False
        )

    def call(self, question: str, id: str = None) -> GeneratorOutput:
        """
        Forward pass: Generates an answer for a given math question.
        """
        return self.generator(
            prompt_kwargs={
                # Pass the current data of the trainable parameter
                "system_prompt": self.system_prompt.data, 
                "input_str": question
            },
            id=id
        )