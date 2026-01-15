"""
Local LLM Client Infrastructure.

This module defines the `LocalLLMClient`, a custom adapter that bridges AdalFlow
with Hugging Face Transformers. It is specifically optimized for running
experiments on consumer hardware (e.g., Google Colab T4 GPUs) by leveraging
4-bit quantization and efficient model loading.

Key responsibilities:
1. Model Loading: Handles BitsAndBytes config for low-memory usage.
2. Chat Templating: Applies the correct system/user format for Instruct models.
3. Protocol Adaptation: Converts AdalFlow inputs/outputs to Transformers format.
"""

import logging
import torch
import re
from typing import Any, Dict, Optional, List
from src.tasks.gsm8k.config import PROMPTS_VERBOSITY

# Third-party libraries
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# AdalFlow core imports
from adalflow.core.model_client import ModelClient
from adalflow.core.types import GeneratorOutput, ModelType

# Configure logger for this module
log = logging.getLogger(__name__)

class LocalLLMClient(ModelClient):
    """
    A truly generic, custom AdalFlow client wrapper for Hugging Face Transformers models.
    
    This client is designed to be robust and adaptable, handling various model-specific
    requirements automatically.

    Key Features:
    - **Conditional Quantization:** Supports loading models in 4-bit (for large models)
      or native bfloat16 (for smaller models).
    - **Flexible Loading Args:** Allows passing custom `from_pretrained` arguments
      to handle model-specific workarounds (e.g., for Phi-3).
    - **Smart Role Handling:** Automatically detects if a model supports a `system`
      role and adjusts the prompt format accordingly.
    """

    def __init__(self, model_name: str, quantize: bool = True, model_load_kwargs: Dict = {}):
        """
        Initialize the client with a specific model and loading configuration.

        Args:
            model_name (str): The Hugging Face model ID.
            quantize (bool): If True, loads the model with 4-bit quantization.
                             If False, loads in its native bfloat16 precision.
                             Defaults to True.
            model_load_kwargs (Dict): Extra arguments for model loading (e.g., attn_implementation).
        """
        super().__init__()
        self.model_name = model_name
        self.quantize = quantize
        self.model_load_kwargs = model_load_kwargs
        self.tokenizer = None
        self.model = None

        # This flag will be set during initialization based on the model's capabilities.
        self.supports_system_role = False

        # Initialize the model immediately upon instantiation        
        self._initialize_model()

    def _initialize_model(self):
        """
        Loads the model and tokenizer from Hugging Face, applying quantization
        conditionally based on the `self.quantize` flag.
        """
        log.info(f"Loading model: {self.model_name}...")
       
        # This dictionary will hold all arguments for the .from_pretrained call.
        load_args = {
            "device_map": "auto",
            "trust_remote_code": True,
            **self.model_load_kwargs
        }

        if self.quantize:
            # --- Option 1: 4-bit Quantization (Low VRAM Mode) ---
            # Ideal for large models (like the Teacher) on limited hardware.
            # This significantly reduces memory footprint at a small cost to precision.
            log.info("4️⃣ Mode: 4-bit Quantization (Low VRAM)")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            load_args["quantization_config"] = bnb_config
        else:
            # --- Option 2: Native bfloat16 Precision (High Accuracy Mode) ---
            # Ideal for smaller models (like the Student) that fit easily in VRAM.
            # This uses the model's standard precision for maximum performance.
            log.info("🛜 Mode: Native bfloat16 Precision (High Accuracy)")
            load_args["torch_dtype"] = torch.bfloat16

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
           
            # Load the model using the dynamically constructed arguments
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_args
            )

            # Capability Diagnosis
            # We perform a safe "dry run" of the apply_chat_template function.
            # This is the most reliable way to determine if a model truly supports the system role.
            try:
                # We create a dummy message list with a system role.
                test_messages = [
                    {"role": "system", "content": "Test system message."},
                    {"role": "user", "content": "Test user message."}
                ]
                
                # We try to apply the chat template.
                self.tokenizer.apply_chat_template(test_messages, tokenize=False)
                
                # If the line above executes without an error, the system role is supported.
                self.supports_system_role = True
                log.info(f"Capability: 🤓 System role is SUPPORTED.")

            except Exception as e:
                # If an error occurs we know the system role is not supported.
                self.supports_system_role = False
                log.info(f"Capability: 🤪 System role is NOT SUPPORTED by this model. Prompts will be merged. (Reason: {e})")

            log.info(f"✅ Successfully loaded {self.model_name}")
            
        except Exception as e:
            log.error(f"❌ Failed to load model {self.model_name}: {e}")
            raise e

    def convert_inputs_to_api_kwargs(
        self,
        input: Any,
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        """
        Adapts AdalFlow's generic input structure to the specific arguments 
        expected by this client's `call` method.

        Args:
            input (Any): The input data (usually the rendered prompt string).
            model_kwargs (Dict): Additional generation parameters.

        Returns:
            Dict: A dictionary ready to be unpacked into `call()`.
        """
        final_args = {
            "input_str": input,
            "model_kwargs": model_kwargs.copy()
        }
        return final_args

    def parse_chat_completion(self, completion: Any) -> GeneratorOutput:
        """
        Parses the raw response from the `call` method into a structured GeneratorOutput.
        
        This method includes logic to clean specific XML tags (like <proposed_variable>)
        that might be generated by the Text-Grad optimizer, ensuring the output 
        contains only the clean prompt content.

        Args:
            completion (Any): The raw string returned by `call()`.

        Returns:
            GeneratorOutput: The wrapped output object expected by AdalFlow components.
        """
        try:
            response_text = str(completion)

            # Clean XML Tags if present.
            # The TGD Optimizer sometimes wraps the proposed prompt in XML tags.
            # We extract the content inside <proposed_variable>...</proposed_variable>.
            if "<proposed_variable>" in response_text:
                match = re.search(r"<proposed_variable>(.*?)</proposed_variable>", response_text, re.DOTALL)
                if match:
                    response_text = match.group(1).strip()

            return GeneratorOutput(data=response_text, raw_response=str(completion))
        except Exception as e:
            # Return an error object instead of crashing if parsing fails
            return GeneratorOutput(data=None, error=str(e))

    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED) -> str:
        """
        Executes the forward pass (inference) using the Hugging Face model.

        This method handles:
        1. Constructing the message history (System + User).
        2. Applying the model's specific chat template.
        3. Generating the response tokens.
        4. Decoding the tokens back to string.

        Args:
            api_kwargs (Dict): Must contain 'input_str' or 'messages', plus optional 'model_kwargs'.

        Returns:
            str: The raw generated text string.
        """
        # Extract All Necessary Data
        full_prompt_str = api_kwargs.get("input_str", "")
        gen_kwargs = api_kwargs.get("model_kwargs", {})

        # Pop the custom role tag for logging purposes. This removes it from `gen_kwargs`
        # so it's not passed to the underlying `model.generate` call, which would cause an error.
        caller_role = gen_kwargs.pop("caller_role", "Unknown")

        system_prompt = ""
        user_prompt = full_prompt_str # Default: assume the whole prompt is a user message

        # Intelligent Prompt Splitting Logic
        # This logic works for both the Student's and the Teacher's templates.
        
        # Define the tags for splitting.
        sys_start_tag = "<START_OF_SYSTEM_PROMPT>"
        sys_end_tag = "<END_OF_SYSTEM_PROMPT>"
        
        sys_start_idx = full_prompt_str.find(sys_start_tag)
        sys_end_idx = full_prompt_str.find(sys_end_tag)

        if sys_start_idx != -1 and sys_end_idx != -1:
            # If the system block is found, split the string into two parts.
            system_prompt = full_prompt_str[sys_start_idx + len(sys_start_tag):sys_end_idx].strip()
            user_prompt = full_prompt_str[sys_end_idx + len(sys_end_tag):].strip()

            # As a final cleanup, remove the user tags themselves from the content.
            user_prompt = user_prompt.replace("<START_OF_USER>", "").replace("<END_OF_USER>", "").strip()

        # Construct the Final `messages` Array
        messages = []
        
        # This block now correctly handles all cases:
        # - Student: will have system_prompt and user_prompt.
        # - Teacher: will have system_prompt and user_prompt.
        # - Simple calls: will have only user_prompt.
        if "messages" in api_kwargs: # For special optimizer cases
            messages = api_kwargs["messages"]
        else:
            if system_prompt and self.supports_system_role:
                # Case A: Model supports system role. Create a two-part message list.
                messages.append({"role": "system", "content": system_prompt})
                if user_prompt:
                    messages.append({"role": "user", "content": user_prompt})
            else:
                # Case B: Model does not support system role (or no system prompt was given).
                # Combine everything into a single user prompt.
                combined_prompt = f"{system_prompt}\n\n{user_prompt}".strip() if system_prompt else user_prompt
                if combined_prompt:
                    messages.append({"role": "user", "content": combined_prompt})

        if not messages:
            log.warning(f"[{caller_role}] - No messages constructed. Skipping call.")
            return "" 

        try:
            # Apply Chat Template (Crucial for Instruct Models)
            text_input = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # If the global verbosity flag is enabled, print the detailed prompt.
            if PROMPTS_VERBOSITY:
                log.info("\n" + "━"*80 +
                         f"\n>>> FINAL RENDERED PROMPT | CALLER: [{caller_role}] | MODEL: [{self.model_name}] <<<\n" +
                         "---\n" +
                         text_input +
                         "\n---" +
                         "\n" + "━"*80 + "\n")
                
            # Tokenize and move to device
            model_inputs = self.tokenizer([text_input], return_tensors="pt").to(self.model.device)

            # Set generation parameters (defaults if not provided)
            max_new_tokens = gen_kwargs.get("max_new_tokens", 1024)
            temperature = gen_kwargs.get("temperature", 0.5)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=(temperature > 0),
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode Response (skipping the input prompt tokens to return only the answer)
            input_length = model_inputs.input_ids.shape[1]
            generated_ids = [output_ids[input_length:] for output_ids in generated_ids]
            response_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            return response_text # Return raw string!

        except Exception as e:
            log.error(f"Generation failed: {e}")
            return ""