"""
GSM8K Training Pipeline.

This module defines the `GSM8KTrainingPipeline`, which acts as the central
orchestrator for the training loop. It connects:
1. The Student (Generator) -> Produces answers.
2. The Evaluator -> Scores answers (0 or 1).
3. The Teacher (Loss/Backward Engine) -> Explains *why* an answer was wrong.

It also handles verbose logging to visualize the chain-of-thought and the 
optimization process in real-time.
"""

import random
import logging
from typing import Tuple, Callable, Dict, Any

import adalflow as adal
from adalflow.eval.answer_match_acc import AnswerMatchAcc
from adalflow.optim.text_grad.text_loss_with_eval_fn import EvalFnToTextLoss
from adalflow.core.generator import BackwardEngine, Generator
from adalflow.core.types import GeneratorOutput

from src.tasks.gsm8k.config import SEED
random.seed(SEED)

# Configure logger for this module
log = logging.getLogger(__name__)


class GSM8KTrainingPipeline(adal.AdalComponent):
    """
    The pipeline component that manages the forward and backward passes.
    
    Inherits from AdalComponent to integrate seamlessly with the AdalFlow Trainer.
    """
    
    def __init__(self, 
                 student_task: adal.Component, 
                 teacher_client: adal.ModelClient, 
                 teacher_model_kwargs: Dict):
        """
        Initialize the pipeline.

        Args:
            student_task: The GSM8KStudent instance (the model being trained).
            teacher_client: The client for the Teacher model (used for gradients).
            teacher_model_kwargs: Parameters for the Teacher (temp, max_tokens).
        """
        
        # 1. Define the Evaluation Metric
        # We use 'exact_match' to compare the parsed number against the ground truth.
        eval_fn = AnswerMatchAcc(type="exact_match").compute_single_item

        # 2. Configure the Teacher (Backward Engine)
        # We wrap the config in a dictionary required by AdalFlow's internal checks.
        teacher_config = {
            "model_client": teacher_client,
            "model_kwargs": teacher_model_kwargs
        }

        # Manually instantiate the BackwardEngine.
        log.info("🛠️  Instantiating BackwardEngine manually...")
        backward_engine = BackwardEngine(**teacher_config)

        # 3. Define the Loss Function (Textual Gradient)
        # This component uses the Teacher (BackwardEngine) to translate a
        # numerical score into textual feedback (gradient).
        loss_fn = EvalFnToTextLoss(
            eval_fn=eval_fn,
            eval_fn_desc="Check if the calculated number matches the ground truth number. Return 1 for match, 0 for mismatch.",
            backward_engine=backward_engine
        )

        # Double Check
        if loss_fn.backward_engine is None:
            log.warning("⚠️ Warning: Loss function missing engine. Forcing attachment.")
            loss_fn.set_backward_engine(backward_engine=backward_engine)
        else:
            log.info("🔗 BackwardEngine successfully attached to Loss Function.")

        # 4. Attach engine to Student's generator
        # For the gradient to flow from the output back to the input prompts,
        # the Generator component itself needs a reference to the BackwardEngine.
        # This is the crucial step that "connects the wires" for backpropagation.
        log.info(" Connecting BackwardEngine to the Student's Generator...")
        if hasattr(student_task, "generator") and isinstance(student_task.generator, Generator):
            student_task.generator.set_backward_engine(backward_engine)
            log.info("🔗 Engine attached successfully to 'student_task.generator'.")
        else:
            raise RuntimeError("CRITICAL: 'generator' attribute not found in student_task.")
     
        # 4. Initialize the Parent AdalComponent
        # We must explicitly pass the teacher config here so the Trainer knows
        # how to configure the optimizers internally.
        super().__init__(
            task=student_task,
            eval_fn=eval_fn,
            loss_fn=loss_fn,
            text_optimizer_model_config=teacher_config,
            backward_engine_model_config=teacher_config,
        )

        # Manually inject the engine after initialization to ensure it's available.
        self.backward_engine = backward_engine
        log.info("🔗 BackwardEngine injected successfully into pipeline.")

    def prepare_task(self, sample) -> Tuple[Callable, Dict]:
        """
        Prepares the input for the Forward Pass (Student).
        Extracts the question and ID from the dataset sample.
        """
        return self.task.call, {"question": sample.question, "id": sample.id}

    def prepare_eval(self, sample, y_pred: GeneratorOutput) -> Tuple[float, Dict]:
        """
        Evaluates the prediction and handles logging.
        
        This method is called during:
        1. Validation (on the full validation set).
        2. Training (to check if a proposal fixed an error).
        """
        # Extract data safely
        parsed_answer = y_pred.data
        raw_reasoning = y_pred.raw_response
        ground_truth = sample.answer
        
        # Determine correctness for logging purposes
        is_correct = (str(parsed_answer) == str(ground_truth))
        status = "✅ CORRECT" if is_correct else "❌ INCORRECT"

        # --- Logging Strategy ---
        # To avoid spamming the console during large training runs:
        # 1. Always log during Validation/Test phases (self.training == False).
        # 2. Randomly sample (~20%) logs during the Training phase.
        should_log = (not self.training) or (random.random() < 0.2)

        if should_log:
            print("\n" + "━"*60)
            q_text = sample.question.strip()
            print(f"📘 QUESTION: {q_text}")
            
            r_text = str(raw_reasoning).strip()
            print(f"💭 STUDENT (Snippet): {r_text}") 
            
            print(f"🎯 PARSED: '{parsed_answer}' | GT: '{ground_truth}'")
            print(f"RESULT: {status}")
            print("━"*60 + "\n")

        return self.eval_fn, {"y": parsed_answer, "y_gt": ground_truth}

    def prepare_loss(self, sample, pred: Any) -> Tuple[Callable, Dict]:
        """
        Backward Loss Preparation.
        """
        # 1. Ground Truth (Parameter)
        y_gt = adal.Parameter(
            name="y_gt", 
            data=str(sample.answer).strip(), 
            eval_input=str(sample.answer).strip(), 
            requires_opt=False
        )

        # 2. Prediction Handling
        # We MUST pass the original 'pred' Parameter to keep the gradient chain intact.
        if hasattr(pred, "data") and hasattr(pred.data, "data"):
             # pred.data is GeneratorOutput -> pred.data.data is the clean string
             clean_string = str(pred.data.data).strip()
             # We tell AdalFlow's evaluator: "For scoring, use this clean string"
             pred.eval_input = clean_string

        # We pass 'pred' (which is still a Parameter) to the loss function.
        return self.loss_fn, {"kwargs": {"y": pred, "y_gt": y_gt}, "id": sample.id}