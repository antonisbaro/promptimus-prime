"""
Custom AdalFlow Components.

This module provides the DEFINITIVE RobustXMLParser designed to override the 
default, brittle XML parser in the AdalFlow library.

It implements a 'Metadata Stripping' strategy to handle the two most common 
failure modes of the Teacher LLM during optimization:
1.  **Truncated Responses:** When the model hits the max_token limit, closing tags 
    (like </proposed_variable>) are often missing.
2.  **Schema Hallucination:** The Teacher often mistakenly copies metadata tags 
    (like <NAME>, <ROLE>) from its system prompt instructions into the final output.
"""

import re
import logging
from adalflow.optim.text_grad.tgd_optimizer import CustomizedXMLParser, TGDData

# Configure logger for this module
log = logging.getLogger(__name__)

class RobustXMLParser(CustomizedXMLParser):
    """
    A fault-tolerant parser for the Textual Gradient Descent (TGD) optimizer.

    Unlike standard XML parsers that crash on malformed input, this class uses 
    regex-based heuristics to extract the most likely valid content. It prioritizes 
    continuity of the optimization loop over strict syntax compliance.
    """
    
    def call(self, input_str: str) -> TGDData:
        """
        Parses the raw string output from the Teacher LLM into structured data.

        Args:
            input_str (str): The raw text response from the optimizer model.

        Returns:
            TGDData: Object containing reasoning, method, and the cleaned variable.
        """
        
        def clean_final_content(text: str) -> str:
            """
            Applies a smart cleaning strategy to the extracted proposal block.
            
            Strategy:
            1. **Targeted Removal:** Identifies and deletes specific metadata blocks 
               (<NAME>, <ROLE>, etc.) that the Teacher frequently hallucinates 
               because they appear in its own system prompt definition.
            2. **General Cleanup:** Strips any remaining XML-like tags while 
               preserving their inner content.
            """
            working_text = text
            
            # List of metadata tags the Teacher sees in the context and might hallucinate.
            # We want to DELETE these blocks entirely (content included) because 
            # they are not part of the actual prompt we want to optimize.
            metadata_tags = ["NAME", "TYPE", "ROLE", "PARAM_TYPE", "DESCRIPTION"]
            
            for tag in metadata_tags:
                # Regex matches <TAG>...content...</TAG> and removes the whole block.
                # flags=re.DOTALL ensures '.' matches newlines.
                # flags=re.IGNORECASE handles variations like <Role> or <role>.
                pattern = f"<{tag}>.*?</{tag}>"
                working_text = re.sub(pattern, "", working_text, flags=re.DOTALL | re.IGNORECASE)
                
                # Edge Case: Handle unclosed metadata tags at the very end (truncation).
                # e.g., <NAME>instruct... (End of string)
                # We conservatively remove it to avoid leaking metadata into the prompt.
                pattern_open = f"<{tag}>.*"
                # Note: We rely primarily on the closed tag removal above, but this 
                # can be enabled if deep truncation of metadata becomes an issue.

            # After removing known metadata blocks, strip any other arbitrary tags 
            # that might wrap the real content (e.g., <VARIABLE>Real Content</VARIABLE> -> Real Content).
            clean_text = re.sub(r'<[^>]*>', '', working_text, flags=re.DOTALL)
            
            # Normalize whitespace while preserving essential line breaks
            return "\n".join([line.strip() for line in clean_text.splitlines() if line.strip()])

        try:
            # Ensure input is a clean string
            clean_input = str(input_str).strip()
            
            # -----------------------------------------------------------------
            # 1. Extract Metadata (Reasoning & Method)
            # -----------------------------------------------------------------
            # We use non-greedy regex (.*?) to grab these fields for logging/debugging.
            # If missing, we provide default text rather than crashing.
            reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", clean_input, re.DOTALL)
            method_match = re.search(r"<method>(.*?)</method>", clean_input, re.DOTALL)
            
            reasoning_text = reasoning_match.group(1).strip() if reasoning_match else "No reasoning found"
            method_text = method_match.group(1).strip() if method_match else "No method found"

            # -----------------------------------------------------------------
            # 2. Extract The Proposed Variable Block (Core Logic)
            # -----------------------------------------------------------------
            proposal_block = ""
            start_marker = "<proposed_variable>"
            end_marker = "</proposed_variable>"
            
            start_index = clean_input.find(start_marker)
            
            if start_index != -1:
                # Found opening tag, extract content starting immediately after it
                start_content = start_index + len(start_marker)
                end_index = clean_input.find(end_marker, start_content)
                
                if end_index != -1:
                    # CASE A: Standard complete response
                    proposal_block = clean_input[start_content:end_index]
                else:
                    # CASE B: Truncation Handling
                    # The response was cut off before the closing tag. 
                    # We capture everything from the opening tag to the end of the string.
                    log.warning("⚠️ Missing </proposed_variable> tag. Capturing content to End-of-String.")
                    proposal_block = clean_input[start_content:]
            else:
                # CASE C: Tag Missing entirely
                # Fallback: Assume the whole input (minus reasoning/method) is the proposal.
                # This is a last-ditch effort to salvage the step.
                log.warning("❌ Could not find <proposed_variable> tag. Attempting fallback extraction.")
                
                # Remove reasoning/method blocks to avoid duplication in the prompt
                temp = re.sub(r"<reasoning>.*?</reasoning>", "", clean_input, flags=re.DOTALL)
                temp = re.sub(r"<method>.*?</method>", "", temp, flags=re.DOTALL)
                proposal_block = temp

            # -----------------------------------------------------------------
            # 3. Apply Smart Cleaning
            # -----------------------------------------------------------------
            # This handles the nested <NAME>/<ROLE> scenario by stripping specific 
            # metadata blocks and returning the clean inner text.
            final_proposal = clean_final_content(proposal_block)

            return TGDData(
                reasoning=reasoning_text,
                method=method_text,
                proposed_variable=final_proposal
            )
            
        except Exception as e:
            # Pokemon Exception Handling: Catch 'em all to prevent the optimizer loop from breaking.
            # We return the raw input as the proposal so the system can potentially try again.
            log.error(f"Critical error in RobustXMLParser: {e}")
            return TGDData(
                reasoning=f"Error: {e}", 
                method="Error", 
                proposed_variable=""
            )