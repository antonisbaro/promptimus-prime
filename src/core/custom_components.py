"""
Custom AdalFlow Components.

This module provides a robust XML parser designed to override the default,
brittle parser in the AdalFlow library. This is implemented to handle
malformed or incomplete XML responses from the Teacher LLM during the
optimization process.
"""

import re
import logging
from adalflow.optim.text_grad.tgd_optimizer import CustomizedXMLParser, TGDData

# Configure logger for this module
log = logging.getLogger(__name__)


class RobustXMLParser(CustomizedXMLParser):
    """
    An overridden version of the default XML parser that uses regex.

    This parser is designed to be resilient against malformed XML output from
    the Teacher LLM. Instead of using a strict XML engine (like xml.etree),
    it uses regular expressions to find and extract content within specific
    tags, gracefully ignoring any surrounding malformed structures or text.
    """
    def call(self, input: str) -> TGDData:
        """
        Parses the XML-like response from the optimizer LLM robustly.

        Args:
            input (str): The raw string output from the Teacher LLM.

        Returns:
            TGDData: A structured data object containing the parsed fields.
        """
        # Helper function to safely extract content from a specific tag.
        def extract_tag_content(tag_name: str, text: str) -> str:
            """
            Finds a tag and returns its content using a non-greedy regex search.
            """
            pattern = f"<{tag_name}>(.*?)</{tag_name}>"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
            return ""

        try:
            clean_input = input.strip()
            
            # Use the robust helper for each expected field
            reasoning = extract_tag_content("reasoning", clean_input)
            method = extract_tag_content("method", clean_input)
            proposed_variable = extract_tag_content("proposed_variable", clean_input)
            
            # Log a warning if the most critical field is missing.
            if not proposed_variable and "<proposed_variable>" in clean_input:
                 log.warning(
                    f"Found <proposed_variable> tags but content is empty. "
                    f"Full output: {clean_input}"
                )
            elif not proposed_variable:
                log.warning(
                    f"Could not find a valid <proposed_variable> tag in the "
                    f"optimizer's response. The Teacher LLM may have failed to follow formatting instructions."
                )

            return TGDData(
                reasoning=reasoning,
                method=method,
                proposed_variable=proposed_variable
            )
            
        except Exception as e:
            # Catch any other unexpected errors during the process
            log.error(f"A critical error occurred in RobustXMLParser.call: {e}")
            return TGDData(
                reasoning=f"Critical parsing error: {e}", 
                method="Error",
                proposed_variable=input
            )