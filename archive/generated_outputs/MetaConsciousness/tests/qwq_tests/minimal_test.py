"""
Minimal QwQ-32B Test Script

This is a simplified test script that only focuses on testing how QwQ-32B responses are processed.
"""

import logging
import requests
import json
import re
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("qwq_minimal_test.log", mode="w"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("qwq_minimal_test")
logger.info("Starting QwQ-32B minimal test")

def safe_dict_get(data, keys, default=None):
    current = data
    try:
        for key in keys:
            if not isinstance(current, dict):
                return default
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default

def test_qwq_response():
    """Test processing a QwQ-32B response with the problematic structure"""
    
    # This simulates the structure of a QwQ-32B response as seen in your logs
    mock_qwq_response = {
        "choices": [
            {
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "\n\nReasoning:\n1. The user greeted with \"hi\", which is a common informal salutation.\n2. The system is in OBSERVING mode (M=OBSERVING), indicating it should respond neutrally and await further input.\n3. Appropriate responses to \"hi\" typically involve a friendly greeting and an offer to assist.\n4. The numerical parameters (A=0.47, R=0.27) may relate to internal system states but do not affect standard greeting protocols.\n\nAnswer:\nHello! How can I assist you today?",
                    "reasoning_content": "\n\nOkay, let's see. The user said \"hi\". So I need to respond appropriately.\n\nFirst, check the state variables given: A=0.47, R=0.27, M=OBSERVING. Not sure how those affect the response here. Maybe they're part of some model parameters or modes? Since the mode is OBSERVING, perhaps I should stay neutral and just reply with a friendly greeting.\n\nThe user probably expects a simple \"hello\" back. No need for overcomplicating. Just make sure to follow the format: reasoning then answer.\n\nWait, the instructions say to provide step-by-step reasoning leading to the final answer. Let me structure that properly.\n\nReasoning steps:\n1. User message is \"hi\".\n2. The current state has M=OBSERVING, so respond appropriately without taking action unless needed.\n3. Appropriate response is a greeting like \"Hello! How can I assist you today?\" \n4. Ensure it's friendly and open-ended to encourage further interaction.\n\nAnswer should just be the final message as per the format.\n"
                }
            }
        ]
    }
    
    # Process the mock response
    logger.info("Processing mock QwQ-32B response")
    
    # Extract content from response
    content = safe_dict_get(mock_qwq_response, ['choices', 0, 'message', 'content'])
    reasoning_content = safe_dict_get(mock_qwq_response, ['choices', 0, 'message', 'reasoning_content'])
    
    logger.info(f"Content present: {content is not None}")
    logger.info(f"Reasoning_content present: {reasoning_content is not None}")
    
    if content:
        logger.info(f"Content (first 100 chars): {content[:100]}")
    if reasoning_content:
        logger.info(f"Reasoning_content (first 100 chars): {reasoning_content[:100]}")
    
    # Extract reasoning and answer from content
    answer = content
    reasoning = reasoning_content or ""
    
    # Check for Reasoning/Answer markers in content
    reasoning_marker = "Reasoning:"
    answer_marker = "Answer:"
    
    logger.info(f"Checking for reasoning/answer markers in content")
    logger.info(f"Reasoning marker in content: {'reasoning:' in content.lower()}")
    logger.info(f"Answer marker in content: {'answer:' in content.lower()}")
    
    if content and reasoning_marker.lower() in content.lower() and answer_marker.lower() in content.lower():
        logger.info("Found both markers in content")
        
        # Find positions of markers
        reasoning_pos = content.lower().find(reasoning_marker.lower())
        answer_pos = content.lower().find(answer_marker.lower())
        
        logger.info(f"Reasoning position: {reasoning_pos}")
        logger.info(f"Answer position: {answer_pos}")
        
        if reasoning_pos != -1 and answer_pos != -1 and answer_pos > reasoning_pos:
            # Extract content between markers
            reasoning_start = reasoning_pos + len(reasoning_marker)
            answer_start = answer_pos + len(answer_marker)
            
            reasoning_from_content = content[reasoning_start:answer_pos].strip()
            answer_from_content = content[answer_start:].strip()
            
            logger.info(f"Extracted reasoning: {reasoning_from_content}")
            logger.info(f"Extracted answer: {answer_from_content}")
            
            # Use extracted content
            answer = answer_from_content
            
            # If we don't have explicit_reasoning_content, use the reasoning from content
            if not reasoning:
                reasoning = reasoning_from_content
            # If we have both, use the longer one
            elif len(reasoning_from_content) > len(reasoning):
                reasoning = reasoning_from_content
    else:
        logger.warning("Could not find expected 'Reasoning:' and 'Answer:' markers in the content")
        
    logger.info(f"Final answer: {answer}")
    logger.info(f"Final reasoning (first 100 chars): {reasoning[:100]}")
    
    return answer, reasoning

if __name__ == "__main__":
    logger.info("Running QwQ-32B minimal test")
    answer, reasoning = test_qwq_response()
    logger.info("Test completed")