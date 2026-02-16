"""
QwQ-32B Model Test Script

This script tests the integration of QwQ-32B model with MetaConsciousness.
It directly calls the LMStudioAdapter's _generate_with_reasoning method
to see how responses from QwQ-32B are being processed.
"""

import os
import sys
import json
import logging
import re

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("qwq_test_log.txt", mode="w"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("qwq_test")
logger.info("Starting QwQ-32B integration test")

# Import the adapter
from MetaConsciousness.models.lmstudio.adapter import LMStudioAdapter

def test_mock_response():
    """Test with a mock QwQ-32B response structure to fix marker detection"""
    
    # This simulates a QwQ-32B response with the extra spaces after markers
    mock_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "\n\nReasoning:  \nThe user greeted with \"hi\", which is a common informal salutation.\nAppropriate responses to \"hi\" typically involve a friendly greeting.\n\nAnswer:  \nHello! How can I assist you today?",
                    "reasoning_content": "Detailed reasoning content here..."
                }
            }
        ]
    }
    
    logger.info("Testing QwQ-32B mock response with extra spaces after markers")
    
    # Extract content from mock response
    content = mock_response["choices"][0]["message"]["content"]
    logger.info(f"Content: {repr(content)}")
    
    # Check for markers with standard approach (which fails)
    reasoning_marker = "Reasoning:"
    answer_marker = "Answer:"
    
    logger.info(f"Standard detection - Reasoning in content: {reasoning_marker in content}")
    logger.info(f"Standard detection - Answer in content: {answer_marker in content}")
    
    # Test detection with regex to handle extra spaces
    reasoning_pattern = re.compile(r"Reasoning:\s*\n", re.IGNORECASE)
    answer_pattern = re.compile(r"Answer:\s*\n", re.IGNORECASE)
    
    logger.info(f"Regex detection - Reasoning in content: {bool(reasoning_pattern.search(content))}")
    logger.info(f"Regex detection - Answer in content: {bool(answer_pattern.search(content))}")
    
    # Test extracting content with regex
    if reasoning_pattern.search(content) and answer_pattern.search(content):
        # Find the start positions
        reasoning_match = reasoning_pattern.search(content)
        answer_match = answer_pattern.search(content)
        
        if reasoning_match and answer_match:
            reasoning_start = reasoning_match.end()
            answer_start = answer_match.end()
            
            # Extract the content between markers
            reasoning_text = content[reasoning_start:answer_match.start()].strip()
            answer_text = content[answer_start:].strip()
            
            logger.info(f"Extracted reasoning: {reasoning_text}")
            logger.info(f"Extracted answer: {answer_text}")
            
            # Successful extraction
            return True, reasoning_text, answer_text
    
    logger.error("Failed to extract reasoning and answer from content")
    return False, "", ""

def run_test():
    """Run a test with QwQ-32B model."""
    logger.info("Initializing LMStudioAdapter for QwQ-32B testing")
    
    # First test with mock data
    success, reasoning, answer = test_mock_response()
    logger.info(f"Mock test result: {'SUCCESS' if success else 'FAILURE'}")
    if success:
        logger.info("Mock test successfully extracted reasoning and answer")
    
    # Create adapter instance
    adapter = LMStudioAdapter(
        api_base_url="http://localhost:1234/v1",
        model="qwq-32b",  # Use the actual name from your logs
        timeout=60
    )
    
    # Check connection and model availability
    if not adapter._is_connected:
        logger.error("Failed to connect to LM Studio server")
        return False
    
    logger.info(f"Connected to LM Studio, using model: {adapter.model}")
    
    # Apply our fix to the adapter's _generate_with_reasoning method
    original_method = adapter._generate_with_reasoning
    
    def fixed_generate_with_reasoning(self, query, system_prompt=None):
        """Fixed version of _generate_with_reasoning that handles QwQ-32B responses better"""
        logger.info(f"Running fixed _generate_with_reasoning for query: '{query}'")
        
        # Call the original method to get the raw response
        response = self._generate_text(query, system_prompt or self.get_current_system_prompt())
        
        # Log the response structure
        if isinstance(response, dict) and 'choices' in response:
            logger.info("Got dictionary response with 'choices'")
            content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
            reasoning_content = response.get('choices', [{}])[0].get('message', {}).get('reasoning_content', '')
            
            logger.info(f"Content: {repr(content[:100])}...")
            logger.info(f"Reasoning content exists: {bool(reasoning_content)}")
            
            # Use our improved pattern matching
            reasoning_pattern = re.compile(r"Reasoning:\s*\n", re.IGNORECASE)
            answer_pattern = re.compile(r"Answer:\s*\n", re.IGNORECASE)
            
            if content and reasoning_pattern.search(content) and answer_pattern.search(content):
                logger.info("Found both markers using regex pattern")
                
                # Find the start positions
                reasoning_match = reasoning_pattern.search(content)
                answer_match = answer_pattern.search(content)
                
                reasoning_start = reasoning_match.end()
                answer_start = answer_match.end()
                
                # Extract the content between markers
                reasoning_text = content[reasoning_start:answer_match.start()].strip()
                answer_text = content[answer_start:].strip()
                
                logger.info(f"Extracted reasoning: {reasoning_text[:50]}...")
                logger.info(f"Extracted answer: {answer_text}")
                
                return answer_text, reasoning_text or reasoning_content
        
        # If our custom extraction fails, fall back to the original method
        logger.info("Custom extraction failed, falling back to original method")
        return original_method(query, system_prompt)
    
    # Replace the method temporarily for testing
    adapter._generate_with_reasoning = fixed_generate_with_reasoning.__get__(adapter, LMStudioAdapter)
    
    # Test 1: Simple query
    logger.info("TEST 1: Running simple 'hi' query with QwQ-32B")
    answer, reasoning = adapter._generate_with_reasoning("hi", "You are a helpful assistant.")
    
    logger.info(f"TEST 1 RESULT - Answer (len={len(answer)}): {answer}")
    logger.info(f"TEST 1 RESULT - Reasoning (len={len(reasoning)}): {reasoning[:100]}...")
    
    # Test 2: Query with system state parameters
    logger.info("TEST 2: Running query with system state parameters")
    answer, reasoning = adapter._generate_with_reasoning(
        "hi (A=0.47, R=0.27, M=OBSERVING)", 
        "You are an assistant with metacognitive awareness. Respond based on the system state parameters."
    )
    
    logger.info(f"TEST 2 RESULT - Answer (len={len(answer)}): {answer}")
    logger.info(f"TEST 2 RESULT - Reasoning (len={len(reasoning)}): {reasoning[:100]}...")
    
    # Print final results
    logger.info("QwQ-32B Test Results Summary:")
    logger.info(f"Connection successful: {adapter._is_connected}")
    logger.info(f"Model used: {adapter.model}")
    logger.info(f"All tests completed")
    
    return answer != "I've processed your input based on the given parameters."  # Test if we got a real answer

if __name__ == "__main__":
    logger.info("Running QwQ-32B test")
    success = run_test()
    logger.info(f"Test completed with result: {'SUCCESS' if success else 'FAILURE'}")

