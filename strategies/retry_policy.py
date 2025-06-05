# voxsigil_supervisor/retry_policy.py
"""
Defines the retry policy for the VoxSigil Supervisor.
This includes logic for determining when and how to retry failed attempts,
when to switch strategies, and when to give up.
"""
from typing import Dict, List, Any, Optional, Tuple
import logging
import time
import random

# Setup logger
logger_retry_policy = logging.getLogger("VoxSigilSupervisor.retry_policy")

class BasicRetryPolicy:
    """
    A simplified retry policy that provides basic retry functionality.
    This policy will retry a fixed number of times with optional backoff.
    """
    
    def __init__(self, max_retries: int = 1, backoff_factor: float = 1.5):
        """
        Initialize the BasicRetryPolicy.
        
        Args:
            max_retries: Maximum number of retry attempts (default: 1).
            backoff_factor: Factor to increase wait time between retries (default: 1.5).
        """
        self.logger = logger_retry_policy
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.logger.info(f"Initialized BasicRetryPolicy with max_retries={max_retries}")
    
    def should_retry(self, 
                    attempt_number: int, 
                    quality_score: Optional[float] = None,
                    error: Optional[Exception] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Determines whether to retry based on the attempt number and quality score.
        
        Args:
            attempt_number: The current attempt number (0-based index).
            quality_score: Optional quality score from the last attempt.
            error: Optional exception that occurred during the attempt.
            
        Returns:
            Tuple of (should_retry, retry_info)
        """
        # Check if we've exceeded the maximum number of retries
        if attempt_number >= self.max_retries:
            self.logger.info(f"Not retrying: max retries ({self.max_retries}) reached")
            return False, {"reason": "max_retries_reached"}
        
        # Calculate backoff time if needed
        backoff_time = 0
        if attempt_number > 0:
            backoff_time = (self.backoff_factor ** attempt_number) * 0.1
        
        # If there was an error, always retry
        if error is not None:
            self.logger.info(f"Retrying due to error: {error}")
            return True, {
                "reason": "error",
                "backoff_time": backoff_time,
                "error": str(error)
            }
        
        # If quality score is below threshold, retry
        if quality_score is not None and quality_score < 0.7:  # Using 0.7 as default threshold
            self.logger.info(f"Retrying due to low quality score: {quality_score}")
            return True, {
                "reason": "low_quality_score",
                "quality_score": quality_score,
                "backoff_time": backoff_time
            }
        
        # Default: don't retry if we've reached this point
        return False, {"reason": "success_or_undefined_condition"}
    
    def get_retry_strategy(self, attempt_number: int) -> Dict[str, Any]:
        """
        Get the strategy to use for a specific retry attempt.
        
        Args:
            attempt_number: The current attempt number (0-based index).
            
        Returns:
            A dictionary with retry strategy information.
        """
        # Simple strategies that can be tried
        strategies = [
            {"name": "add_instructions", "description": "Adding more detailed instructions"},
            {"name": "simplify_query", "description": "Simplifying the query"},
            {"name": "different_scaffold", "description": "Using a different reasoning structure"}
        ]
        
        # Select a strategy based on the attempt number
        strategy_index = min(attempt_number, len(strategies) - 1)
        selected_strategy = strategies[strategy_index]
        
        self.logger.info(f"Selected retry strategy: {selected_strategy['name']}")
        return selected_strategy
    
    def apply_retry_strategy(self, 
                           strategy: Dict[str, Any],
                           query: str,
                           context: Optional[str] = None,
                           scaffold: Optional[str] = None) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Apply the selected retry strategy to modify the inputs.
        
        Args:
            strategy: The strategy to apply.
            query: The original query.
            context: Optional context information.
            scaffold: Optional reasoning scaffold.
            
        Returns:
            Tuple of (modified_query, modified_context, modified_scaffold)
        """
        strategy_name = strategy.get("name", "")
        
        if strategy_name == "add_instructions":
            # Add more explicit instructions to the query
            enhanced_query = (
                f"{query}\n\n"
                f"Please provide a detailed, step-by-step response. "
                f"Break down complex concepts and ensure thoroughness."
            )
            return enhanced_query, context, scaffold
            
        elif strategy_name == "simplify_query":
            # Attempt to simplify the query
            # This is a very basic simplification that just adds a request for simplicity
            simplified_query = (
                f"I need a clear and simple answer to this question: {query}\n\n"
                f"Please focus on the core concepts and explain them clearly."
            )
            return simplified_query, context, scaffold
            
        elif strategy_name == "different_scaffold":
            # Use a different reasoning scaffold
            # Here we're just using a simple alternative scaffold
            alternative_scaffold = (
                "1. Understand the key question\n"
                "2. Identify the main concepts\n"
                "3. Analyze step-by-step\n"
                "4. Provide clear conclusions"
            )
            return query, context, alternative_scaffold
            
        # Default: return unchanged inputs
        return query, context, scaffold

class RetryPolicy:
    """
    Handles the logic for retrying failed attempts with different strategies.
    """
    
    def __init__(self, max_attempts: int = 3, min_score_threshold: float = 0.7):
        """
        Initialize the RetryPolicy.
        
        Args:
            max_attempts: Maximum number of attempts before giving up.
            min_score_threshold: Minimum score considered successful.
        """
        self.logger = logger_retry_policy
        self.max_attempts = max_attempts
        self.min_score_threshold = min_score_threshold
        
        # Different strategies to try on retries
        self.strategies = [
            {"name": "basic", "description": "Basic approach"},
            {"name": "different_model", "description": "Using a different model"},
            {"name": "more_context", "description": "Using more context"},
            {"name": "different_scaffold", "description": "Using a different reasoning scaffold"},
            {"name": "increase_model_tier", "description": "Using a stronger model"},
            {"name": "detailed_instructions", "description": "Using more detailed instructions"}
        ]
    
    def should_retry(self, 
                    attempt_history: List[Dict[str, Any]], 
                    last_score: float) -> Tuple[bool, Dict[str, Any]]:
        """
        Determines whether to retry based on the attempt history and last score.
        
        Args:
            attempt_history: List of previous attempt dictionaries.
            last_score: The score from the last attempt.
            
        Returns:
            Tuple of (should_retry, retry_strategy).
        """
        current_attempt = len(attempt_history)
        
        # If we've reached max attempts, don't retry
        if current_attempt >= self.max_attempts:
            self.logger.info(f"Max attempts ({self.max_attempts}) reached. Not retrying.")
            return False, {}
        
        # If last attempt was successful (above threshold), don't retry
        if last_score >= self.min_score_threshold:
            self.logger.info(f"Last attempt successful with score {last_score}. Not retrying.")
            return False, {}
        
        # Decide on a strategy for the retry
        strategy = self._select_retry_strategy(attempt_history, last_score)
        
        self.logger.info(f"Retry {current_attempt + 1}/{self.max_attempts} with strategy: {strategy['name']}")
        
        return True, strategy
    
    def _select_retry_strategy(self, 
                              attempt_history: List[Dict[str, Any]], 
                              last_score: float) -> Dict[str, Any]:
        """
        Selects a strategy for the next retry attempt.
        
        Args:
            attempt_history: List of previous attempt dictionaries.
            last_score: The score from the last attempt.
            
        Returns:
            Dictionary describing the selected strategy.
        """
        # If this is the first retry, use a basic strategy
        if len(attempt_history) == 1:
            return self.strategies[0]
        
        # Extract previous strategies
        previous_strategies = [a.get("strategy", {}).get("name") for a in attempt_history if "strategy" in a]
        
        # Available strategies (those not already tried)
        available_strategies = [s for s in self.strategies if s["name"] not in previous_strategies]
        
        # If we've tried all strategies, start over but avoid the most recent one
        if not available_strategies:
            available_strategies = [s for s in self.strategies if s["name"] != previous_strategies[-1]]
        
        # Special logic for very low scores
        if last_score < 0.3 and "increase_model_tier" in [s["name"] for s in available_strategies]:
            for strategy in available_strategies:
                if strategy["name"] == "increase_model_tier":
                    return strategy
        
        # Otherwise, choose a random strategy from available ones
        return random.choice(available_strategies)
    
    def get_retry_parameters(self, 
                            strategy: Dict[str, Any], 
                            current_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gets parameters for the next attempt based on the selected strategy.
        
        Args:
            strategy: The selected retry strategy.
            current_params: Current parameters used for the last attempt.
            
        Returns:
            Updated parameters for the next attempt.
        """
        updated_params = current_params.copy()
        
        # Apply strategy-specific modifications
        strategy_name = strategy.get("name", "basic")
        
        if strategy_name == "different_model":
            # Modify model selection parameters
            current_services = updated_params.get("model_params", {}).get("service_priority", ["ollama"])
            if len(current_services) > 1:
                # Move the first service to the end
                updated_params.setdefault("model_params", {})["service_priority"] = current_services[1:] + [current_services[0]]
            else:
                # If only one service, we can't change it
                pass
        
        elif strategy_name == "more_context":
            # Increase context retrieval
            updated_params["max_results"] = updated_params.get("max_results", 5) + 3
            updated_params["expand_context"] = True
        
        elif strategy_name == "different_scaffold":
            # Force selection of a different scaffold
            # The scaffold router handles this when prev_scaffold_id is provided
            updated_params["force_new_scaffold"] = True
        
        elif strategy_name == "increase_model_tier":
            # Increase model strength tier
            updated_params.setdefault("model_params", {})["model_tier"] = min(5, updated_params.get("model_params", {}).get("model_tier", 3) + 1)
        
        elif strategy_name == "detailed_instructions":
            # Add more detailed instructions
            updated_params["detailed_instructions"] = True
        
        # Add the strategy to the parameters
        updated_params["strategy"] = strategy
        
        return updated_params
    
    def generate_retry_prompt(self, 
                             query: str, 
                             last_response: str, 
                             evaluation: Dict[str, Any], 
                             strategy: Dict[str, Any]) -> str:
        """
        Generates a modified prompt for the retry attempt.
        
        Args:
            query: The original query.
            last_response: The response from the last attempt.
            evaluation: Evaluation results from the last attempt.
            strategy: The selected retry strategy.
            
        Returns:
            Modified query for the retry attempt.
        """
        strategy_name = strategy.get("name", "basic")
        summary = evaluation.get("summary", "The previous response was inadequate.")
        
        # Base retry prompt
        retry_prompt = f"The following query needs a better response:\n\n{query}\n\n"
        
        # Add strategy-specific elements
        if strategy_name == "detailed_instructions":
            retry_prompt += (
                f"Previous response evaluation: {summary}\n\n"
                f"Please provide a more detailed and comprehensive response. "
                f"Make sure to address all aspects of the query and provide specific information. "
                f"Structure your response with clear sections, and include any relevant details "
                f"that help answer the query completely."
            )
        else:
            retry_prompt += (
                f"Previous response evaluation: {summary}\n\n"
                f"Please provide an improved response that addresses these issues."
            )
        
        return retry_prompt
