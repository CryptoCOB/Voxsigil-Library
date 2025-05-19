#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modified VoxSigil supervisor engine for better compatibility.
This module is a drop-in replacement for the original supervisor_engine.py.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import defaultdict

# Utils
from .utils.logging_utils import setup_supervisor_logging, SUPERVISOR_LOGGER_NAME

logger = logging.getLogger(SUPERVISOR_LOGGER_NAME)

class VoxSigilSupervisorCompat:
    """
    A compatible version of the VoxSigilSupervisor that works with our custom components.
    """
    
    def __init__(self,
                 rag_interface,
                 llm_interface,
                 scaffold_router,
                 evaluation_heuristics,
                 retry_policy,
                 memory_interface=None,
                 execution_strategy=None,
                 default_system_prompt=None,
                 max_iterations=3):
        """
        Initializes the VoxSigilSupervisor.

        Args:
            rag_interface: An instance of a RAG interface implementation.
            llm_interface: An instance of an LLM interface implementation.
            scaffold_router: Component for selecting reasoning scaffolds.
            evaluation_heuristics: Component for evaluating LLM responses.
            retry_policy: Component for determining retry logic.
            memory_interface: (Optional) Interface for long-term memory.
            execution_strategy: (Optional) Strategy for executing complex scaffolds.
            default_system_prompt: (Optional) Default system prompt to use.
            max_iterations: Maximum number of iteration attempts.
        """
        self.rag_interface = rag_interface
        self.llm_interface = llm_interface
        self.scaffold_router = scaffold_router
        self.evaluator = evaluation_heuristics
        self.retry_policy = retry_policy
        self.memory_interface = memory_interface
        self.execution_strategy = execution_strategy
        self.default_system_prompt = default_system_prompt or "You are a helpful AI assistant."
        self.max_iterations = max_iterations
        
        # Attempt counters
        self.attempt_counters = defaultdict(int)
        
        logger.info("VoxSigil Supervisor initialized")
    
    def process_query(self, 
                    query: str, 
                    system_prompt: Optional[str] = None,
                    context: Optional[str] = None,
                    scaffold: Optional[str] = None) -> str:
        """
        Process a query through the VoxSigil pipeline.
        
        Args:
            query: The user's query to process.
            system_prompt: Optional system prompt to use instead of default.
            context: Optional context to use. If None, RAG will be used if available.
            scaffold: Optional reasoning scaffold. If None, scaffold router will be used.
            
        Returns:
            A response for the query.
        """
        logger.info(f"Processing query: {query[:50]}...")
        
        # Create a unique ID for this query attempt
        query_id = str(hash(query))[:8]
        
        # If no context provided, try to get it from RAG
        if not context and self.rag_interface:
            try:
                context = self.rag_interface.retrieve_context(query)
                logger.info(f"Retrieved context of length {len(context) if context else 0}")
            except Exception as e:
                logger.error(f"Error retrieving context: {e}")
        
        # If no scaffold provided, try to get it from the scaffold router
        if not scaffold and self.scaffold_router:
            try:
                scaffold_template = self.scaffold_router.select_scaffold(query, context)
                scaffold = self.scaffold_router.apply_scaffold(scaffold_template, query, context)
                logger.info(f"Generated scaffold of length {len(scaffold) if scaffold else 0}")
            except Exception as e:
                logger.error(f"Error generating scaffold: {e}")
        
        # Use the provided system prompt or fall back to default
        system_prompt = system_prompt or self.default_system_prompt
        
        # Build the prompt for the LLM
        try:
            prompt = self.llm_interface.build_prompt(query, context, scaffold)
        except Exception as e:
            logger.error(f"Error building prompt: {e}")
            prompt = query  # Fall back to just the query
        
        # Initialize attempt tracking
        attempt = 0
        best_response = None
        best_response_score = 0.0
        
        # Try generating responses with retries if needed
        while attempt < self.max_iterations:
            logger.info(f"Attempt {attempt+1}/{self.max_iterations} for query {query_id}")
            
            try:
                # Format messages for the LLM
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                
                # Generate response
                logger.info(f"Generating response with LLM interface")
                response, model_info, generation_info = self.llm_interface.generate_response(
                    messages, temperature=0.7 if attempt == 0 else 0.8
                )
                
                if not response:
                    logger.warning("Empty response from LLM")
                    attempt += 1
                    continue
                
                # Evaluate the response if we have an evaluator
                if self.evaluator:
                    logger.info("Evaluating response")
                    evaluation = self.evaluator.evaluate_response(query, response, context)
                    logger.info(f"Evaluation results: {evaluation}")
                    
                    response_score = evaluation.get("quality_score", 0.0)
                    passes_threshold = evaluation.get("passes_threshold", False)
                    
                    # Track the best response
                    if response_score > best_response_score:
                        best_response = response
                        best_response_score = response_score
                    
                    # Check if we should retry
                    if not passes_threshold:
                        logger.info(f"Response quality {response_score:.2f} below threshold")
                        
                        if self.retry_policy:
                            should_retry, retry_info = self.retry_policy.should_retry(
                                attempt, response_score, None
                            )
                            
                            if should_retry and attempt < self.max_iterations - 1:
                                logger.info(f"Retry policy suggests retry")
                                
                                # Get retry strategy
                                strategy = self.retry_policy.get_retry_strategy(attempt)
                                logger.info(f"Using retry strategy: {strategy}")
                                
                                # Apply retry strategy
                                modified_query, modified_context, modified_scaffold = (
                                    self.retry_policy.apply_retry_strategy(
                                        strategy, query, context, scaffold
                                    )
                                )
                                
                                # Update prompt with modified inputs
                                prompt = self.llm_interface.build_prompt(
                                    modified_query, modified_context, modified_scaffold
                                )
                                
                                attempt += 1
                                continue
                    
                    # If we're here, either the response passed the threshold or we shouldn't retry
                    if passes_threshold:
                        logger.info(f"Response passed quality threshold")
                        return response
                else:
                    # No evaluator, just return the response
                    logger.info("No evaluator available, accepting response")
                    return response
            
            except Exception as e:
                logger.error(f"Error in attempt {attempt+1}: {e}")
                
                # Check if we should retry after an error
                if self.retry_policy:
                    should_retry, retry_info = self.retry_policy.should_retry(
                        attempt, None, e
                    )
                    
                    if should_retry and attempt < self.max_iterations - 1:
                        logger.info(f"Retrying after error: {e}")
                        attempt += 1
                        continue
                
                # If we shouldn't retry, return an error message
                return f"Error generating response: {e}"
            
            # Increment attempt counter
            attempt += 1
        
        # If we've exhausted all attempts, return the best response or an error
        if best_response:
            logger.info(f"Returning best response after {attempt} attempts")
            return best_response
        else:
            logger.warning(f"Failed to generate a satisfactory response after {attempt} attempts")
            return "I was unable to generate a satisfactory response after multiple attempts."

# For backward compatibility
VoxSigilSupervisor = VoxSigilSupervisorCompat
