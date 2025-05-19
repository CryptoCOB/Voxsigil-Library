#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VoxSigil Supervisor compatibility wrapper.
This script provides a compatibility layer for the VoxSigil Supervisor.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("voxsigil_compat")

class VoxSigilSupervisorWrapper:
    """A simplified VoxSigil Supervisor that uses our custom components"""
    
    def __init__(self, 
                rag_interface, 
                llm_interface, 
                scaffold_router,
                evaluation_heuristics,
                retry_policy,
                max_iterations=2):
        """Initialize the VoxSigil Supervisor wrapper."""
        self.rag_interface = rag_interface
        self.llm_interface = llm_interface
        self.scaffold_router = scaffold_router
        self.evaluator = evaluation_heuristics
        self.retry_policy = retry_policy
        self.max_iterations = max_iterations
        self.logger = logging.getLogger("voxsigil_supervisor.wrapper")
        self.logger.info("VoxSigil Supervisor wrapper initialized")
    
    def process_query(self, 
                    query: str, 
                    system_prompt: Optional[str] = None,
                    context: Optional[str] = None,
                    scaffold: Optional[str] = None) -> str:
        """
        Process a query using the VoxSigil components.
        
        Args:
            query: The user's query
            system_prompt: Optional system prompt to use
            context: Optional context from RAG
            scaffold: Optional reasoning scaffold
            
        Returns:
            The generated response
        """
        self.logger.info(f"Processing query: {query[:50]}...")
        
        # If no context provided, try to get it from RAG
        if not context and self.rag_interface:
            try:
                context = self.rag_interface.retrieve_context(query)
                self.logger.info(f"Retrieved context of length {len(context) if context else 0}")
            except Exception as e:
                self.logger.error(f"Error retrieving context: {e}")
        
        # If no scaffold provided, try to get it from the scaffold router
        if not scaffold and self.scaffold_router:
            try:
                scaffold_template = self.scaffold_router.select_scaffold(query, context)
                scaffold = self.scaffold_router.apply_scaffold(scaffold_template, query, context)
                self.logger.info(f"Generated scaffold of length {len(scaffold) if scaffold else 0}")
            except Exception as e:
                self.logger.error(f"Error generating scaffold: {e}")
        
        # Build the prompt for the LLM
        prompt = self.llm_interface.build_prompt(query, context, scaffold)
        
        # Generate a response using the LLM
        attempt = 0
        response = None
        
        while attempt < self.max_iterations:
            try:
                # Format messages for the LLM
                messages = [
                    {"role": "system", "content": system_prompt or "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                
                # Generate response
                response, model_info, generation_info = self.llm_interface.generate_response(
                    messages, temperature=0.7
                )
                
                # Evaluate the response if we have an evaluator
                if self.evaluator and response:
                    evaluation = self.evaluator.evaluate_response(query, response, context)
                    self.logger.info(f"Response evaluation: {evaluation}")
                    
                    # Check if the response passes the quality threshold
                    if evaluation.get("passes_threshold", True):
                        self.logger.info("Response passed quality threshold")
                        break
                    else:
                        self.logger.info("Response did not pass quality threshold")
                        
                        # Check if we should retry
                        should_retry, retry_info = self.retry_policy.should_retry(
                            attempt, evaluation.get("quality_score"), None
                        )
                        
                        if should_retry:
                            # Get retry strategy
                            strategy = self.retry_policy.get_retry_strategy(attempt)
                            
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
                            self.logger.info(f"Retrying (attempt {attempt})")
                            continue
                        else:
                            self.logger.info("Not retrying despite low quality")
                            break
                else:
                    # No evaluator, just accept the response
                    break
                    
            except Exception as e:
                self.logger.error(f"Error generating response (attempt {attempt}): {e}")
                
                # Check if we should retry
                if self.retry_policy:
                    should_retry, retry_info = self.retry_policy.should_retry(
                        attempt, None, e
                    )
                    
                    if should_retry and attempt < self.max_iterations - 1:
                        attempt += 1
                        self.logger.info(f"Retrying after error (attempt {attempt})")
                        continue
                
                # Return error message if all retries failed
                return f"Error generating response: {e}"
            
            attempt += 1
        
        return response or "Failed to generate a response"
