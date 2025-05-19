# voxsigil_supervisor/supervisor_engine.py
"""
Main orchestrator class for the VoxSigil Supervisor.
"""
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import defaultdict

# Interfaces
from .interfaces.rag_interface import BaseRagInterface
from .interfaces.llm_interface import BaseLlmInterface
from .interfaces.memory_interface import BaseMemoryInterface # Optional

# Strategy components
from .strategies.scaffold_router import ScaffoldRouter
from .strategies.evaluation_heuristics import ResponseEvaluator
from .strategies.retry_policy import RetryPolicy
from .strategies.execution_strategy import BaseExecutionStrategy # If using complex execution

# Utils
from .utils.logging_utils import setup_supervisor_logging, SUPERVISOR_LOGGER_NAME

logger = logging.getLogger(SUPERVISOR_LOGGER_NAME)

class VoxSigilSupervisor:
    """
    The VoxSigilSupervisor orchestrates the interaction between RAG, LLM,
    and strategic reasoning components to process queries using VoxSigil.
    """
    
    def __init__(self,
                 rag_interface: BaseRagInterface,
                 llm_interface: BaseLlmInterface,
                 scaffold_router: ScaffoldRouter,
                 evaluation_heuristics: ResponseEvaluator,
                 retry_policy: RetryPolicy,
                 memory_interface: Optional[BaseMemoryInterface] = None,
                 execution_strategy: Optional[BaseExecutionStrategy] = None,
                 default_system_prompt: Optional[str] = None,
                 max_iterations: int = 3):
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
            default_system_prompt: Default system prompt for the LLM.
            max_iterations: Maximum number of reasoning iterations per query.
        """
        self.rag_interface = rag_interface
        self.llm_interface = llm_interface
        self.scaffold_router = scaffold_router
        self.evaluation_heuristics = evaluation_heuristics
        self.retry_policy = retry_policy
        self.memory_interface = memory_interface
        self.execution_strategy = execution_strategy
        self.default_system_prompt = default_system_prompt
        self.max_iterations = max_iterations
        
        # Statistics tracking
        self.stats = {
            "queries_processed": 0,
            "total_retries": 0,
            "scaffold_usage": defaultdict(int),
            "avg_evaluation_score": 0,
            "execution_times": []
        }
        
        logger.info("🟢 VoxSigilSupervisor initialized successfully")
        
    def solve(self, query: str, task_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main entry point for processing a query through the VoxSigil Supervisor.
        
        Args:
            query: The user query or task to process.
            task_metadata: Optional metadata about the task/query (e.g., domain, complexity).
            
        Returns:
            A dictionary containing the final response, evaluation metrics, and process metadata.
        """
        start_time = time.time()
        self.stats["queries_processed"] += 1
        task_metadata = task_metadata or {}
        
        logger.info(f"Processing query: '{query[:50]}{'...' if len(query) > 50 else ''}'")
        
        # Initialize tracking for this query
        iterations = 0
        history = []
        best_response = None
        best_score = 0
        
        # 1. Select initial scaffold based on query
        selected_scaffold = self.scaffold_router.select_scaffold(query, task_metadata)
        self.stats["scaffold_usage"][selected_scaffold] += 1
        logger.info(f"Selected scaffold: {selected_scaffold}")
        
        # 2. Retrieve relevant context from RAG
        sigils_context = self.rag_interface.retrieve_context(query, {
            "scaffold": selected_scaffold,
            **task_metadata
        })
        logger.debug(f"Retrieved context length: {len(sigils_context)} chars")
        
        # Store conversation history for multi-turn reasoning
        messages = [{"role": "user", "content": query}]
        
        while iterations < self.max_iterations:
            iterations += 1
            
            # 3. Build prompt with scaffold, context, and query
            if self.execution_strategy and selected_scaffold:
                # Use execution strategy for complex scaffolds
                prompt = self.execution_strategy.build_scaffold_prompt(
                    query, sigils_context, selected_scaffold, history, task_metadata
                )
            else:
                # Simple prompt building with just context
                prompt = self._build_simple_prompt(query, sigils_context, selected_scaffold)
            
            # 4. Get LLM response
            messages[-1]["content"] = prompt  # Update the last message with enhanced prompt
            response_text, model_info, response_metadata = self.llm_interface.generate_response(
                messages=messages,
                system_prompt_override=self.default_system_prompt,
                task_requirements={"scaffold": selected_scaffold, **task_metadata}
            )
            
            if not response_text:
                logger.error("Failed to get response from LLM")
                break
                  # 5. Evaluate response
            evaluation_result = self.evaluation_heuristics.evaluate(
                query=query,
                response=response_text,
                context=selected_scaffold
            )
            
            # Record this iteration
            current_iteration = {
                "iteration": iterations,
                "scaffold": selected_scaffold,
                "response": response_text,
                "evaluation": evaluation_result,
                "model_info": model_info
            }
            history.append(current_iteration)
            
            # Track best response so far
            current_score = evaluation_result.get("total_score", 0)
            if current_score > best_score:
                best_response = current_iteration
                best_score = current_score
            
            # 6. Check if we should retry with different approach
            if self.retry_policy.should_retry(evaluation_result, iterations, self.max_iterations):
                self.stats["total_retries"] += 1
                
                # Get retry recommendations
                retry_scaffold, retry_context_params = self.retry_policy.get_retry_strategy(
                    query, selected_scaffold, evaluation_result, history
                )
                
                # Update scaffold if changed
                if retry_scaffold != selected_scaffold:
                    selected_scaffold = retry_scaffold
                    self.stats["scaffold_usage"][selected_scaffold] += 1
                    logger.info(f"Retrying with new scaffold: {selected_scaffold}")
                    
                    # Get new context if scaffold changed
                    sigils_context = self.rag_interface.retrieve_context(query, {
                        "scaffold": selected_scaffold,
                        **retry_context_params,
                        **task_metadata
                    })
                else:
                    logger.info(f"Retrying with same scaffold: {selected_scaffold}")
            else:
                # No need to retry, we're done
                logger.info(f"No retry needed. Process complete after {iterations} iterations.")
                break
        
        # 7. Save to memory if available
        if self.memory_interface:
            memory_key = self.memory_interface.store(
                query=query,
                response=best_response["response"] if best_response else response_text,
                metadata={
                    "scaffold": selected_scaffold,
                    "score": best_score,
                    "iterations": iterations
                }
            )
            logger.debug(f"Stored in memory with key: {memory_key}")
        
        # Update stats
        execution_time = time.time() - start_time
        self.stats["execution_times"].append(execution_time)
        total_scores = sum(h["evaluation"].get("total_score", 0) for h in history if "evaluation" in h)
        if history:  # Avoid division by zero
            self.stats["avg_evaluation_score"] = (
                (self.stats["avg_evaluation_score"] * (self.stats["queries_processed"] - 1) + 
                 total_scores / len(history)) / self.stats["queries_processed"]
            )
        
        # 8. Return final result with all metadata
        final_response = best_response or history[-1] if history else {"response": "Failed to generate response"}
        result = {
            "query": query,
            "response": final_response.get("response") if isinstance(final_response, dict) else "Error",
            "scaffold": selected_scaffold,
            "evaluation": final_response.get("evaluation") if isinstance(final_response, dict) else {},
            "iterations": iterations,
            "history": history,
            "execution_time_seconds": execution_time,
            "model_info": model_info
        }
        
        logger.info(f"Query processed in {execution_time:.2f}s with {iterations} iterations. Score: {best_score:.2f}")
        return result

    def _build_simple_prompt(self, query: str, context: str, scaffold: Optional[str]) -> str:
        """Simple prompt builder when not using a complex execution strategy."""
        prompt_parts = []
        
        if scaffold:
            prompt_parts.append(f"REASONING SCAFFOLD: {scaffold}")
        
        prompt_parts.append("CONTEXT:")
        prompt_parts.append(context)
        prompt_parts.append("\nQUERY:")
        prompt_parts.append(query)
        
        return "\n".join(prompt_parts)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Returns performance statistics from the supervisor."""
        return {
            "queries_processed": self.stats["queries_processed"],
            "total_retries": self.stats["total_retries"],
            "scaffold_usage": dict(self.stats["scaffold_usage"]),
            "avg_evaluation_score": self.stats["avg_evaluation_score"],
            "avg_execution_time": sum(self.stats["execution_times"]) / len(self.stats["execution_times"]) 
                if self.stats["execution_times"] else 0
        }
        self.rag_interface = rag_interface
        self.llm_interface = llm_interface
        self.scaffold_router = scaffold_router
        self.evaluation_heuristics = evaluation_heuristics
        self.retry_policy = retry_policy
        self.memory_interface = memory_interface
        # self.execution_strategy = execution_strategy or DefaultExecutionStrategy()

        self.default_system_prompt = default_system_prompt or \
            "You are Voxka, a symbolic reasoning AI. You operate using Voxsigils and scaffolds. Be precise and structured."
        self.max_iterations = max_iterations

        logger.info("VoxSigilSupervisor initialized.")

    def process_query(self, user_query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Processes a user query through the VoxSigil reasoning pipeline.

        Args:
            user_query: The input query from the user.
            conversation_history: Optional list of past conversation turns.

        Returns:
            A tuple containing:
                - The final response string.
                - A dictionary with metadata about the reasoning process (trace, sigils used, etc.).
        """
        logger.info(f"Processing query: '{user_query[:100]}...'")
        if conversation_history is None:
            conversation_history = []

        current_iteration = 0
        current_reasoning_state = {"query": user_query, "history": conversation_history}
        reasoning_trace = [] # To log steps

        # 1. Initial Strategy & Context Retrieval
        # Select a primary scaffold for the query
        scaffold_info = self.scaffold_router.select_scaffold( # MODIFIED: Store the returned dict
            user_query, current_reasoning_state
        )
        selected_scaffold_name = scaffold_info.get("id") # MODIFIED: Extract id
        scaffold_details = scaffold_info # MODIFIED: Assign the whole dict to details

        if not selected_scaffold_name:
            logger.warning("No suitable scaffold selected. Proceeding with default strategy.")
            # Handle default strategy or error out
            return "Error: Could not determine a reasoning strategy.", {"trace": reasoning_trace, "error": "No scaffold selected"}
        
        reasoning_trace.append({"action": "scaffold_selected", "scaffold": selected_scaffold_name, "details": scaffold_details})

        # Retrieve initial sigils/context using RAG interface
        # Query for RAG might be different based on scaffold
        rag_query = f"{user_query} (Context: Applying scaffold {selected_scaffold_name})"
        retrieved_sigils = self.rag_interface.retrieve_sigils(
            query=rag_query,
            top_k=5 # Example, could be scaffold-dependent
        )
        reasoning_trace.append({"action": "sigils_retrieved", "sigils": [s.get('sigil', s.get('id', 'unknown')) for s in retrieved_sigils]})
        
        # This is where the main loop for scaffold execution would go.
        # For now, a simplified single-shot + retry example.
        # A more complex execution_strategy would manage scaffold plans.

        for attempt in range(self.max_iterations):
            logger.debug(f"Reasoning iteration {attempt + 1} with scaffold '{selected_scaffold_name}'")

            # 2. Build Prompt for LLM
            # This needs a prompt builder, perhaps in utils or strategy, or here for now
            prompt_messages = self._build_llm_prompt(
                user_query=current_reasoning_state["query"], # current task for this iteration
                conversation_history=current_reasoning_state["history"],
                system_prompt=self.default_system_prompt,
                scaffold_name=selected_scaffold_name,
                scaffold_instructions=scaffold_details.get("instructions", ""),
                retrieved_sigils=retrieved_sigils
            )
            reasoning_trace.append({"action": "prompt_built", "prompt_length": len(prompt_messages[-1]['content'])})

            # 3. Call LLM
            llm_response_text = self.llm_interface.generate_response(prompt_messages)
            if not llm_response_text:
                logger.error("LLM failed to generate a response.")
                reasoning_trace.append({"action": "llm_call_failed"})
                # Potentially use retry policy here for API failures
                if self.retry_policy.should_retry_api_error(): continue
                else: break
            
            reasoning_trace.append({"action": "llm_response_received", "response_preview": llm_response_text[:100]})            # 4. Evaluate Response
            evaluation_results = self.evaluation_heuristics.evaluate(
                query=user_query,
                response=llm_response_text,
                context=selected_scaffold_name  # Use selected scaffold as context
            )
            reasoning_trace.append({"action": "response_evaluated", "evaluation": evaluation_results})

            # 5. Retry Policy
            retry_action, feedback_prompt_modifier = self.retry_policy.get_retry_action(
                evaluation_results, current_attempt=attempt
            )

            if retry_action == "ACCEPT":
                logger.info(f"Response accepted after {attempt + 1} iterations.")
                # Update conversation history before returning
                current_reasoning_state["history"].append({"role": "user", "content": user_query})
                current_reasoning_state["history"].append({"role": "assistant", "content": llm_response_text})
                return llm_response_text, {"trace": reasoning_trace, "final_evaluation": evaluation_results}
            elif retry_action == "RETRY":
                logger.info(f"Retrying based on evaluation. Feedback: {feedback_prompt_modifier}")
                # Modify current_reasoning_state for the next iteration
                # e.g., current_reasoning_state["query"] = llm_response_text + "\n" + feedback_prompt_modifier
                # Or, more sophisticated: update "task" based on scaffold and feedback
                current_reasoning_state["query"] = f"Original Query: {user_query}\nPrevious Attempt: {llm_response_text}\nFeedback: {feedback_prompt_modifier}\nPlease revise and respond."
                current_reasoning_state["history"].append({"role": "user", "content": user_query}) # Add original query that led to this
                current_reasoning_state["history"].append({"role": "assistant", "content": llm_response_text}) # Add failed response
                current_reasoning_state["history"].append({"role": "user", "content": f"[Supervisory Feedback]: {feedback_prompt_modifier}"}) # Add feedback as user turn
                
                # Potentially re-retrieve sigils or adjust scaffold here
                # retrieved_sigils = self.rag_interface.retrieve_sigils(...)
            elif retry_action == "FAIL":
                logger.warning(f"Max retries or unrecoverable error. Failing query after {attempt + 1} iterations.")
                return "Error: Could not produce a satisfactory response after multiple attempts.", {"trace": reasoning_trace, "final_evaluation": evaluation_results}
        
        # Fallthrough if max_iterations reached without "ACCEPT"
        logger.warning(f"Max iterations ({self.max_iterations}) reached. Returning last response.")
        return llm_response_text if 'llm_response_text' in locals() else "Error: Max iterations reached without valid response.", {"trace": reasoning_trace}


    def _build_llm_prompt(self, user_query: str,
                          conversation_history: List[Dict[str, str]],
                          system_prompt: str,
                          scaffold_name: str,
                          scaffold_instructions: str,
                          retrieved_sigils: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Constructs the messages payload for the LLM.
        This is a simplified builder. A dedicated utility/class might be better.
        """
        from .utils.sigil_formatting import format_sigils_for_prompt # Lazy import

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add some conversation history (e.g., last 2 turns)
        for turn in conversation_history[-4:]: # Example: last 2 user/assistant pairs
            messages.append(turn)
        
        # Build current user prompt
        user_content = f"Applied Reasoning Scaffold: {scaffold_name}\n"
        if scaffold_instructions:
            user_content += f"Scaffold Instructions: {scaffold_instructions}\n\n"
        
        if retrieved_sigils:
            user_content += "Relevant Voxsigils (Context):\n"
            user_content += format_sigils_for_prompt(retrieved_sigils, detail_level="standard") # Use a utility
            user_content += "\n---\n"
        
        user_content += f"Current Task/Query: {user_query}"
        messages.append({"role": "user", "content": user_content})
        return messages