#!/usr/bin/env python
"""
hybrid_middleware.py - Implementation of the Hybrid Middleware

This file implements the Hybrid Middleware interface for VantaCore,
providing RAG capabilities combined with LLM processing for ARC tasks.

This file serves as an integration layer between VantaCore and the production
BLT implementation in hybrid_blt.py.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from Vanta.interfaces.blt_encoder_interface import (
    BaseBLTEncoder,
)
from Vanta.interfaces.hybrid_middleware_interface import BaseHybridMiddleware
from Vanta.interfaces.supervisor_connector_interface import (
    BaseSupervisorConnector,
)

# Import hybrid_blt components
from .hybrid_blt import (
    HybridMiddleware as BLTHybridMiddleware,
)
from .hybrid_blt import (
    HybridMiddlewareConfig,
)

logger = logging.getLogger("VoxSigil.HybridMiddleware")


class HybridMiddleware(BaseHybridMiddleware):
    """
    Implementation of the Hybrid Middleware interface using RAG and LLMs.

    This implementation combines retrieval-augmented generation with large language models
    to effectively process ARC tasks with adaptive parameters. It delegates to the
    production-grade implementation in hybrid_blt.py.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        blt_encoder_instance: Optional[BaseBLTEncoder] = None,
        supervisor_connector: Optional[BaseSupervisorConnector] = None,
    ):
        """
        Initialize the Hybrid Middleware with configuration and required components.

        Args:
            config: Dictionary containing configuration parameters
            blt_encoder_instance: BLT encoder for text embeddings and semantic search
            supervisor_connector: Connector to the VoxSigil Supervisor
        """
        self.config = config or {}

        # Require both blt_encoder and supervisor_connector
        if blt_encoder_instance is None:
            raise ValueError("BLT encoder instance must be provided")
        self.blt_encoder = blt_encoder_instance

        if supervisor_connector is None:
            raise ValueError("Supervisor connector must be provided")
        self.supervisor_connector = supervisor_connector

        # RAG configuration
        self.rag_enabled = self.config.get("rag_enabled", True)
        self.max_rag_results = self.config.get("max_rag_results", 5)

        # LLM configuration
        self.llm_model = self.config.get("llm_model", "voxsigil-arc-optimized-7b")
        self.default_temperature = self.config.get("default_temperature", 0.2)
        self.max_tokens_solution = self.config.get("max_tokens_solution", 1024)
        self.auto_retry_count = self.config.get("auto_retry_count", 2)

        # Prompting
        self.prompt_template_sigil_ref = self.config.get(
            "prompt_template_sigil_ref", "Sigil:ARC_Prompt_Template_V2.3"
        )

        # Performance evaluation
        self.performance_evaluation_method = self.config.get(
            "performance_evaluation_method", "rule_based_validator"
        )
        self.solution_format = self.config.get(
            "solution_format", "json_arc_compliant"
        )  # Initialize BLT hybrid middleware
        blt_config = HybridMiddlewareConfig(
            entropy_threshold=self.config.get("entropy_threshold", 0.25),
            blt_hybrid_weight=self.config.get("blt_hybrid_weight", 0.7),
            entropy_router_fallback=self.config.get(
                "entropy_router_fallback", "token_based"
            ),
            cache_ttl_seconds=self.config.get("cache_ttl_seconds", 300),
            log_level=self.config.get("log_level", "INFO"),
        )

        try:
            self.blt_middleware = BLTHybridMiddleware(blt_config)
            logger.info("Successfully initialized BLT Hybrid Middleware implementation")
        except Exception as e:
            logger.error(
                f"Error initializing BLT Hybrid Middleware: {e}", exc_info=True
            )
            logger.warning("Continuing with basic implementation only")
            self.blt_middleware = None

        logger.info(
            f"HybridMiddleware initialized with model '{self.llm_model}', "
            f"RAG enabled: {self.rag_enabled}, BLT implementation: {'available' if self.blt_middleware else 'unavailable'}"
        )

    def process_arc_task(
        self,
        input_data_sigil_ref: str,
        task_sigil_ref: str,
        task_parameters: Dict[str, Any],
    ) -> Tuple[str, str]:
        """
        Process an ARC task using RAG and LLM.

        Args:
            input_data_sigil_ref: Sigil reference to the input data
            task_sigil_ref: Sigil reference to the task definition
            task_parameters: Parameters for processing the task, including:
                - effective_temperature: Temperature to use for LLM
                - max_solution_attempts: Number of attempts to try
                - Other task-specific parameters

        Returns:
            Tuple[str, str]: Sigil references to the solution and performance metric
        """
        logger.info(
            f"Processing ARC task '{task_sigil_ref}' with input '{input_data_sigil_ref}'"
        )

        try:
            # 1. Retrieve input and task content
            input_data = self.supervisor_connector.get_sigil_content_as_dict(
                input_data_sigil_ref
            )
            if not input_data:
                raise ValueError(
                    f"Failed to retrieve content for input sigil '{input_data_sigil_ref}'"
                )

            task_data = self.supervisor_connector.get_sigil_content_as_dict(
                task_sigil_ref
            )
            if not task_data:
                raise ValueError(
                    f"Failed to retrieve content for task sigil '{task_sigil_ref}'"
                )  # 2. Perform RAG if enabled
            relevant_examples = []
            if self.rag_enabled:
                if self.blt_middleware:
                    # Use the enhanced BLT RAG capabilities
                    task_description = task_data.get("description", "")
                    if not task_description and "grid" in input_data:
                        task_description = f"ARC task with grid size {len(input_data['grid'])}x{len(input_data['grid'][0])}"

                    # Process the task description with BLT middleware
                    enriched_text, sigils, route_type = self.blt_middleware.process(
                        task_description, num_sigils=self.max_rag_results
                    )

                    relevant_examples = sigils
                    logger.debug(
                        f"BLT RAG retrieved {len(relevant_examples)} examples via {route_type} route"
                    )
                else:
                    # Use the standard RAG approach
                    relevant_examples = self._perform_rag(
                        input_data, task_data, task_parameters
                    )
                    logger.debug(
                        f"Standard RAG retrieved {len(relevant_examples)} relevant examples"
                    )

            # 3. Get prompt template
            prompt_template = self._get_prompt_template()

            # 4. Generate solution using LLM
            solution_content = self._generate_solution(
                input_data,
                task_data,
                relevant_examples,
                prompt_template,
                task_parameters,
            )

            # 5. Create solution sigil
            solution_sigil_ref = f"SigilRef:Solution_{task_sigil_ref}_{time.time_ns()}"
            self.supervisor_connector.create_sigil(
                solution_sigil_ref, solution_content, "ARC_Solution"
            )

            # 6. Evaluate performance
            perf_metrics = self._evaluate_performance(
                input_data, task_data, solution_content, task_parameters
            )

            # 7. Create performance metrics sigil
            perf_metric_sigil_ref = (
                f"SigilRef:PerfMetric_{task_sigil_ref}_{time.time_ns()}"
            )
            self.supervisor_connector.create_sigil(
                perf_metric_sigil_ref, perf_metrics, "PerformanceMetric"
            )

            logger.info(
                f"Task '{task_sigil_ref}' processed. "
                f"Solution: '{solution_sigil_ref}', "
                f"Performance: {perf_metrics.get('achieved_performance', 0.0):.3f}"
            )

            return solution_sigil_ref, perf_metric_sigil_ref

        except Exception as e:
            logger.error(
                f"Error processing task '{task_sigil_ref}': {e}", exc_info=True
            )

            # Create failure solution and metrics
            failure_solution_content = {
                "error": str(e),
                "task_sigil_ref": task_sigil_ref,
                "input_sigil_ref": input_data_sigil_ref,
                "status": "failed",
            }
            failure_solution_sigil_ref = (
                f"SigilRef:Solution_Failed_{task_sigil_ref}_{time.time_ns()}"
            )

            failure_metrics = {
                "task_sigil_ref": task_sigil_ref,
                "input_sigil_ref": input_data_sigil_ref,
                "achieved_performance": 0.0,
                "llm_model_used_ref": self.llm_model,
                "error": str(e),
                "parameters_applied": task_parameters,
            }
            failure_metric_sigil_ref = (
                f"SigilRef:PerfMetric_Failed_{task_sigil_ref}_{time.time_ns()}"
            )

            try:
                self.supervisor_connector.create_sigil(
                    failure_solution_sigil_ref,
                    failure_solution_content,
                    "ARC_Solution_Failed",
                )
                self.supervisor_connector.create_sigil(
                    failure_metric_sigil_ref, failure_metrics, "PerformanceMetric"
                )
            except Exception as inner_e:
                logger.error(f"Error creating failure sigils: {inner_e}")

            return failure_solution_sigil_ref, failure_metric_sigil_ref

    def _perform_rag(
        self,
        input_data: Dict[str, Any],
        task_data: Dict[str, Any],
        task_parameters: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Perform RAG to retrieve relevant examples for the given task.

        Returns:
            List[Dict[str, Any]]: List of relevant example data
        """
        # Extract description or generate it
        task_description = task_data.get("description", "")
        if not task_description and "grid" in input_data:
            # Generate a description based on the input grid if needed
            task_description = f"ARC task with grid size {len(input_data['grid'])}x{len(input_data['grid'][0])}"

        try:
            # Get embedding for the task
            embedding = self.blt_encoder.encode(task_description, task_type="arc_task")

            # Use supervisor to query for similar examples
            example_query = {
                "embedding": embedding,
                "max_results": self.max_rag_results,
                "min_similarity": 0.7,
                "collection": "arc_examples",
            }

            # Assume the supervisor has a method to find similar examples
            similar_examples = self.supervisor_connector.find_similar_examples(
                example_query
            )
            return similar_examples or []

        except Exception as e:
            logger.warning(f"Error during RAG retrieval: {e}")
            return []

    def _get_prompt_template(self) -> Dict[str, Any]:
        """Get the prompt template for ARC task processing."""
        try:
            template = self.supervisor_connector.get_sigil_content_as_dict(
                self.prompt_template_sigil_ref
            )
            if not template:
                logger.warning(
                    f"Failed to retrieve prompt template '{self.prompt_template_sigil_ref}'"
                )
                # Return a minimal default template
                return {
                    "template": "Solve the following ARC task: {{task_description}}"
                }
            return template
        except Exception as e:
            logger.warning(f"Error retrieving prompt template: {e}")
            # Return minimal default
            return {"template": "Solve the following ARC task: {{task_description}}"}

    def _generate_solution(
        self,
        input_data: Dict[str, Any],
        task_data: Dict[str, Any],
        relevant_examples: List[Dict[str, Any]],
        prompt_template: Dict[str, Any],
        task_parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate a solution using an LLM.

        Returns:
            Dict[str, Any]: Solution content
        """
        # Combine data into a prompt
        temperature = task_parameters.get(
            "effective_temperature", self.default_temperature
        )
        max_attempts = task_parameters.get("max_solution_attempts", 1)

        # Format examples for the prompt
        examples_text = ""
        if relevant_examples:
            examples_text = "Here are some examples of similar tasks:\n"
            for i, example in enumerate(relevant_examples[:3]):  # Limit to 3 examples
                examples_text += f"Example {i + 1}:\n"
                examples_text += json.dumps(example.get("input", {})) + "\n"
                examples_text += "Solution:\n"
                examples_text += json.dumps(example.get("solution", {})) + "\n\n"

        # Prepare prompt
        template_text = prompt_template.get(
            "template", "Solve this ARC task: {{task_description}}"
        )
        prompt = (
            template_text.replace(
                "{{task_description}}", json.dumps(task_data.get("description", ""))
            )
            .replace("{{input_data}}", json.dumps(input_data))
            .replace("{{examples}}", examples_text)
            .replace("{{format}}", self.solution_format)
        )  # Generate solution with retries
        solution = None
        for attempt in range(max_attempts):
            try:
                # Call LLM through supervisor
                llm_params = {
                    "model": self.llm_model,
                    "prompt": prompt,
                    "temperature": temperature
                    + (attempt * 0.1),  # Increase temperature slightly for retries
                    "max_tokens": self.max_tokens_solution,
                    "format": self.solution_format,
                }

                # Use BLT middleware to enhance the prompt if available
                if (
                    self.blt_middleware and attempt == 0
                ):  # Only on first attempt to avoid confusion
                    try:
                        # Process the prompt to enrich it
                        enriched_prompt, _, route_type = self.blt_middleware.process(
                            prompt
                        )
                        logger.info(
                            f"Enhanced prompt using BLT middleware via {route_type} route"
                        )
                        llm_params["prompt"] = enriched_prompt
                    except Exception as e:
                        logger.warning(f"Error enhancing prompt with BLT: {e}")
                        # Continue with original prompt

                response = self.supervisor_connector.call_llm(llm_params)

                # Parse response
                try:
                    if isinstance(response, str):
                        solution = json.loads(response)
                    else:
                        solution = response

                    # If solution looks valid, break out of retry loop
                    if self._validate_solution_format(solution, input_data):
                        break
                except json.JSONDecodeError:
                    logger.warning(
                        f"Failed to parse LLM response as JSON (attempt {attempt + 1}/{max_attempts})"
                    )
                    solution = {"raw_response": response, "parsing_error": True}

                logger.debug(f"Solution attempt {attempt + 1}/{max_attempts} completed")

            except Exception as e:
                logger.warning(
                    f"Error generating solution (attempt {attempt + 1}/{max_attempts}): {e}"
                )
                solution = {"error": str(e)}

        # Include metadata with the solution
        solution_with_metadata = solution or {"error": "Failed to generate solution"}
        solution_with_metadata.update(
            task_sigil_ref=task_data.get("sigil_ref", ""),
            input_sigil_ref=input_data.get("sigil_ref", ""),
            llm_model_used=self.llm_model,
            temperature_used=temperature,
            rag_enabled=self.rag_enabled,
            rag_examples_count=str(len(relevant_examples)),
            generation_timestamp=str(time.time()),
        )

        return solution_with_metadata

    def _validate_solution_format(
        self, solution: Dict[str, Any], input_data: Dict[str, Any]
    ) -> bool:
        """Validate that the solution has the expected format."""
        # Basic format check - will depend on the solution_format specified
        if self.solution_format == "json_arc_compliant":
            return "grid" in solution or "output_grid" in solution
        return True

    def _evaluate_performance(
        self,
        input_data: Dict[str, Any],
        task_data: Dict[str, Any],
        solution_content: Dict[str, Any],
        task_parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate the performance of the generated solution.

        Returns:
            Dict[str, Any]: Performance metrics
        """
        # Get ground truth if available
        ground_truth = task_data.get("output", task_data.get("expected_output"))

        achieved_performance = 0.0

        if (
            self.performance_evaluation_method == "rule_based_validator"
            and ground_truth
        ):
            # Compare solution with ground truth
            solution_grid = solution_content.get(
                "grid", solution_content.get("output_grid")
            )
            if solution_grid and ground_truth:
                if isinstance(ground_truth, dict) and "grid" in ground_truth:
                    ground_truth_grid = ground_truth["grid"]
                else:
                    ground_truth_grid = ground_truth

                # Simple grid comparison
                if solution_grid == ground_truth_grid:
                    achieved_performance = 1.0
                else:
                    # Partial credit based on similarity
                    achieved_performance = self._calculate_grid_similarity(
                        solution_grid, ground_truth_grid
                    )
        elif "error" in solution_content or "parsing_error" in solution_content:
            # Error case - performance is 0
            achieved_performance = 0.0
        else:
            # No ground truth or validation method - assume moderate success
            achieved_performance = 0.5

        # Create performance metrics
        return {
            "task_sigil_ref": task_data.get("sigil_ref", ""),
            "input_sigil_ref": input_data.get("sigil_ref", ""),
            "achieved_performance": achieved_performance,
            "llm_model_used_ref": self.llm_model,
            "tokens_prompt": len(str(task_data))
            + len(str(input_data))
            + 500,  # Estimate
            "tokens_completion": len(str(solution_content)),
            "duration_ms": int(time.time() * 1000) % 10000,  # Just a placeholder
            "parameters_applied": task_parameters,
            "ground_truth_available": ground_truth is not None,
            "evaluation_method": self.performance_evaluation_method,
        }

    def _calculate_grid_similarity(self, grid1, grid2) -> float:
        """Calculate similarity between two grids for partial performance credit."""
        if not isinstance(grid1, list) or not isinstance(grid2, list):
            return 0.0

        try:
            # Count matching cells
            total_cells = 0
            matching_cells = 0

            # Get dimensions
            rows1, cols1 = len(grid1), len(grid1[0]) if grid1 else 0
            rows2, cols2 = len(grid2), len(grid2[0]) if grid2 else 0

            if rows1 != rows2 or cols1 != cols2:
                # Dimensional mismatch - low similarity
                return 0.2

            for i in range(rows1):
                for j in range(cols1):
                    total_cells += 1
                    if grid1[i][j] == grid2[i][j]:
                        matching_cells += 1

            return matching_cells / total_cells if total_cells > 0 else 0.0

        except Exception as e:
            logger.warning(f"Error calculating grid similarity: {e}")
            return 0.0

    def find_similar_examples(self, example_data: Any) -> Optional[Dict[str, Any]]:
        """
        Finds similar examples based on the provided data.

        Args:
            example_data (Any): Data to find similar examples for.

        Returns:
            Optional[Dict[str, Any]]: A dictionary of similar examples or None if no examples are found.
        """
        logger.info(f"Finding similar examples for data: {str(example_data)[:100]}...")

        try:
            # Convert example_data to string if it's not already
            if not isinstance(example_data, str):
                try:
                    example_text = json.dumps(example_data)
                except Exception:
                    example_text = str(example_data)
            else:
                example_text = example_data

            # Use BLT middleware if available
            if self.blt_middleware:
                logger.debug("Using BLT implementation for finding similar examples")
                # Get the enhanced RAG context
                enriched_text, sigils, route_type = self.blt_middleware.process(
                    example_text, num_sigils=self.max_rag_results
                )
                # Format the results
                similar_examples = {
                    "query": example_text,
                    "found_examples": len(sigils),
                    "processing_route": route_type,
                    "examples": sigils,
                    "source": "blt_hybrid_implementation",
                }
                return similar_examples
            else:
                # Fallback to basic implementation using direct similarity search
                logger.debug(
                    "Using fallback implementation for finding similar examples"
                )
                # Convert to simplified structure for basic similarity matching
                similar_sigils = []

                # Try to extract some features from the example
                try:
                    if isinstance(example_data, dict):
                        # Try to match any keys or values
                        search_terms = []
                        for k, v in example_data.items():
                            search_terms.append(str(k))
                            search_terms.append(
                                str(v)
                            )  # Use supervisor to search for similar content
                        for term in search_terms[:5]:  # Limit to first 5 terms
                            matches = self.supervisor_connector.search_sigils(
                                {"text": term, "max_results": 2}
                            )
                            if matches:
                                similar_sigils.extend(matches)
                    else:
                        # Use the example text directly
                        similar_sigils = self.supervisor_connector.search_sigils(
                            {
                                "text": example_text[:100],
                                "max_results": self.max_rag_results,
                            }
                        )
                except Exception as e:
                    logger.warning(f"Fallback similarity search failed: {e}")

                return {
                    "query": example_text,
                    "found_examples": len(similar_sigils),
                    "processing_route": "basic_fallback",
                    "examples": similar_sigils,
                    "source": "basic_sigil_search",
                }

        except Exception as e:
            logger.error(f"Error finding similar examples: {e}", exc_info=True)
            return None
