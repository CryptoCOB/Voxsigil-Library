import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# --- Setup Logging ---
logger = logging.getLogger("VantaMiddleware")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# --- Configuration & Schema Stubs (Replace with actual imports/definitions) ---


class HybridMiddlewareConfig:
    def __init__(self, **kwargs):
        self.entropy_threshold = kwargs.get("entropy_threshold", 0.25)
        self.blt_hybrid_weight = kwargs.get("blt_hybrid_weight", 0.7)
        self.entropy_router_fallback = kwargs.get(
            "entropy_router_fallback", "token_based"
        )
        self.cache_ttl_seconds = kwargs.get("cache_ttl_seconds", 300)
        self.log_level = kwargs.get("log_level", "INFO")
        # For MessageContextEnhancementMiddleware
        self.rag_enabled = kwargs.get("rag_enabled", True)
        self.max_rag_results = kwargs.get("max_rag_results", 5)
        self.llm_model = kwargs.get("llm_model", "default-llm-7b")
        self.default_temperature = kwargs.get("default_temperature", 0.2)
        # Add other fields as used by the original classes


DEFAULT_SIGIL_SCHEMA = {
    "type": "object",
    "properties": {
        "sigil": {"type": "string"},
        "tag": {"type": "string"},
        "tags": {"type": ["array", "string"], "items": {"type": "string"}},
        "principle": {"type": "string"},
        "usage": {"type": "object"},
        "relationships": {"type": "object"},
        # Add other expected fields
    },
    "required": ["sigil"],
}

# --- Base Class Stubs (Replace with actual imports/definitions) ---


class BaseHybridMiddleware:
    def process_arc_task(
        self,
        input_data_sigil_ref: str,
        task_sigil_ref: str,
        task_parameters: Dict[str, Any],
    ) -> Tuple[str, str]:
        raise NotImplementedError


class BaseBLTEncoder:
    def encode(self, text: str, task_type: Optional[str] = None) -> List[float]:
        logger.debug(f"Stub BaseBLTEncoder.encode called for: {text[:50]}...")
        return [0.1] * 128  # Dummy embedding


class BaseSupervisorConnector:
    def get_sigil_content_as_dict(self, sigil_ref: str) -> Optional[Dict[str, Any]]:
        logger.debug(f"Stub Supervisor.get_sigil_content_as_dict for: {sigil_ref}")
        if "ARC_Prompt_Template" in sigil_ref:
            return {
                "template": "Solve {{task_description}} with {{input_data}} and {{examples}}. Format: {{format}}"
            }
        return {
            "sigil_ref": sigil_ref,
            "content": "dummy content",
            "description": f"Description for {sigil_ref}",
        }

    def create_sigil(self, sigil_ref: str, content: Dict[str, Any], sigil_type: str):
        logger.debug(f"Stub Supervisor.create_sigil: {sigil_ref}, type: {sigil_type}")

    def find_similar_examples(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        logger.debug(f"Stub Supervisor.find_similar_examples with query: {query}")
        return [
            {
                "input": {"grid": [[0]]},
                "solution": {"grid": [[1]]},
                "_similarity_score": 0.8,
            }
        ]

    def call_llm(self, params: Dict[str, Any]) -> Union[Dict[str, Any], str]:
        logger.debug(f"Stub Supervisor.call_llm with model: {params.get('model')}")
        return {
            "grid": [[1, 2], [3, 4]],
            "comment": "LLM generated solution",
        }  # Dummy ARC solution

    def search_sigils(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        logger.debug(f"Stub Supervisor.search_sigils with params: {params}")
        return [{"sigil": "stub_search_result", "content": "found by search"}]


# --- RAG/Processor Stubs (Replace with actual imports/definitions) ---
class StandardRAGComponent:  # Placeholder for self.processor.standard_rag
    def __init__(self, supervisor_connector, blt_encoder):
        self._loaded_sigils: List[Dict[str, Any]] = [
            {
                "sigil": "ExampleSigil1",
                "tag": "example",
                "relationships": ["related_to_2"],
                "_last_modified": time.time() - 86400 * 10,
                "_similarity_score": 0.7,
            },
            {
                "sigil": "ExampleSigil2",
                "tag": "test",
                "relationships": {"type1": "ExampleSigil1"},
                "_last_modified": time.time() - 86400 * 5,
                "_similarity_score": 0.6,
            },
        ]
        self._sigil_cache: Dict[str, Any] = {}
        self.supervisor_connector = supervisor_connector
        self.blt_encoder = blt_encoder
        logger.info("Stub StandardRAGComponent initialized.")

    def load_all_sigils(self, force_reload: bool = False) -> List[Dict[str, Any]]:
        logger.debug("Stub StandardRAGComponent.load_all_sigils called.")
        # Simulate loading or returning cached sigils
        if force_reload or not self._loaded_sigils:
            # In a real scenario, this would fetch from a DB or files via supervisor_connector
            self._loaded_sigils = [
                {
                    "sigil": "FileSystemSigil",
                    "content": "Loaded from file system simulation",
                    "tag": "system",
                    "tags": ["core", "data"],
                    "principle": "Accessibility",
                    "usage": {"description": "Represents core data objects."},
                    "relationships": {"linkedTo": "AnotherSigil"},
                    "_source_file": "data/core/file_system_sigil.json",
                    "_last_modified": time.time() - 3600 * 24 * 5,  # 5 days ago
                    "_similarity_score": 0.0,  # Initial score before matching
                },
                {
                    "sigil": "RecentSigil",
                    "content": "A recently modified sigil",
                    "tag": "recent",
                    "tags": ["important"],
                    "principle": "Timeliness",
                    "usage": {"description": "Represents time-sensitive information."},
                    "relationships": {},
                    "_source_file": "data/recent/recent_sigil.yaml",
                    "_last_modified": time.time() - 3600 * 2,  # 2 hours ago
                    "_similarity_score": 0.0,
                },
            ]
            logger.info(f"Simulated loading {len(self._loaded_sigils)} sigils.")
        return self._loaded_sigils

    def create_rag_context(
        self, query: str, num_sigils: int = 5, **kwargs
    ) -> Tuple[str, List[Dict[str, Any]]]:
        logger.debug(
            f"Stub StandardRAGComponent.create_rag_context for query: {query[:30]}..."
        )
        # Dummy RAG: return first N loaded sigils with scores
        all_sigils = self.load_all_sigils()
        # Simulate scoring based on query (very naively)
        for s in all_sigils:
            s["_similarity_score"] = (
                0.5 + (hash(s["sigil"] + query) % 50) / 100.0
            )  # pseudo-random score
            s["_similarity_explanation"] = (
                f"Matched '{query[:10]}' due to keyword overlap (simulated)."
            )

        # Sort by score and take top N
        matched_sigils = sorted(
            all_sigils, key=lambda x: x.get("_similarity_score", 0.0), reverse=True
        )[:num_sigils]

        context_parts = [
            f"Sigil: {s['sigil']} (Score: {s['_similarity_score']:.2f})"
            for s in matched_sigils
        ]
        return "\n".join(context_parts), matched_sigils


# --- Core Hybrid Logic ---
class EntropyRouter:
    """Routes inputs based on their dynamically calculated entropy level."""

    def __init__(self, config: HybridMiddlewareConfig):
        self.config = config
        self.patch_encoder = SigilPatchEncoder()
        logger.info(
            f"EntropyRouter initialized with threshold: {self.config.entropy_threshold}, fallback: {self.config.entropy_router_fallback}"
        )

    def route(self, text: str) -> tuple[str, list[str] | None, list[float]]:
        if not text:
            logger.warning("Empty text received for routing. Using fallback.")
            return self.config.entropy_router_fallback, None, [0.5]

        try:
            patches_content, entropy_scores = self.patch_encoder.analyze_entropy(text)

            if not entropy_scores:
                logger.warning(
                    f"Entropy calculation returned no scores for text: '{text[:50]}...'. Applying heuristic."
                )
                # Apply heuristic as in original
                if (
                    any(c in text for c in ["<", ">", "{", "}", "[", "]"])
                    and len(text) < 200
                ):
                    avg_entropy = 0.15
                elif len(text) < 50 and " " not in text:
                    avg_entropy = 0.2
                else:
                    avg_entropy = 0.75
                entropy_scores = [avg_entropy]
                # Ensure patches_content is a list of strings
                patches_content = patches_content or [text]
                if not all(isinstance(p, str) for p in patches_content):
                    patches_content = [str(p) for p in patches_content]

            avg_entropy = (
                sum(entropy_scores) / len(entropy_scores) if entropy_scores else 0.5
            )
            logger.info(
                f"Text avg_entropy: {avg_entropy:.4f} (threshold: {self.config.entropy_threshold}) for query: '{text[:30]}...'"
            )

            # Ensure patches_content contains strings
            valid_patches: list[str] | None = None
            if patches_content:
                valid_patches = [str(p) for p in patches_content]

            if avg_entropy < self.config.entropy_threshold:
                return "patch_based", valid_patches, entropy_scores
            else:
                return (
                    "token_based",
                    None,
                    entropy_scores,
                )  # No patches needed for token_based
        except Exception as e:
            logger.error(
                f"Entropy calculation/routing failed: {e}. Using fallback path: {self.config.entropy_router_fallback}",
                exc_info=True,
            )
            return self.config.entropy_router_fallback, None, [0.5]



class HybridProcessor:  # Placeholder
    def __init__(
        self,
        config: HybridMiddlewareConfig,
        supervisor_connector: BaseSupervisorConnector,
        blt_encoder: BaseBLTEncoder,
    ):
        self.config = config
        self.router = EntropyRouter(config)
        # standard_rag should be an instance of a RAG component.
        self.standard_rag = StandardRAGComponent(supervisor_connector, blt_encoder)
        # blt_enhanced_rag could be another RAG component or a wrapper.
        self.blt_enhanced_rag = StandardRAGComponent(
            supervisor_connector, blt_encoder
        )  # Using same for simplicity
        logger.info("Stub HybridProcessor initialized.")

    def get_rag_context_and_route(
        self, query: str, num_sigils: int = 5, **kwargs
    ) -> Tuple[str, List[Dict[str, Any]], str]:
        route_type, _, _ = self.router.route(query)
        logger.debug(
            f"HybridProcessor determined route: {route_type} for query: {query[:30]}..."
        )

        # Use the appropriate RAG component based on routing or a combined strategy
        # This is a simplification; real HybridProcessor would be more complex
        if route_type == "patch_based" and hasattr(self, "blt_enhanced_rag"):
            rag_component = self.blt_enhanced_rag
            logger.debug("Using BLT Enhanced RAG (stub).")
        else:
            rag_component = self.standard_rag
            logger.debug("Using Standard RAG (stub).")

        context_str, sigils_list = rag_component.create_rag_context(
            query, num_sigils, **kwargs
        )
        return context_str, sigils_list, route_type


class BLTHybridMiddleware:  # Placeholder for the one used internally by ARCTaskProcessingMiddleware
    def __init__(self, config: HybridMiddlewareConfig):
        self.config = config
        logger.info("Stub BLTHybridMiddleware initialized.")

    def process(
        self, text: str, num_sigils: int = 5
    ) -> Tuple[str, List[Dict[str, Any]], str]:
        logger.debug(f"Stub BLTHybridMiddleware.process for text: {text[:30]}...")
        # Simulate processing, return some dummy sigils and route type
        dummy_sigils = [
            {"sigil": f"blt_processed_sigil_{i + 1}", "content": "Processed by BLT"}
            for i in range(num_sigils)
        ]
        return f"Enriched: {text}", dummy_sigils, "blt_route"


class DynamicExecutionBudgeter:  # Placeholder
    def __init__(self):
        logger.info("Stub DynamicExecutionBudgeter initialized.")

    def allocate_budget(
        self, route_method: str, avg_entropy: float, query_length: int
    ) -> float:
        budget = 100.0  # dummy budget
        logger.debug(
            f"Stub DynamicExecutionBudgeter.allocate_budget: {budget} for route {route_method}"
        )
        return budget


# --- Middleware Implementation 1: ARCTaskProcessingMiddleware ---
class ARCTaskProcessingMiddleware(BaseHybridMiddleware):
    """
    Processes ARC tasks using RAG and LLMs, potentially enhanced by MessageContextEnhancementMiddleware.
    """

    def __init__(
        self,
        config: Dict[str, Any],  # Specific config for ARC
        blt_encoder_instance: BaseBLTEncoder,
        supervisor_connector: BaseSupervisorConnector,
        context_enhancer: Optional[
            "MessageContextEnhancementMiddleware"
        ] = None,  # For enhanced RAG
        global_hybrid_config: Optional[
            HybridMiddlewareConfig
        ] = None,  # For internal BLT or fallback
    ):
        self.config = config
        self.blt_encoder = blt_encoder_instance
        self.supervisor_connector = supervisor_connector
        self.context_enhancer = context_enhancer  # Use this for primary RAG

        self.rag_enabled = self.config.get("rag_enabled", True)
        self.max_rag_results = self.config.get("max_rag_results", 5)
        self.llm_model = self.config.get("llm_model", "voxsigil-arc-optimized-7b")
        self.default_temperature = self.config.get("default_temperature", 0.2)
        self.max_tokens_solution = self.config.get("max_tokens_solution", 1024)
        self.prompt_template_sigil_ref = self.config.get(
            "prompt_template_sigil_ref", "Sigil:ARC_Prompt_Template_V2.3"
        )
        self.performance_evaluation_method = self.config.get(
            "performance_evaluation_method", "rule_based_validator"
        )
        self.solution_format = self.config.get("solution_format", "json_arc_compliant")

        # Initialize internal BLT hybrid middleware for fallback or if context_enhancer not provided
        # This config might be different from the main context_enhancer's config
        blt_internal_config = global_hybrid_config or HybridMiddlewareConfig(
            entropy_threshold=self.config.get(
                "blt_entropy_threshold", 0.3
            ),  # Potentially different thresholds
            blt_hybrid_weight=self.config.get("blt_hybrid_weight_internal", 0.6),
            log_level=self.config.get("log_level", "INFO"),
        )
        try:
            self.blt_middleware_internal = BLTHybridMiddleware(blt_internal_config)
            logger.info(
                "ARCTask: Successfully initialized internal BLTHybridMiddleware for fallback."
            )
        except Exception as e:
            logger.error(
                f"ARCTask: Error initializing internal BLTHybridMiddleware: {e}",
                exc_info=True,
            )
            self.blt_middleware_internal = None

        logger.info(
            f"ARCTaskProcessingMiddleware initialized with model '{self.llm_model}', "
            f"RAG enabled: {self.rag_enabled}, Context Enhancer: {'available' if self.context_enhancer else 'unavailable'}, "
            f"Internal BLT: {'available' if self.blt_middleware_internal else 'unavailable'}."
        )

    def process_arc_task(
        self,
        input_data_sigil_ref: str,
        task_sigil_ref: str,
        task_parameters: Dict[str, Any],
    ) -> Tuple[str, str]:
        logger.info(
            f"ARCTask: Processing ARC task '{task_sigil_ref}' with input '{input_data_sigil_ref}'"
        )
        try:
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
                )

            relevant_examples: List[Dict[str, Any]] = []
            task_description = task_data.get("description", "")
            if (
                not task_description
                and input_data
                and "grid" in input_data
                and isinstance(input_data["grid"], list)
                and input_data["grid"]
            ):
                task_description = f"ARC task with grid size {len(input_data['grid'])}x{len(input_data['grid'][0])}"
            elif not task_description:
                task_description = f"Solve ARC task: {task_sigil_ref}"

            if self.rag_enabled and task_description:
                rag_method_used = "none"
                if self.context_enhancer:
                    logger.info(
                        f"ARCTask: Using ContextEnhancer for RAG on: {task_description[:50]}..."
                    )
                    # Use enhanced_rag_process which returns (formatted_context_str, sigils_list)
                    _, relevant_examples = self.context_enhancer.enhanced_rag_process(
                        task_description, num_sigils=self.max_rag_results
                    )
                    rag_method_used = "ContextEnhancer"
                elif self.blt_middleware_internal:
                    logger.info(
                        f"ARCTask: Using internal BLTHybridMiddleware for RAG on: {task_description[:50]}..."
                    )
                    _, relevant_examples, _ = self.blt_middleware_internal.process(
                        task_description, num_sigils=self.max_rag_results
                    )
                    rag_method_used = "InternalBLT"
                else:  # Fallback to basic RAG
                    logger.info(
                        f"ARCTask: Using basic _perform_rag on: {task_description[:50]}..."
                    )
                    relevant_examples = self._perform_rag(
                        input_data, task_data, task_parameters
                    )
                    rag_method_used = "BasicPerformRAG"
                logger.debug(
                    f"ARCTask: RAG ({rag_method_used}) retrieved {len(relevant_examples)} examples."
                )

            prompt_template = self._get_prompt_template()
            solution_content = self._generate_solution(
                input_data,
                task_data,
                relevant_examples,
                prompt_template,
                task_parameters,
            )
            solution_sigil_ref = (
                f"SigilRef:Solution_{Path(task_sigil_ref).name}_{time.time_ns()}"
            )
            self.supervisor_connector.create_sigil(
                solution_sigil_ref, solution_content, "ARC_Solution"
            )

            perf_metrics = self._evaluate_performance(
                input_data, task_data, solution_content, task_parameters
            )
            perf_metric_sigil_ref = (
                f"SigilRef:PerfMetric_{Path(task_sigil_ref).name}_{time.time_ns()}"
            )
            self.supervisor_connector.create_sigil(
                perf_metric_sigil_ref, perf_metrics, "PerformanceMetric"
            )

            logger.info(
                f"ARCTask: Task '{task_sigil_ref}' processed. Sol: '{solution_sigil_ref}', Perf: {perf_metrics.get('achieved_performance', 0.0):.3f}"
            )
            return solution_sigil_ref, perf_metric_sigil_ref

        except Exception as e:
            logger.error(
                f"ARCTask: Error processing task '{task_sigil_ref}': {e}", exc_info=True
            )
            # Create failure sigils
            # (Code for failure sigils is omitted for brevity but would be similar to original)
            failure_solution_sigil_ref = (
                f"SigilRef:Solution_Failed_{Path(task_sigil_ref).name}_{time.time_ns()}"
            )
            failure_metric_sigil_ref = f"SigilRef:PerfMetric_Failed_{Path(task_sigil_ref).name}_{time.time_ns()}"
            # self.supervisor_connector.create_sigil(...)
            return failure_solution_sigil_ref, failure_metric_sigil_ref

    def _perform_rag(
        self,
        input_data: Dict[str, Any],
        task_data: Dict[str, Any],
        task_parameters: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        task_description = task_data.get("description", "")
        if (
            not task_description
            and "grid" in input_data
            and isinstance(input_data["grid"], list)
            and input_data["grid"]
        ):
            task_description = f"ARC task with grid size {len(input_data['grid'])}x{len(input_data['grid'][0])}"

        try:
            embedding = self.blt_encoder.encode(task_description, task_type="arc_task")
            example_query = {
                "embedding": embedding,
                "max_results": self.max_rag_results,
                "min_similarity": 0.7,
                "collection": "arc_examples",
            }
            similar_examples = self.supervisor_connector.find_similar_examples(
                example_query
            )
            return similar_examples or []
        except Exception as e:
            logger.warning(f"ARCTask: Error during basic RAG retrieval: {e}")
            return []

    def _get_prompt_template(self) -> Dict[str, Any]:
        try:
            template = self.supervisor_connector.get_sigil_content_as_dict(
                self.prompt_template_sigil_ref
            )
            if not template:
                logger.warning(
                    f"ARCTask: Failed to retrieve prompt template '{self.prompt_template_sigil_ref}'. Using default."
                )
                return {
                    "template": "Solve the following ARC task: {{task_description}} with input {{input_data}}. Examples: {{examples}}. Format: {{format}}"
                }
            return template
        except Exception as e:
            logger.warning(
                f"ARCTask: Error retrieving prompt template: {e}. Using default."
            )
            return {
                "template": "Solve the following ARC task: {{task_description}} with input {{input_data}}. Examples: {{examples}}. Format: {{format}}"
            }

    def _generate_solution(
        self,
        input_data: Dict[str, Any],
        task_data: Dict[str, Any],
        relevant_examples: List[Dict[str, Any]],
        prompt_template: Dict[str, Any],
        task_parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        temperature = task_parameters.get(
            "effective_temperature", self.default_temperature
        )
        max_attempts = task_parameters.get("max_solution_attempts", 1)

        examples_text = ""
        if relevant_examples:
            examples_text = "Here are some examples of similar tasks:\n"
            for i, example in enumerate(relevant_examples[:3]):
                examples_text += f"Example {i + 1}:\nInput: {json.dumps(example.get('input', {}))}\nSolution: {json.dumps(example.get('solution', {}))}\n\n"

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
        )

        solution = None
        for attempt in range(max_attempts):
            try:
                llm_params = {
                    "model": self.llm_model,
                    "prompt": prompt,
                    "temperature": temperature + (attempt * 0.1),
                    "max_tokens": self.max_tokens_solution,
                    "format": self.solution_format,
                }
                # Enhanced prompt via BLT only if context_enhancer wasn't used for RAG (avoid double processing)
                # AND internal BLT is available, AND it's the first attempt.
                if (
                    not self.context_enhancer  # If context_enhancer was used, assume its RAG output is already good
                    and self.blt_middleware_internal
                    and attempt == 0
                ):
                    try:
                        enriched_prompt, _, _ = self.blt_middleware_internal.process(
                            prompt
                        )  # Assuming it returns enriched text
                        logger.info(
                            "ARCTask: Enhanced prompt using internal BLT middleware."
                        )
                        llm_params["prompt"] = enriched_prompt
                    except Exception as e:
                        logger.warning(
                            f"ARCTask: Error enhancing prompt with internal BLT: {e}"
                        )

                response = self.supervisor_connector.call_llm(llm_params)
                if isinstance(response, str):
                    solution = json.loads(response)
                else:
                    solution = response
                if self._validate_solution_format(solution, input_data):
                    break
            except json.JSONDecodeError:
                logger.warning(
                    f"ARCTask: Failed to parse LLM response as JSON (attempt {attempt + 1}/{max_attempts})"
                )
                solution = {"raw_response": response, "parsing_error": True}
            except Exception as e:
                logger.warning(
                    f"ARCTask: Error generating solution (attempt {attempt + 1}/{max_attempts}): {e}"
                )
                solution = {"error": str(e)}

        solution_with_metadata = solution or {"error": "Failed to generate solution"}
        # ... add metadata as in original ...
        return solution_with_metadata

    def _validate_solution_format(
        self, solution: Dict[str, Any], input_data: Dict[str, Any]
    ) -> bool:
        if self.solution_format == "json_arc_compliant":
            return "grid" in solution or "output_grid" in solution  # Example validation
        return True  # Default to true if no specific format check

    def _evaluate_performance(
        self,
        input_data: Dict[str, Any],
        task_data: Dict[str, Any],
        solution_content: Dict[str, Any],
        task_parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        # ... evaluation logic as in original, simplified for brevity ...
        ground_truth = task_data.get("output", task_data.get("expected_output"))
        achieved_performance = 0.0
        if "error" in solution_content or "parsing_error" in solution_content:
            achieved_performance = 0.0
        elif ground_truth:
            # Simple check
            solution_grid = solution_content.get(
                "grid", solution_content.get("output_grid")
            )
            if solution_grid == ground_truth:  # Simplified comparison
                achieved_performance = 1.0
            else:
                achieved_performance = 0.3  # Partial if grids mismatch
        else:
            achieved_performance = 0.5  # Default if no ground truth
        return {
            "task_sigil_ref": task_data.get("sigil_ref", ""),
            "achieved_performance": achieved_performance,
            # ... other metrics ...
        }

    def find_similar_examples(self, example_data: Any) -> Optional[Dict[str, Any]]:
        """Finds similar examples. If context_enhancer is available, uses its RAG. Else, uses BLTHybrid or basic search."""
        logger.info(
            f"ARCTask: Finding similar examples for data: {str(example_data)[:100]}..."
        )
        example_text = (
            json.dumps(example_data)
            if not isinstance(example_data, str)
            else example_data
        )

        if self.context_enhancer:
            logger.debug("ARCTask: Using ContextEnhancer for find_similar_examples.")
            _, sigils = self.context_enhancer.enhanced_rag_process(
                example_text, num_sigils=self.max_rag_results
            )
            return {
                "query": example_text,
                "found_examples": len(sigils),
                "processing_route": "context_enhancer_rag",
                "examples": sigils,
                "source": "ContextEnhancementMiddleware",
            }
        elif self.blt_middleware_internal:
            logger.debug(
                "ARCTask: Using internal BLTHybridMiddleware for find_similar_examples."
            )
            _, sigils, route_type = self.blt_middleware_internal.process(
                example_text, num_sigils=self.max_rag_results
            )
            return {
                "query": example_text,
                "found_examples": len(sigils),
                "processing_route": route_type,
                "examples": sigils,
                "source": "internal_blt_hybrid_middleware",
            }
        else:  # Fallback to basic supervisor search
            logger.debug(
                "ARCTask: Using basic supervisor search for find_similar_examples."
            )
            similar_sigils = self.supervisor_connector.search_sigils(
                {"text": example_text[:100], "max_results": self.max_rag_results}
            )
            return {
                "query": example_text,
                "found_examples": len(similar_sigils),
                "processing_route": "basic_supervisor_search",
                "examples": similar_sigils,
                "source": "basic_sigil_search",
            }


# --- Middleware Implementation 2: MessageContextEnhancementMiddleware ---
class MessageContextEnhancementMiddleware:
    """Middleware to enhance incoming messages with RAG context."""

    def __init__(
        self,
        config: HybridMiddlewareConfig,
        supervisor_connector: BaseSupervisorConnector,
        blt_encoder: BaseBLTEncoder,
    ):
        self.config = config
        logger.info("MessageContextEnhancer: Initializing with config.")
        # Pass supervisor_connector and blt_encoder to HybridProcessor if it needs them
        self.processor = HybridProcessor(config, supervisor_connector, blt_encoder)
        self.budgeter = DynamicExecutionBudgeter()
        self._context_cache: dict[
            str, tuple[str, list[dict[str, Any]], str, float]
        ] = {}
        self._request_counter = 0
        self._processing_times: list[float] = []
        self.sigil_schema = DEFAULT_SIGIL_SCHEMA  # Added schema for validation

        # These attributes were part of _initialize_voxsigil_components in the original
        # self.voxsigil_rag_component is now self.processor.standard_rag (or blt_enhanced_rag)
        self.conversation_history: list[Any] = []
        self.selected_sigils_history: dict[Any, Any] = {}
        self.turn_counter = 0
        self.rag_off_keywords = ["@@norag@@", "norag"]
        self.min_prompt_len_for_rag = 5
        self._rag_cache: dict[Any, Any] = {}

        # Initializing components like normalization
        self._normalize_all_sigil_relationships_on_init()

    def process_text_for_rag(
        self, text: str, num_sigils: int = 5, **kwargs
    ) -> tuple[str, list[dict[str, Any]], str]:
        """Alias for processor's method to provide a simple processing interface for RAG."""
        return self.processor.get_rag_context_and_route(
            text, num_sigils=num_sigils, **kwargs
        )

    def _normalize_all_sigil_relationships_on_init(self):
        """Normalizes relationships for all sigils known to the RAG components."""
        rag_components_to_normalize = []
        if hasattr(self.processor, "standard_rag") and self.processor.standard_rag:
            rag_components_to_normalize.append(self.processor.standard_rag)
        if (
            hasattr(self.processor, "blt_enhanced_rag")
            and self.processor.blt_enhanced_rag
            and self.processor.blt_enhanced_rag is not self.processor.standard_rag
        ):  # Avoid double-processing if same obj
            rag_components_to_normalize.append(self.processor.blt_enhanced_rag)

        if not rag_components_to_normalize:
            logger.warning(
                "MessageContextEnhancer: No RAG components found for relationship normalization."
            )
            return

        total_normalized_count = 0
        for rag_component in rag_components_to_normalize:
            if not hasattr(rag_component, "_loaded_sigils"):
                if hasattr(rag_component, "load_all_sigils"):
                    rag_component.load_all_sigils(
                        force_reload=False
                    )  # Ensure sigils are loaded
                else:
                    logger.warning(
                        "MessageContextEnhancer: RAG component cannot load sigils. Skipping normalization for it."
                    )
                    continue

            if not rag_component._loaded_sigils:
                logger.warning(
                    "MessageContextEnhancer: No sigils loaded in RAG component to normalize."
                )
                continue

            current_normalized_count = 0
            newly_loaded_sigils = []
            for sigil in rag_component._loaded_sigils:
                # Validate before normalization (optional, but good practice)
                # self._validate_sigil_data(sigil, sigil.get('_source_file'))

                original_relationships = sigil.get("relationships")
                # Ensure working on a copy if _normalize_single_sigil_relationships modifies in-place
                # or if the sigil dicts are shared. Here, we assume it returns a new/modified dict.
                modified_sigil = self._normalize_single_sigil_relationships(
                    sigil.copy()
                )  # Operate on a copy

                if (
                    modified_sigil.get("relationships") != original_relationships
                ):  # Basic check for change
                    current_normalized_count += 1
                newly_loaded_sigils.append(modified_sigil)

            if current_normalized_count > 0:
                rag_component._loaded_sigils = (
                    newly_loaded_sigils  # Update the component's sigil list
                )
                if hasattr(
                    rag_component, "_sigil_cache"
                ):  # Clear any caches if content changed
                    rag_component._sigil_cache = {}
                logger.info(
                    f"MessageContextEnhancer: Normalized relationships for {current_normalized_count} sigils in a RAG component."
                )
            total_normalized_count += current_normalized_count

        if total_normalized_count > 0:
            logger.info(
                f"MessageContextEnhancer: Total {total_normalized_count} sigil relationships normalized across RAG components."
            )

    def _normalize_single_sigil_relationships(
        self, sigil: dict[str, Any]
    ) -> dict[str, Any]:
        if "relationships" not in sigil:
            return sigil  # No relationships to normalize

        current_relationships = sigil["relationships"]

        # If it's already a dict, assume it's in the desired format (or needs deeper checks)
        if isinstance(current_relationships, dict):
            # Optionally, could further validate/normalize keys/values within the dict here
            return sigil

        # If not a dict, attempt to convert
        new_rels: dict[str, Any] = {}
        if isinstance(current_relationships, list):
            for i, rel_item in enumerate(current_relationships):
                if isinstance(rel_item, str):
                    # Try to parse "type:target" or use a default relation type
                    if ":" in rel_item:
                        parts = rel_item.split(":", 1)
                        rel_type, rel_target = parts[0].strip(), parts[1].strip()
                        # Handle multiple targets for the same type (e.g., "related: A, B")
                        if rel_type in new_rels:
                            if not isinstance(new_rels[rel_type], list):
                                new_rels[rel_type] = [new_rels[rel_type]]
                            new_rels[rel_type].append(rel_target)
                        else:
                            new_rels[rel_type] = rel_target
                    else:
                        new_rels[f"relation_{i}"] = (
                            rel_item  # Fallback to indexed relation
                        )
                elif isinstance(rel_item, dict) and len(rel_item) == 1:
                    key, value = next(iter(rel_item.items()))
                    # Similar handling for multiple targets for the same key from list of dicts
                    if key in new_rels:
                        if not isinstance(new_rels[key], list):
                            new_rels[key] = [new_rels[key]]
                        new_rels[key].append(value)
                    else:
                        new_rels[key] = value
                else:
                    new_rels[f"unknown_relation_{i}"] = (
                        rel_item  # Store as is, needs review
                    )
        elif isinstance(current_relationships, str):  # A single string relationship
            if ":" in current_relationships:
                parts = current_relationships.split(":", 1)
                new_rels[parts[0].strip()] = parts[1].strip()
            else:
                new_rels["default_relation"] = current_relationships
        else:
            # Non-list, non-dict, non-string - wrap it
            new_rels["unstructured_relation"] = current_relationships
            logger.warning(
                f"MessageContextEnhancer: Encountered unexpected relationship type for sigil '{sigil.get('sigil', 'N/A')}': {type(current_relationships)}. Wrapped."
            )

        sigil["relationships"] = new_rels
        logger.debug(
            f"MessageContextEnhancer: Normalized relationships for sigil '{sigil.get('sigil', 'N/A')}'"
        )
        return sigil

    def format_sigil_for_context(
        self,
        sigil: dict[str, Any],
        detail_level: str = "standard",
        include_explanations: bool = False,
    ) -> str:
        # (Implementation from original, slightly adjusted for clarity if needed)
        output = []
        if "sigil" in sigil:
            output.append(f'Sigil: "{sigil["sigil"]}"')

        all_tags = []
        if sigil.get("tag"):
            all_tags.append(str(sigil["tag"]))
        if isinstance(sigil.get("tags"), list):
            all_tags.extend(str(t) for t in sigil["tags"] if str(t) not in all_tags)
        elif isinstance(sigil.get("tags"), str) and sigil["tags"] not in all_tags:
            all_tags.append(sigil["tags"])
        if all_tags:
            output.append("Tags: " + ", ".join(f'"{tag}"' for tag in all_tags))

        if "principle" in sigil:
            output.append(f'Principle: "{sigil["principle"]}"')

        if detail_level.lower() == "summary":
            return "\n".join(output)
        if detail_level.lower() == "minimal":  # Added minimal level
            return f'Sigil: "{sigil.get("sigil", "N/A")}" (Tags: {", ".join(all_tags[:2])}{"..." if len(all_tags) > 2 else ""})'

        usage = sigil.get("usage", {})
        if isinstance(usage, dict):
            if "description" in usage:
                output.append(f'Usage: "{usage["description"]}"')
            if "examples" in usage and usage["examples"]:
                ex_str = (
                    f'"{usage["examples"][0]}"'
                    if isinstance(usage["examples"], list)
                    else f'"{usage["examples"]}"'
                )
                output.append(f"Example: {ex_str}")

        if sigil.get("_source_file"):
            output.append(f"Source File: {Path(sigil['_source_file']).name}")
        if include_explanations and sigil.get("_similarity_explanation"):
            output.append(f"Match Information: {sigil['_similarity_explanation']}")
        if include_explanations and sigil.get("_similarity_score") is not None:
            output.append(f"Relevance Score: {sigil['_similarity_score']:.3f}")

        if detail_level.lower() in ("detailed", "full"):
            if isinstance(sigil.get("relationships"), dict) and sigil["relationships"]:
                # Prettier relationship formatting
                rel_strs = []
                for rel_type, rel_target in sigil["relationships"].items():
                    if isinstance(rel_target, list):
                        rel_strs.append(
                            f"{rel_type}: [{', '.join(map(str, rel_target))}]"
                        )
                    else:
                        rel_strs.append(f"{rel_type}: {rel_target}")
                output.append(f"Relationships: {'; '.join(rel_strs)}")

            if sigil.get("scaffolds"):
                output.append(f"Scaffolds: {sigil['scaffolds']}")
            # Add more fields for 'full' if necessary, e.g., full content
            if detail_level.lower() == "full" and "content" in sigil:
                output.append(f"Content Preview: {str(sigil['content'])[:100]}...")

        return "\n".join(output)

    def format_sigils_for_context(
        self,
        sigils: list[dict[str, Any]],
        detail_level: str = "standard",
        include_explanations: bool = False,
    ) -> str:
        # (Implementation from original)
        if not sigils:
            return ""
        output_sections = [
            f"--- VoxSigil Context ({len(sigils)} sigils, Detail: {detail_level}) ---"
        ]
        for idx, sigil in enumerate(sigils, 1):
            sigil_text = self.format_sigil_for_context(
                sigil, detail_level, include_explanations
            )
            output_sections.append(f"---\nSIGIL {idx}:\n{sigil_text}")
        output_sections.append("--- End VoxSigil Context ---")
        return "\n\n".join(output_sections)

    def _get_cache_key(self, query: str) -> str:
        # (Implementation from original)
        normalized_query = " ".join(query.lower().strip().split())
        return (
            hashlib.sha256(normalized_query.encode()).hexdigest()
            if len(normalized_query) > 256
            else normalized_query
        )

    def _clean_expired_cache_entries(self) -> None:
        # (Implementation from original)
        current_time = time.monotonic()
        expired_keys = [
            key
            for key, (_, _, _, timestamp) in self._context_cache.items()
            if current_time - timestamp > self.config.cache_ttl_seconds
        ]
        for key in expired_keys:
            del self._context_cache[key]
        if expired_keys:
            logger.info(
                f"MessageContextEnhancer: Cleaned {len(expired_keys)} expired cache entries."
            )

    def _extract_query_from_messages(
        self, messages: list[dict[str, Any]]
    ) -> str | None:
        # (Implementation from original)
        if not messages:
            return None
        for msg in reversed(messages):
            if (
                isinstance(msg, dict)
                and msg.get("role") == "user"
                and isinstance(msg.get("content"), str)
            ):
                return msg["content"]
        last_message = messages[-1]  # Fallback, might not be 'user'
        if isinstance(last_message, dict) and isinstance(
            last_message.get("content"), str
        ):
            return last_message["content"]
        logger.warning(
            f"MessageContextEnhancer: Could not extract valid query from messages: {messages}"
        )
        return None

    def _enhance_messages_with_context(
        self, messages: list[dict[str, Any]], context: str
    ) -> list[dict[str, Any]]:
        # (Implementation from original)
        if not context:
            return messages
        enhanced_messages = [
            msg.copy() for msg in messages
        ]  # Deep copy if messages contain mutable objects

        system_message_content = (
            f"You are a helpful assistant. Use the following VoxSigil context to answer the user's query accurately. "
            f"Prioritize information from the VoxSigil context. If the context is not relevant or insufficient, "
            f"you may use your general knowledge. Clearly indicate when you are using information primarily from the context.\n\n"
            f"--- VOXSIGIL CONTEXT START ---\n{context}\n--- VOXSIGIL CONTEXT END ---"
        )

        # Find existing system message or prepend a new one
        system_message_found = False
        for msg in enhanced_messages:
            if msg.get("role") == "system":
                msg["content"] = (
                    f"{system_message_content}\n\nOriginal System Instructions:\n{msg.get('content', '')}"
                )
                system_message_found = True
                break
        if not system_message_found:
            enhanced_messages.insert(
                0, {"role": "system", "content": system_message_content}
            )

        logger.debug("MessageContextEnhancer: Messages enhanced with RAG context.")
        return enhanced_messages

    def __call__(self, model_input: dict[str, Any]) -> dict[str, Any]:
        """Processes model_input (typically for an LLM) to enhance messages with RAG context."""
        self._request_counter += 1
        start_time_total = time.monotonic()

        messages = model_input.get("messages")
        if not isinstance(messages, list) or not messages:
            logger.warning(
                "MessageContextEnhancer: No messages/invalid format in model_input. Skipping."
            )
            return model_input

        query = self._extract_query_from_messages(messages)
        if not query:
            logger.warning("MessageContextEnhancer: No query extractable. Skipping.")
            return model_input

        # Skip RAG for very short queries or if RAG-off keyword is present
        if len(query) < self.min_prompt_len_for_rag or any(
            keyword in query.lower() for keyword in self.rag_off_keywords
        ):
            logger.info(
                f"MessageContextEnhancer: Skipping RAG for query: '{query[:50]}...' (short or keyword)."
            )
            model_input.setdefault("voxsigil_metadata", {}).update(
                {
                    "rag_skipped": True,
                    "reason": "Query too short or RAG-off keyword present.",
                }
            )
            return model_input

        if self._request_counter % 50 == 0:
            self._clean_expired_cache_entries()

        cache_key = self._get_cache_key(query)
        cached_entry = self._context_cache.get(cache_key)
        current_monotonic_time = time.monotonic()

        context_str, sigils_list, route_method, cache_hit = "", [], "unknown", False

        if cached_entry:
            cached_context_str, cached_sigils_list, cached_route_method, timestamp = (
                cached_entry
            )
            if current_monotonic_time - timestamp <= self.config.cache_ttl_seconds:
                logger.info(
                    f"MessageContextEnhancer: Cache HIT for query (key: {cache_key[:30]}...). Route: {cached_route_method}"
                )
                context_str, sigils_list, route_method, cache_hit = (
                    cached_context_str,
                    cached_sigils_list,
                    cached_route_method,
                    True,
                )
                # Refresh timestamp
                self._context_cache[cache_key] = (
                    context_str,
                    sigils_list,
                    route_method,
                    current_monotonic_time,
                )
            else:
                logger.info(
                    f"MessageContextEnhancer: Cache STALE (key: {cache_key[:30]}...). Re-computing."
                )
                del self._context_cache[cache_key]

        if not cache_hit:
            logger.info(
                f"MessageContextEnhancer: Cache MISS (key: {cache_key[:30]}...). Processing with enhanced_rag_process."
            )
            try:
                # Using the full enhanced_rag_process method
                context_str, sigils_list = self.enhanced_rag_process(
                    query,
                    num_sigils=self.config.max_rag_results,  # from main config
                    # Pass other relevant parameters for enhanced_rag_process from config if needed
                    # augment_query_flag=self.config.get("augment_query_flag", True), # Example
                    # enable_context_optimization=self.config.get("enable_context_optimization", True), # Example
                )
                # Determine route_method after the fact for logging, if not returned by enhanced_rag_process
                # Or enhanced_rag_process could return it. For now, using processor's router.
                determined_route, _, _ = self.processor.router.route(query)
                route_method = determined_route

                if context_str:  # Only cache if context was generated
                    self._context_cache[cache_key] = (
                        context_str,
                        sigils_list,
                        route_method,
                        current_monotonic_time,
                    )
                    logger.info(
                        f"MessageContextEnhancer: Processed and cached query. Route decision: {route_method}, Context length: {len(context_str)}"
                    )
                else:
                    logger.info(
                        f"MessageContextEnhancer: No context generated for query. Route decision: {route_method}."
                    )

            except Exception as e:
                logger.critical(
                    f"MessageContextEnhancer: Error during RAG for query '{query[:50]}...': {e}",
                    exc_info=True,
                )
                return model_input  # Passthrough on critical error

        # Budgeting based on the main query and routing for this request
        # Need to get entropy scores. processor.router.route returns (route_type, entropy_values, token_probabilities)
        _, entropy_scores_for_budget, _ = self.processor.router.route(query)
        avg_entropy_for_budget = (
            sum(entropy_scores_for_budget) / len(entropy_scores_for_budget)
            if entropy_scores_for_budget
            else 0.5  # Default entropy if none
        )
        budget = self.budgeter.allocate_budget(
            route_method, avg_entropy_for_budget, len(query)
        )

        enhanced_messages = self._enhance_messages_with_context(messages, context_str)

        total_processing_time = time.monotonic() - start_time_total
        self._processing_times.append(total_processing_time)

        log_metadata = {
            "request_id": hashlib.md5(query.encode()).hexdigest()[:8],
            "query_preview": query[:50] + "...",
            "route_method_used": route_method,
            "cache_hit": cache_hit,
            "context_length": len(context_str),
            "num_sigils_retrieved": len(sigils_list),
            "allocated_budget": round(budget, 2),
            "total_processing_time_ms": round(total_processing_time * 1000, 2),
            "avg_entropy_for_budget": round(avg_entropy_for_budget, 3),
        }
        logger.info(f"MessageContextEnhancer: Processed request: {log_metadata}")

        model_input["messages"] = enhanced_messages
        model_input.setdefault("voxsigil_metadata", {}).update(log_metadata)
        return model_input

    def get_stats(self) -> dict[str, Any]:
        # (Implementation from original)
        avg_proc_time = (
            sum(self._processing_times) / len(self._processing_times)
            if self._processing_times
            else 0
        )
        return {
            "total_requests_processed": self._request_counter,
            "cache_size": len(self._context_cache),
            "average_processing_time_seconds": round(avg_proc_time, 4),
        }

    def _augment_query(self, query: str) -> str:
        # (Implementation from original, ensure synonym_map is initialized or loaded)
        if not hasattr(self, "synonym_map") or not self.synonym_map:  # Check if empty
            self.synonym_map = {  # Default or load from config
                "ai": [
                    "artificial intelligence",
                    "ml",
                    "machine learning",
                    "deep learning",
                ],
                "llm": ["large language model", "language model"],
                "voxsigil": ["vox sigil", "sigil language", "sigil system"],
                "arc task": ["abstraction and reasoning corpus", "arc challenge"],
                # Add more relevant synonyms for your domain
            }
            logger.info(
                f"MessageContextEnhancer: Initialized synonym map with {len(self.synonym_map)} terms."
            )

        augmented_parts = [query]
        query_lower = query.lower()

        # Simple word-based synonym expansion for terms found in query
        words_in_query = set(query_lower.split())
        added_synonyms = set()

        for term, synonyms in self.synonym_map.items():
            if (
                term in words_in_query or term in query_lower
            ):  # check for multi-word terms too
                for syn in synonyms:
                    # Add synonym if it's not already part of the query (case-insensitive)
                    # and not already added to avoid excessive redundancy
                    if (
                        syn.lower() not in query_lower
                        and syn.lower() not in added_synonyms
                    ):
                        augmented_parts.append(syn)
                        added_synonyms.add(syn.lower())

        # More advanced: replace term with "term OR synonym1 OR synonym2" for search engines
        # For simple concatenation, this is fine.
        augmented_query = " ".join(augmented_parts)

        if augmented_query != query:
            logger.info(
                f"MessageContextEnhancer: Query augmented: '{query}' -> '{augmented_query}'"
            )
        return augmented_query

    def _validate_sigil_data(
        self, sigil_data: dict[str, Any], file_path: Optional[Union[str, Path]] = None
    ) -> bool:
        # (Implementation from original, ensure self.sigil_schema is available)
        if not hasattr(self, "sigil_schema") or not self.sigil_schema:
            logger.error(
                "MessageContextEnhancer: Sigil schema not available for validation."
            )
            return False  # Cannot validate without schema

        # Dynamic import of jsonschema if not already imported at module level
        try:
            from jsonschema import ValidationError, validate
        except ImportError:
            logger.error(
                "MessageContextEnhancer: jsonschema library not installed. Cannot validate sigils."
            )
            return False  # Cannot validate if library is missing

        try:
            validate(instance=sigil_data, schema=self.sigil_schema)
            return True
        except ValidationError as e:
            path_str = " -> ".join(map(str, e.path)) if e.path else "root"
            file_info = f" for {Path(file_path).name}" if file_path else ""
            logger.warning(
                f"MessageContextEnhancer: Schema validation failed{file_info}: {e.message} (at path: '{path_str}')"
            )
            return False
        except Exception as e_generic:  # Catch other errors during validation
            logger.error(
                f"MessageContextEnhancer: Generic error during schema validation for {file_path or 'sigil'}: {e_generic}",
                exc_info=True,
            )
            return False

    def _apply_recency_boost(
        self,
        sigils_with_scores: list[dict[str, Any]],
        recency_boost_factor: float = 0.05,
        recency_max_days: int = 90,
    ) -> list[dict[str, Any]]:
        # (Implementation from original)
        if not recency_boost_factor > 0:
            return sigils_with_scores

        current_time_utc_ts = datetime.now(timezone.utc).timestamp()
        recency_max_seconds = recency_max_days * 24 * 60 * 60
        boosted_sigils: list[dict[str, Any]] = []

        for s_data in sigils_with_scores:
            sigil_copy = s_data.copy()  # Work on a copy
            last_modified_ts = sigil_copy.get(
                "_last_modified"
            )  # Expecting Unix timestamp
            original_score = sigil_copy.get("_similarity_score", 0.0)

            if isinstance(last_modified_ts, (int, float)):
                age_seconds = current_time_utc_ts - last_modified_ts
                if 0 <= age_seconds < recency_max_seconds:  # Sigil is recent enough
                    # Linear decay for recency boost
                    boost = recency_boost_factor * (
                        1.0 - (age_seconds / recency_max_seconds)
                    )
                    new_score = min(1.0, original_score + boost)  # Cap score at 1.0
                    if new_score > original_score:  # Log if boost was effective
                        sigil_copy["_similarity_score"] = new_score
                        sigil_copy["_recency_boost_applied"] = round(boost, 4)
                        logger.debug(
                            f"MessageContextEnhancer: Applied recency boost {boost:.3f} to '{sigil_copy.get('sigil', 'N/A')}', new score: {new_score:.3f}"
                        )
            boosted_sigils.append(sigil_copy)
        return boosted_sigils

    def auto_fuse_related_sigils(
        self, base_sigils: list[dict[str, Any]], max_additional: int = 3
    ) -> list[dict[str, Any]]:
        # (Implementation from original - ensure self.processor.standard_rag (or chosen RAG component) is correct)
        if not base_sigils or max_additional <= 0:
            return base_sigils

        # Determine which RAG component holds the "master list" of sigils
        # Assuming standard_rag is the primary one, or make it configurable
        rag_component_for_all_sigils = self.processor.standard_rag
        if not rag_component_for_all_sigils or not hasattr(
            rag_component_for_all_sigils, "load_all_sigils"
        ):
            logger.warning(
                "MessageContextEnhancer: Cannot auto-fuse, RAG component with all sigils not available."
            )
            return base_sigils

        all_system_sigils = rag_component_for_all_sigils.load_all_sigils()
        if not all_system_sigils:
            logger.warning(
                "MessageContextEnhancer: No system sigils loaded for auto-fusion."
            )
            return base_sigils

        sigil_index_by_id = {s["sigil"]: s for s in all_system_sigils if "sigil" in s}
        current_sigil_ids = {s["sigil"] for s in base_sigils if "sigil" in s}
        fused_sigils_list = list(base_sigils)  # Start with a copy of base sigils
        added_count = 0

        # Iterate over a copy of the base_sigils list as we might modify fused_sigils_list
        for sigil_item in list(base_sigils):
            if added_count >= max_additional:
                break
            source_id = sigil_item.get("sigil")
            if not source_id:
                continue

            # Check explicit relationships (assuming normalized format)
            relationships = sigil_item.get("relationships")
            if isinstance(relationships, dict):
                for rel_type, rel_targets_val in relationships.items():
                    if added_count >= max_additional:
                        break
                    targets = (
                        rel_targets_val
                        if isinstance(rel_targets_val, list)
                        else [rel_targets_val]
                    )

                    for target_id_any_type in targets:
                        if added_count >= max_additional:
                            break
                        target_id = str(target_id_any_type)  # Ensure string ID

                        if (
                            target_id in sigil_index_by_id
                            and target_id not in current_sigil_ids
                        ):
                            related_s = sigil_index_by_id[
                                target_id
                            ].copy()  # Get a copy from the master index
                            related_s["_fusion_reason"] = (
                                f"related_to:{source_id}(type:{rel_type})"
                            )
                            # Assign a moderate base score, could be boosted by its own merit later
                            related_s.setdefault(
                                "_similarity_score",
                                0.4
                                + (
                                    0.1
                                    * (max_additional - added_count)
                                    / max_additional
                                ),
                            )  # Higher for earlier fusions
                            fused_sigils_list.append(related_s)
                            current_sigil_ids.add(target_id)  # Track added sigils
                            added_count += 1
                            logger.debug(
                                f"MessageContextEnhancer: Auto-fused '{target_id}' based on relation from '{source_id}'."
                            )

            # (Optional: Shared tags logic could be added here if desired, similar to relationships)

        if added_count > 0:
            logger.info(
                f"MessageContextEnhancer: Auto-fused {added_count} additional sigils."
            )
        return fused_sigils_list

    def _optimize_context_by_chars(
        self,
        sigils_for_context: list[dict[str, Any]],
        initial_detail_level: str,
        target_char_budget: int,
    ) -> tuple[list[dict[str, Any]], str]:
        # (Implementation from original, check detail_levels order)
        final_sigils = list(sigils_for_context)  # Work with a copy
        current_detail = initial_detail_level.lower()

        if not target_char_budget or not final_sigils:
            return final_sigils, current_detail

        # Order from most verbose to least verbose for iterative reduction
        detail_levels = ["full", "detailed", "standard", "summary", "minimal"]
        try:
            current_detail_idx = detail_levels.index(current_detail)
        except ValueError:  # Fallback if initial_detail_level is invalid
            current_detail_idx = detail_levels.index("standard")
            current_detail = "standard"
            logger.warning(
                f"MessageContextEnhancer: Invalid initial_detail_level '{initial_detail_level}', defaulting to 'standard'."
            )

        # Helper to estimate character count
        def estimate_chars(s_list, d_lvl):
            return sum(
                len(self.format_sigil_for_context(s, d_lvl, include_explanations=True))
                for s in s_list
            )

        current_chars = estimate_chars(final_sigils, current_detail)
        logger.debug(
            f"MessageContextEnhancer Context Optimizer: Initial: {len(final_sigils)} sigils, Detail: {current_detail}, Chars: {current_chars}, Budget: {target_char_budget}"
        )

        # Step 1: Reduce detail level
        while (
            current_chars > target_char_budget
            and current_detail_idx < len(detail_levels) - 1
        ):
            current_detail_idx += 1
            new_detail = detail_levels[current_detail_idx]
            logger.info(
                f"MessageContextEnhancer Context Optimizer: Budget {target_char_budget}, Chars {current_chars}. Reducing detail {current_detail} -> {new_detail}"
            )
            current_detail = new_detail
            current_chars = estimate_chars(final_sigils, current_detail)

        # Step 2: If still over budget, remove sigils (lowest score first, assuming sorted)
        # Ensure sigils are sorted by score (descending) before this function, or sort here.
        # For simplicity, assume they are sorted by relevance. If not, sort them:
        final_sigils.sort(key=lambda x: x.get("_similarity_score", 0.0), reverse=True)

        while (
            current_chars > target_char_budget and len(final_sigils) > 1
        ):  # Keep at least one if possible
            # Remove the least relevant sigil (last in the sorted list)
            removed_s = final_sigils.pop()
            logger.info(
                f"MessageContextEnhancer Context Optimizer: Detail {current_detail}. Over budget. Removing sigil '{removed_s.get('sigil', 'N/A')}' (score: {removed_s.get('_similarity_score', 0.0):.2f})"
            )
            current_chars = estimate_chars(final_sigils, current_detail)

        # Final check: if even one sigil at minimal detail is too long (edge case)
        if (
            len(final_sigils) == 1
            and current_chars > target_char_budget
            and current_detail == "minimal"
        ):
            logger.warning(
                f"MessageContextEnhancer Context Optimizer: Single sigil at minimal detail ({current_chars} chars) exceeds budget ({target_char_budget}). Context may be truncated by LLM."
            )

        logger.info(
            f"MessageContextEnhancer Context Optimizer: Final: {len(final_sigils)} sigils, Detail: {current_detail}, Chars: {current_chars}."
        )
        return final_sigils, current_detail

    def enhanced_rag_process(
        self,
        query: str,
        num_sigils: int = 5,
        filter_tags: Optional[list[str]] = None,
        exclude_tags: Optional[list[str]] = None,
        detail_level: str = "standard",
        apply_recency_boost_flag: bool = True,  # Renamed from apply_recency_boost
        augment_query_flag: bool = True,
        enable_context_optimization: bool = True,
        max_context_chars: int = 8000,
        auto_fuse_related_flag: bool = True,  # Renamed from auto_fuse_related
        max_fusion_sigils: int = 3,
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Comprehensive RAG processing with augmentation, filtering, boosting, fusion, and optimization.
        Returns: Formatted context string and the list of sigils used for that context.
        """
        if (
            not hasattr(self.processor, "standard_rag")
            or not self.processor.standard_rag
        ):
            logger.warning(
                "MessageContextEnhancer: Main RAG component not available for enhanced_rag_process."
            )
            return "", []

        effective_query = self._augment_query(query) if augment_query_flag else query
        logger.info(
            f"MessageContextEnhancer Enhanced RAG: Effective query: '{effective_query[:70]}...'"
        )

        # Initial retrieval using the HybridProcessor's get_rag_context_and_route
        # This method handles the core routing (entropy-based) and initial RAG.
        # We need to pass filter_tags/exclude_tags if the underlying RAG components support them.
        # For this example, we assume create_rag_context in StandardRAGComponent can take these.
        # If not, filtering would happen post-retrieval.
        _raw_context_str, retrieved_sigils, route_method = (
            self.processor.get_rag_context_and_route(
                query=effective_query,
                num_sigils=num_sigils
                * 2,  # Retrieve more initially to allow for filtering/ranking
                filter_tags=filter_tags,  # Pass to underlying RAG
                exclude_tags=exclude_tags,  # Pass to underlying RAG
            )
        )
        logger.info(
            f"MessageContextEnhancer Enhanced RAG: Initial retrieval (route: {route_method}) got {len(retrieved_sigils)} sigils."
        )

        # Post-retrieval processing steps:
        # 1. (Optional) Explicit filtering if not done by RAG component. (Skipped if RAG handles it)

        # 2. Recency Boost
        if apply_recency_boost_flag and retrieved_sigils:
            # Use configured boost factor and max days if available, else defaults
            boost_factor = (
                self.config.blt_hybrid_weight * 0.1
                if hasattr(self.config, "blt_hybrid_weight")
                else 0.05
            )
            max_days = 90  # Could be from config
            retrieved_sigils = self._apply_recency_boost(
                retrieved_sigils, boost_factor, max_days
            )
            # Re-sort by score after boosting
            retrieved_sigils.sort(
                key=lambda x: x.get("_similarity_score", 0.0), reverse=True
            )
            logger.debug(
                f"MessageContextEnhancer Enhanced RAG: After recency boost: {len(retrieved_sigils)} sigils."
            )

        # 3. Auto-fuse related sigils (operates on the current top sigils)
        # Take the top `num_sigils` *before* fusion to decide which ones to fuse from.
        top_sigils_for_fusion_base = retrieved_sigils[:num_sigils]
        if auto_fuse_related_flag and top_sigils_for_fusion_base:
            fused_sigils = self.auto_fuse_related_sigils(
                top_sigils_for_fusion_base, max_additional=max_fusion_sigils
            )
            # Combine fused with any remaining original sigils, ensuring no duplicates and respecting num_sigils roughly
            # This logic ensures fused sigils are prioritized but doesn't inflate beyond a reasonable limit.
            # Create a dict of sigils by ID to easily merge and keep the best score if duplicates arise.
            final_sigil_candidates = {s["sigil"]: s for s in fused_sigils}
            for s in (
                retrieved_sigils
            ):  # Add remaining from original if not already in fused set
                if s["sigil"] not in final_sigil_candidates:
                    final_sigil_candidates[s["sigil"]] = s

            retrieved_sigils = sorted(
                list(final_sigil_candidates.values()),
                key=lambda x: x.get("_similarity_score", 0.0),
                reverse=True,
            )
            logger.debug(
                f"MessageContextEnhancer Enhanced RAG: After auto-fusion: {len(retrieved_sigils)} sigils."
            )

        # 4. Trim to `num_sigils` after all additions/re-ranking (final selection)
        final_selected_sigils = retrieved_sigils[:num_sigils]

        # 5. Context Optimization (size and detail level)
        final_detail_level = detail_level
        if (
            enable_context_optimization
            and max_context_chars > 0
            and final_selected_sigils
        ):
            final_selected_sigils, final_detail_level = self._optimize_context_by_chars(
                final_selected_sigils, detail_level, max_context_chars
            )
            logger.debug(
                f"MessageContextEnhancer Enhanced RAG: After context optimization: {len(final_selected_sigils)} sigils, detail: {final_detail_level}."
            )

        # 6. Format the final list of sigils into a context string
        formatted_context = self.format_sigils_for_context(
            final_selected_sigils,
            detail_level=final_detail_level,
            include_explanations=True,
        )

        logger.info(
            f"MessageContextEnhancer Enhanced RAG: Final context ready ({len(formatted_context)} chars, {len(final_selected_sigils)} sigils)."
        )
        return formatted_context, final_selected_sigils


# --- Unified Access Point: VantaMiddlewareSuite ---
class VantaMiddlewareSuite:
    def __init__(
        self,
        arc_task_config: Dict[str, Any],
        message_enhancer_config: HybridMiddlewareConfig,  # Uses the specific HybridMiddlewareConfig
        supervisor_connector: BaseSupervisorConnector,
        blt_encoder: BaseBLTEncoder,
    ):
        """
        Initializes and manages the suite of Vanta middlewares.
        """
        self.supervisor_connector = supervisor_connector
        self.blt_encoder = blt_encoder
        self.arc_task_config = arc_task_config
        self.message_enhancer_config = message_enhancer_config

        logger.info("Initializing VantaMiddlewareSuite...")

        # Initialize MessageContextEnhancementMiddleware first, as ARCMiddleware might use it
        self.message_enhancer = MessageContextEnhancementMiddleware(
            config=self.message_enhancer_config,
            supervisor_connector=self.supervisor_connector,
            blt_encoder=self.blt_encoder,
        )
        logger.info("MessageContextEnhancementMiddleware initialized in suite.")

        # ARCTaskProcessingMiddleware can use the message_enhancer for RAG
        self.arc_task_processor = ARCTaskProcessingMiddleware(
            config=self.arc_task_config,
            blt_encoder_instance=self.blt_encoder,
            supervisor_connector=self.supervisor_connector,
            context_enhancer=self.message_enhancer,  # Pass the instance here
            global_hybrid_config=self.message_enhancer_config,  # Can also use this for its internal BLT config base
        )
        logger.info("ARCTaskProcessingMiddleware initialized in suite.")

        # Placeholder for other middlewares from the map
        # self.echo_middleware = EchoMiddleware(...)
        # self.dream_middleware = DreamMiddleware(...)
        # ... and so on for others from the map

        logger.info("VantaMiddlewareSuite initialized successfully.")

    def get_arc_processor(self) -> ARCTaskProcessingMiddleware:
        return self.arc_task_processor

    def get_message_enhancer(self) -> MessageContextEnhancementMiddleware:
        return self.message_enhancer

    # Add methods to get other middlewares once they are defined
    # def get_echo_middleware(self): ...


# --- Vanta Middleware Map (from user prompt, slightly updated names if applicable) ---
"""
📚 Vanta Middleware Map (⟠∆∇𓂀⟁)

A comprehensive index of middleware components across the Vanta + VoxSigil ecosystem.
This map includes current, in-use middleware, partials, and recommended future modules.

✅ Current Middleware Implemented (as demonstrated in this unified script)

Middleware Class                   Purpose                                                           Status
----------------------------------- ----------------------------------------------------------------- --------------------
ARCTaskProcessingMiddleware        Solves ARC tasks with RAG, LLM prompting, retries, and eval       ✅ Active in Script
MessageContextEnhancementMiddleware Enhances LLM chat inputs with RAG-based sigil context             ✅ Active in Script
HybridProcessor                    (Internal) Entropy-based routing between token and BLT RAG paths ✅ Used by Message Enhancer
BLTHybridMiddleware                (Internal) BLT-enhanced wrapper for RAG context (fallback)        ⚠️ Used internally by ARC Task Proc.
EntropyRouter                      (Internal) Routes queries based on entropy                      ✅ Used indirectly

⚠️ Middleware Missing / Partially Implemented (To be added to VantaMiddlewareSuite)

Proposed Middleware              Purpose                                                           Suggested File
-------------------------------- ----------------------------------------------------------------- ---------------------------------
EchoMiddleware                   Injects echoed memory or recent context into prompt chains        middleware/echo_middleware.py
DreamMiddleware                  Handles async/sleep-mode processing, memory distillation          middleware/dream_middleware.py
SigilValidationMiddleware        Validates sigils with schema, checks relationship coherence       middleware/validation_middleware.py
InteractionLoggingMiddleware     Tracks sigil usage, route decisions, retry history, metrics       middleware/logging_middleware.py
AgentCoordinationMiddleware      Distributes subtasks and routing across multi-agent chains        middleware/agent_coordination.py
GuardrailMiddleware              Blocks sigils with expired trust, injection content, or unsafe    middleware/guardrail_middleware.py
FallbackReplayMiddleware         Controls retry chains with mutation (temp, path, detail-level)    middleware/fallback_replay.py

🧠 Utility Layered Under Middleware (Many of these are methods within the implemented classes)

Utility                             Description                                                     Consumed By
----------------------------------- --------------------------------------------------------------- ---------------------------------
_augment_query                      Synonym-based query expansion                                   MessageContextEnhancementMiddleware
_apply_recency_boost                Adds time-weight to recently modified sigils                    MessageContextEnhancementMiddleware
_normalize_sigil_relationships    Ensures relationships are dict-based and formatted cleanly        MessageContextEnhancementMiddleware
_optimize_context_by_chars          Adjusts detail level & sigil count to fit char budget           MessageContextEnhancementMiddleware
auto_fuse_related_sigils            Adds related sigils to context based on links/tags              MessageContextEnhancementMiddleware
hybrid_embedding_utility            (Implicit) Direct route/entropy-based hybrid embedding          Internal to HybridProcessor
entropy_router_util                 (Implicit) Quick access wrapper to EntropyRouter                Internal to HybridProcessor

🔮 Next Steps
This map is version-controlled. Update after any new middleware class is registered.
Forge the path. Route the echoes. Guard the light.
"""

# --- Example Usage ---
if __name__ == "__main__":
    # Configure logging for detailed output during example run
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # 1. Setup dummy external components
    supervisor = BaseSupervisorConnector()
    encoder = BaseBLTEncoder()

    # 2. Define configurations
    arc_config_params = {
        "rag_enabled": True,
        "max_rag_results": 3,
        "llm_model": "arc-solver-v1",
        "log_level": "DEBUG",
    }

    # This config is for HybridMiddlewareConfig which is used by MessageContextEnhancementMiddleware
    # and also can be a base for ARCTaskProcessingMiddleware's internal BLT.
    enhancer_hybrid_config = HybridMiddlewareConfig(
        entropy_threshold=0.3,
        blt_hybrid_weight=0.75,  # Used for recency boost factor in MessageContextEnhancer
        cache_ttl_seconds=600,
        log_level="DEBUG",
        max_rag_results=7,  # Default for MessageContextEnhancer RAG calls
        # Parameters for enhanced_rag_process defaults:
        # "augment_query_flag": True,
        # "enable_context_optimization": True,
    )

    # 3. Initialize the suite
    print("\n--- Initializing VantaMiddlewareSuite ---")
    suite = VantaMiddlewareSuite(
        arc_task_config=arc_config_params,
        message_enhancer_config=enhancer_hybrid_config,
        supervisor_connector=supervisor,
        blt_encoder=encoder,
    )
    print("--- VantaMiddlewareSuite Initialized ---")

    # 4. Use MessageContextEnhancementMiddleware
    print("\n--- Testing MessageContextEnhancementMiddleware ---")
    chat_enhancer = suite.get_message_enhancer()

    # Test the __call__ interface for enhancing LLM inputs
    llm_input_raw = {
        "model": "chat-gpt-x",
        "messages": [
            {
                "role": "user",
                "content": "Tell me about VoxSigil and its core principles related to AI alignment.",
            }
        ],
    }
    print(f"\nRaw LLM Input:\n{json.dumps(llm_input_raw, indent=2)}")
    enhanced_llm_input = chat_enhancer(llm_input_raw.copy())  # Pass a copy
    print(f"\nEnhanced LLM Input:\n{json.dumps(enhanced_llm_input, indent=2)}")

    # Test enhanced_rag_process directly
    query_for_rag = (
        "What are the best practices for managing complex sigil relationships in Vanta?"
    )
    print(f"\nDirectly calling enhanced_rag_process for query: '{query_for_rag}'")
    context_string, retrieved_s = chat_enhancer.enhanced_rag_process(
        query=query_for_rag,
        num_sigils=3,
        detail_level="standard",
        max_context_chars=1000,
    )
    print(f"Retrieved Context String (first 300 chars):\n{context_string[:300]}...")
    print(f"Number of sigils in context: {len(retrieved_s)}")
    if retrieved_s:
        print(f"First sigil in context: {retrieved_s[0].get('sigil', 'N/A')}")

    # 5. Use ARCTaskProcessingMiddleware
    print("\n--- Testing ARCTaskProcessingMiddleware ---")
    arc_processor = suite.get_arc_processor()

    # Simulate creating task and input sigils in supervisor (or assume they exist)
    dummy_input_sigil = "SigilRef:DummyInput_ARC_123"
    dummy_task_sigil = "SigilRef:DummyTask_ARC_123"
    supervisor.create_sigil(
        dummy_input_sigil,
        {"grid": [[1, 0], [0, 1]], "description": "A 2x2 input grid"},
        "ARC_Input",
    )
    supervisor.create_sigil(
        dummy_task_sigil,
        {
            "description": "Transform the input by inverting colors.",
            "expected_output": {"grid": [[0, 1], [1, 0]]},
        },
        "ARC_Task",
    )

    solution_ref, perf_ref = arc_processor.process_arc_task(
        input_data_sigil_ref=dummy_input_sigil,
        task_sigil_ref=dummy_task_sigil,
        task_parameters={"effective_temperature": 0.1, "max_solution_attempts": 1},
    )
    print(
        f"ARC Task processed. Solution Sigil: {solution_ref}, Performance Sigil: {perf_ref}"
    )

    # 6. Get stats from MessageContextEnhancementMiddleware
    print("\n--- Message Enhancer Stats ---")
    stats = chat_enhancer.get_stats()
    print(json.dumps(stats, indent=2))

    print("\n--- End of Example Usage ---")
