# ARC/arc_reasoner.py
import json
import logging
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple  # Added TypedDict, Set

import numpy as np  # For Feature 1

# --- Module Imports ---
# Critical dependencies required for this module to function
from ARC.arc_config import (
    ARC_SOLVER_SIGIL_NAME,
    CATENGINE_SIGIL_NAME,
    DEFAULT_STRATEGY_PRIORITY,  # For Feature 2
    DETAILED_PROMPT_METADATA,  # For Feature 4
    LLM_SYNTHESIZER_TEMPERATURE,
    SYNTHESIS_FAILURE_FALLBACK_STRATEGY,  # For Feature 9
)
from handlers.arc_llm_handler import (
    robust_parse_arc_grid_from_llm_text,
)
from llm.arc_voxsigil_loader import VoxSigilComponent
from llm.llm_api_compat import call_llm_api

logger = logging.getLogger(__name__)


# --- Feature 1: Typed Task Properties Object ---
@dataclass
class ARCTaskProperties:
    """Structured properties of an ARC task for reasoning and strategy selection."""

    task_id: str
    num_train_examples: int
    num_test_examples: int
    input_colors: Set[int]
    output_colors: Set[int]
    avg_grid_size_train: Tuple[float, float]  # (avg_height, avg_width)
    avg_grid_size_test: Tuple[float, float]
    needs_categorization: bool = False
    # For Feature 8 (Advanced Tag-Based Strategy Filtering)
    required_tags_for_strategy: List[str] = field(default_factory=list)
    excluded_tags_for_strategy: List[str] = field(default_factory=list)
    # Add other relevant features as identified:
    # e.g., uses_symmetry: bool, involves_counting: bool, etc.


# --- Feature 7: Fallback Strategy Cache (Module-level) ---
_strategy_availability_cache: Dict[
    Tuple[str, str], bool
] = {}  # Key: (strategy_name, task_prop_fingerprint), Value: is_available


# --- Task Analysis ---
def _validate_task_data_for_analysis(
    task_data: Dict[str, Any],
) -> bool:  # Helper for Feature 6
    """Validates basic structure of task_data needed for analysis."""
    if not isinstance(task_data, dict):
        logger.warning("Task data for analysis is not a dictionary.")
        return False
    if not task_data.get("train") or not isinstance(task_data["train"], list):
        logger.debug("Task data 'train' field missing or not a list for analysis.")
        return False
    # Add more checks as needed (e.g., for 'input'/'output' keys in examples)
    return True


def analyze_task_properties(task_id: str, task_data: Dict[str, Any]) -> ARCTaskProperties:
    """Analyzes ARC task data and returns a structured ARCTaskProperties object."""
    # Feature 6: Input Validation (basic example)
    if not _validate_task_data_for_analysis(task_data):
        logger.warning(f"Invalid task_data for {task_id} analysis. Returning default properties.")
        return ARCTaskProperties(
            task_id=task_id,
            num_train_examples=0,
            num_test_examples=0,
            input_colors=set(),
            output_colors=set(),
            avg_grid_size_train=(0, 0),
            avg_grid_size_test=(0, 0),
            needs_categorization=False,
        )

    train_examples = task_data.get("train", [])
    test_examples = task_data.get("test", [])

    input_colors_overall = set()
    output_colors_overall = set()
    train_heights, train_widths = [], []
    test_heights, test_widths = [], []

    for ex_list, heights_list, widths_list in [
        (train_examples, train_heights, train_widths),
        (test_examples, test_heights, test_widths),
    ]:
        for ex in ex_list:
            if not isinstance(ex, dict) or "input" not in ex:
                continue  # Basic check
            # Input properties
            if ex.get("input") and isinstance(ex["input"], list) and ex["input"]:
                input_colors_overall.update(cell for row in ex["input"] for cell in row)
                heights_list.append(len(ex["input"]))
                widths_list.append(len(ex["input"][0]) if ex["input"][0] else 0)
            # Output properties (only for training examples for color analysis here)
            if ex_list is train_examples and ex.get("output") and isinstance(ex["output"], list):
                output_colors_overall.update(cell for row in ex["output"] for cell in row)

    # Categorization Heuristics
    high_color_diversity = (
        len(input_colors_overall) > 4
        and len(output_colors_overall) >= len(input_colors_overall) * 0.7
    )

    def has_large_monochromatic_regions(grid: List[List[int]], min_region_size=3) -> bool:
        if not grid or not grid[0]:
            return False
        rows, cols = len(grid), len(grid[0])
        for r in range(rows):
            for c in range(cols):
                color = grid[r][c]
                if color == 0:
                    continue
                for dr, dc in [(0, 1), (1, 0)]:  # Check horizontal and vertical runs
                    run_len, curr_r, curr_c = 0, r, c
                    while (
                        0 <= curr_r < rows and 0 <= curr_c < cols and grid[curr_r][curr_c] == color
                    ):
                        run_len += 1
                        curr_r += dr
                        curr_c += dc
                    if run_len >= min_region_size:
                        return True
        return False

    clustered_regions_present = any(
        has_large_monochromatic_regions(ex.get("input", [])) for ex in train_examples
    )

    color_preservation = False
    for ex in train_examples:
        if "input" not in ex or "output" not in ex:
            continue
        ex_in_colors = {cell for row in ex["input"] for cell in row if cell != 0}
        ex_out_colors = {cell for row in ex["output"] for cell in row if cell != 0}
        if len(ex_in_colors.intersection(ex_out_colors)) >= 2:
            color_preservation = True
            break

    needs_cat = sum([high_color_diversity, clustered_regions_present, color_preservation]) >= 2

    return ARCTaskProperties(
        task_id=task_id,
        num_train_examples=len(train_examples),
        num_test_examples=len(test_examples),
        input_colors=input_colors_overall,
        output_colors=output_colors_overall,
        avg_grid_size_train=(
            float(np.mean(train_heights)) if train_heights else 0.0,
            float(np.mean(train_widths)) if train_widths else 0.0,
        ),
        avg_grid_size_test=(
            float(np.mean(test_heights)) if test_heights else 0.0,
            float(np.mean(test_widths)) if test_widths else 0.0,
        ),
        needs_categorization=needs_cat,
    )


# --- Strategy Selection ---
def _check_strategy_availability(
    strategy_name: str,
    task_prop_fingerprint: str,
    available_components: List[VoxSigilComponent],
    sigil_name_to_match: str,
) -> Optional[VoxSigilComponent]:
    """Helper for Feature 7: Checks cache or finds component."""
    cache_key = (strategy_name, task_prop_fingerprint)
    if cache_key in _strategy_availability_cache and not _strategy_availability_cache[cache_key]:
        logger.debug(
            f"Strategy '{strategy_name}' previously determined unavailable for similar task props. Skipping."
        )
        return None  # Cached as unavailable

    components = [
        c
        for c in available_components
        if c.sigil == sigil_name_to_match and c.prompt_template_content
    ]
    if components:
        _strategy_availability_cache[cache_key] = True
        return random.choice(components)
    else:
        _strategy_availability_cache[cache_key] = False
        return None


def select_reasoning_strategy(
    task_props: ARCTaskProperties,  # Feature 1
    available_voxsigil_components: List[VoxSigilComponent],
    force_catengine_env: bool,
    force_arc_solver_env: bool,
    strategy_priority: List[str] = DEFAULT_STRATEGY_PRIORITY,  # Feature 2
) -> Tuple[Optional[str], Optional[VoxSigilComponent]]:
    # Feature 6: Input validation (basic example)
    if not isinstance(task_props, ARCTaskProperties) or not available_voxsigil_components:
        logger.error("Invalid input to select_reasoning_strategy.")
        return None, None

    # Create a simple fingerprint for task properties for caching (Feature 7)
    # For more complex caching, a more robust fingerprinting mechanism would be needed.
    task_prop_fingerprint = f"cat:{task_props.needs_categorization}"

    for strategy_type in strategy_priority:
        selected_component: Optional[VoxSigilComponent] = None
        strategy_name_for_logging = ""

        if strategy_type == "ENV_FORCE":
            if force_catengine_env:
                selected_component = _check_strategy_availability(
                    "FORCED_CATENGINE",
                    task_prop_fingerprint,
                    available_voxsigil_components,
                    CATENGINE_SIGIL_NAME,
                )
                if selected_component:
                    strategy_name_for_logging = (
                        f"Forced CATENGINE (env): {selected_component.sigil}"
                    )
                    break
                else:
                    logger.warning(
                        "FORCE_CATENGINE env var set, but CATENGINE component not found/available."
                    )
            if force_arc_solver_env:  # Can be forced even if CATENGINE was too
                selected_component = _check_strategy_availability(
                    "FORCED_ARC_SOLVER",
                    task_prop_fingerprint,
                    available_voxsigil_components,
                    ARC_SOLVER_SIGIL_NAME,
                )
                if selected_component:
                    strategy_name_for_logging = (
                        f"Forced ARC_SOLVER (env): {selected_component.sigil}"
                    )
                    break
                else:
                    logger.warning(
                        "FORCE_ARC_SOLVER env var set, but ARC_SOLVER component not found/available."
                    )

        elif strategy_type == "CATENGINE_IF_NEEDED" and task_props.needs_categorization:
            selected_component = _check_strategy_availability(
                "CATENGINE_PROPERTY_BASED",
                task_prop_fingerprint,
                available_voxsigil_components,
                CATENGINE_SIGIL_NAME,
            )
            if selected_component:
                strategy_name_for_logging = f"CATENGINE (props): {selected_component.sigil}"
                break

        elif strategy_type == "ARC_SOLVER_EXACT":
            selected_component = _check_strategy_availability(
                "ARC_SOLVER_EXACT",
                task_prop_fingerprint,
                available_voxsigil_components,
                ARC_SOLVER_SIGIL_NAME,
            )
            if selected_component:
                strategy_name_for_logging = f"ARC_SOLVER (exact): {selected_component.sigil}"
                break

        elif strategy_type == "TAGGED_ARC":
            # Feature 8: Advanced Tag-Based Strategy Filtering
            candidates = [c for c in available_voxsigil_components if c.prompt_template_content]
            if task_props.required_tags_for_strategy:
                candidates = [
                    c
                    for c in candidates
                    if all(
                        req_tag.lower() in (t.lower() for t in c.tags)
                        for req_tag in task_props.required_tags_for_strategy
                    )
                ]
            if task_props.excluded_tags_for_strategy:
                candidates = [
                    c
                    for c in candidates
                    if not any(
                        ex_tag.lower() in (t.lower() for t in c.tags)
                        for ex_tag in task_props.excluded_tags_for_strategy
                    )
                ]
            # Default ARC/SpatialReasoning tags if no specific required_tags are set by task_props
            if not task_props.required_tags_for_strategy:
                candidates = [
                    c
                    for c in candidates
                    if any(t.lower() in ["arc", "spatialreasoning"] for t in c.tags)
                ]

            if candidates:
                selected_component = random.choice(candidates)
                strategy_name_for_logging = (
                    f"TAGGED ({selected_component.tags}): {selected_component.sigil}"
                )
                break
        elif strategy_type == "GENERAL_REASONING":
            candidates = [
                c
                for c in available_voxsigil_components
                if c.prompt_template_content
                and (
                    any(
                        t.lower() in ["reasoningstrategy", "problemsolving", "logic"]
                        for t in c.tags
                    )
                    or c.execution_mode in ["simulation", "transformation", "generation"]
                )
            ]
            if candidates:
                selected_component = random.choice(candidates)
                strategy_name_for_logging = (
                    f"GENERAL ({selected_component.tags}): {selected_component.sigil}"
                )
                break

        if selected_component:  # Found a strategy based on current strategy_type in priority list
            logger.info(f"🎯 Selected Strategy ({strategy_name_for_logging})")
            return (
                selected_component.sigil,
                selected_component,
            )  # Return if a component is selected

    logger.error(
        "🚫 No suitable VoxSigil reasoning components with prompt templates found after checking all priorities!"
    )
    return None, None


# --- Prompt Building ---
def _resolve_dynamic_placeholders(
    component: VoxSigilComponent, task_props: ARCTaskProperties
) -> Dict[str, Any]:  # Helper for Feature 3
    """Resolves dynamic placeholders defined in the VoxSigilComponent."""
    resolved_params: Dict[str, Any] = {}
    if hasattr(component, "expected_placeholders") and component.expected_placeholders:
        for placeholder_name in component.expected_placeholders:
            # Example logic: try to find directly in task_props
            if hasattr(task_props, placeholder_name):
                resolved_params[placeholder_name] = getattr(task_props, placeholder_name)
            # Add more complex resolution logic here if needed
            # e.g., mapping component placeholder names to different task_prop names
            # or deriving values.
            else:
                logger.debug(
                    f"Placeholder '{placeholder_name}' expected by sigil '{component.sigil}' not found in task_props."
                )
    return resolved_params


def build_arc_prompt(
    selected_voxsigil_component: VoxSigilComponent,
    arc_task_data: Dict[str, Any],
    task_props: ARCTaskProperties,  # Feature 1
) -> Tuple[str, Dict[str, Any]]:  # Feature 4: Return prompt and metadata
    # Feature 6: Input validation
    if (
        not isinstance(selected_voxsigil_component, VoxSigilComponent)
        or not arc_task_data
        or not isinstance(task_props, ARCTaskProperties)
    ):
        logger.error("Invalid inputs to build_arc_prompt.")
        return (
            "Error: Invalid inputs for prompt building.",
            {},
        )  # Ensure prompt_template_content exists
    if not selected_voxsigil_component.prompt_template_content:
        logger.error(
            f"VoxSigilComponent {selected_voxsigil_component.sigil} has no prompt template content."
        )
        return "Error: No prompt template available for this component.", {}

    prompt_metadata: Dict[str, Any] = {  # Feature 4
        "sigil_used": selected_voxsigil_component.sigil,
        "template_name_or_source": selected_voxsigil_component.prompt_template_content[:50]
        + "...",  # Example
        "task_id": task_props.task_id,
        "execution_mode": selected_voxsigil_component.execution_mode,
        "injected_parameters": {},
        "dynamic_placeholders_resolved": {},
    }

    voxsigil_template_str = selected_voxsigil_component.prompt_template_content

    # Standard Parameterization (from schema)
    parameters_from_schema = {}
    if selected_voxsigil_component.parameterization_schema and isinstance(
        selected_voxsigil_component.parameterization_schema.get("parameters"), list
    ):
        for param_def in selected_voxsigil_component.parameterization_schema["parameters"]:
            param_name = param_def.get("name")
            # Use task_props attributes (which are now typed and structured)
            if hasattr(task_props, param_name):  # Check ARCTaskProperties object
                parameters_from_schema[param_name] = getattr(task_props, param_name)
            elif "default_value" in param_def:
                parameters_from_schema[param_name] = param_def["default_value"]
        if parameters_from_schema:
            logger.info(
                f"  Injecting Schema VoxSigil params for '{selected_voxsigil_component.sigil}': {parameters_from_schema}"
            )
            prompt_metadata["injected_parameters"].update(parameters_from_schema)

    # Feature 3: Dynamic Placeholder Injection
    dynamic_resolved_placeholders = _resolve_dynamic_placeholders(
        selected_voxsigil_component, task_props
    )
    if dynamic_resolved_placeholders:
        logger.info(
            f"  Injecting Dynamic Placeholders for '{selected_voxsigil_component.sigil}': {dynamic_resolved_placeholders}"
        )
        prompt_metadata["dynamic_placeholders_resolved"].update(dynamic_resolved_placeholders)

    def grid_to_compact_str(grid: List[List[int]]) -> str:
        if not grid:
            return "[]"  # Handle empty grid case
        return "[" + ",".join("[" + ",".join(map(str, row)) + "]" for row in grid) + "]"

    train_examples_str = "\nNo training examples provided.\n"
    if arc_task_data.get("train") and isinstance(arc_task_data["train"], list):
        train_examples_str = ""
        for i, pair in enumerate(arc_task_data["train"]):
            if isinstance(pair, dict) and "input" in pair and "output" in pair:
                train_examples_str += f"\nExample {i + 1} Input:\n{grid_to_compact_str(pair['input'])}\nExample {i + 1} Output:\n{grid_to_compact_str(pair['output'])}\n"

    test_input_str = "Test input not available."
    if (
        arc_task_data.get("test")
        and isinstance(arc_task_data["test"], list)
        and arc_task_data["test"][0]
        and isinstance(arc_task_data["test"][0], dict)
        and arc_task_data["test"][0].get("input") is not None
    ):
        test_input_str = grid_to_compact_str(arc_task_data["test"][0]["input"])

    strict_output_format_instruction = (
        "IMPORTANT INSTRUCTION: Your final answer MUST BE ONLY the predicted output grid. "
        "Format this grid as a pure JSON array-of-arrays (e.g., [[1,0],[0,1]] or [[2,3,4],[5,6,7],[8,9,0]]). "
        "Do NOT include any other text, explanations, dialogue, apologies, or markdown formatting. "
        "Your output must be valid JSON. ONLY the final JSON array of arrays."
    )

    prompt = voxsigil_template_str
    # Apply parameter injections (schema-based first, then dynamic)
    all_params_to_inject = {**parameters_from_schema, **dynamic_resolved_placeholders}
    for p_name, p_value in all_params_to_inject.items():
        for placeholder_format in [f"{{{{{p_name}}}}}", f"{{{{params.{p_name}}}"]:
            if placeholder_format in prompt:
                prompt = prompt.replace(placeholder_format, str(p_value))

    # Standard placeholders (fixed)
    # Feature 10: Customizable Task Description
    try:
        task_desc_str = selected_voxsigil_component.get_task_description(
            task_props.task_id, task_props
        )
    except (AttributeError, Exception) as e:
        logger.warning(f"Failed to get task description: {e}")
        task_desc_str = f"Solve ARC puzzle '{task_props.task_id}' using '{selected_voxsigil_component.sigil}' VoxSigil strategy."

    fixed_placeholders = {
        "{{task_id}}": task_props.task_id,
        "{{task_description}}": task_desc_str,
        "{{train_examples}}": train_examples_str,
        "{{test_input}}": test_input_str,
        "{{execution_mode}}": selected_voxsigil_component.execution_mode
        or "instructional_analysis",
        "{{output_instructions}}": strict_output_format_instruction,
    }
    for ph, val in fixed_placeholders.items():
        if ph in prompt:
            prompt = prompt.replace(ph, val)

    # Ensure critical output instruction is present
    try:
        if "{{output_instructions}}" not in voxsigil_template_str and not re.search(
            r"MUST BE ONLY the predicted output grid|ONLY.*JSON.*array-of-arrays",
            prompt,
            re.IGNORECASE | re.DOTALL,
        ):
            prompt = strict_output_format_instruction + "\n\n" + prompt
            prompt_metadata["output_instruction_prepended"] = True
    except Exception as e:
        logger.warning(f"Error checking output instructions: {e}")
        # Safe fallback - always prepend
        prompt = strict_output_format_instruction + "\n\n" + prompt
        prompt_metadata["output_instruction_prepended"] = True

    return prompt, prompt_metadata if DETAILED_PROMPT_METADATA else {}


# --- Prediction Synthesis ---
def _calculate_synthesis_confidence(  # Helper for Feature 5
    synthesized_grid: Optional[List[List[int]]],
    valid_grids: List[List[List[int]]],
    vote_counter: Counter,
    method_used: str,
) -> float:
    if not synthesized_grid or not valid_grids:
        return 0.0

    if method_used == "majority_vote" or method_used == "unanimous_vote":
        grid_tuple = tuple(map(tuple, synthesized_grid))
        return vote_counter[grid_tuple] / len(valid_grids) if len(valid_grids) > 0 else 0.0
    elif method_used == "llm_synthesis":
        # If LLM synthesized, it's harder to get a direct confidence.
        # One heuristic: how many of the original valid_grids match the LLM's synthesis?
        matches = sum(1 for vg in valid_grids if vg == synthesized_grid)
        # Base confidence for LLM could be lower, e.g., 0.6, boosted by agreement
        base_confidence = 0.6
        agreement_boost = (matches / len(valid_grids)) * 0.3 if len(valid_grids) > 0 else 0
        return min(1.0, base_confidence + agreement_boost)
    return 0.3  # Default low confidence for unknown method


def synthesize_llm_predictions(
    arc_task_full_prompt: str,
    parsed_solver_predictions: List[Optional[List[List[int]]]],
    synthesizer_cfg: Dict[str, Any],
    solver_cfgs: List[Dict[str, Any]],
    voxsigil_system_prompt_text: str,
    use_voxsigil_system_prompt_flag: bool,
    # Feature 9: Configurable Synthesis Fallback
    failure_fallback_strategy: str = SYNTHESIS_FAILURE_FALLBACK_STRATEGY,
) -> Tuple[List[List[int]], float, str]:  # Return grid, confidence, method_used
    # Feature 6: Input Validation
    if not isinstance(parsed_solver_predictions, list) or not synthesizer_cfg:
        logger.error("Invalid inputs for synthesize_llm_predictions.")
        return [[0]], 0.0, "error_input_validation"

    valid_grids = [
        grid for grid in parsed_solver_predictions if grid and isinstance(grid, list) and grid[0]
    ]  # Ensure grid[0] exists (not empty row list)
    method_used = "no_valid_predictions"
    confidence = 0.0

    if not valid_grids:
        logger.warning("Synthesize: No valid grids provided for synthesis.")
        return [[0]], confidence, method_used

    if len(valid_grids) == 1:
        logger.info("Synthesize: Only one valid prediction. Using it.")
        method_used = "single_valid_prediction"
        confidence = 0.9  # High confidence for single valid solver output
        return valid_grids[0], confidence, method_used

    grid_tuples = [tuple(map(tuple, g)) for g in valid_grids]
    vote_counter = Counter(grid_tuples)
    logger.info(f"Synthesize: Vote counts: {vote_counter}")

    most_common_grid_tuple, majority_count = vote_counter.most_common(1)[0]

    # Check for unanimous or strong majority
    if len(vote_counter) == 1:  # Unanimous
        method_used = "unanimous_vote"
        logger.info(
            f"Synthesizing by unanimous vote (all {majority_count} predictions are identical)."
        )
        final_grid = [list(row) for row in most_common_grid_tuple]
        confidence = _calculate_synthesis_confidence(
            final_grid, valid_grids, vote_counter, method_used
        )
        return final_grid, confidence, method_used

    if majority_count > len(valid_grids) / 2:  # Strong Majority
        method_used = "majority_vote"
        logger.info(
            f"Synthesizing by clear majority vote (count {majority_count} out of {len(valid_grids)})."
        )
        final_grid = [list(row) for row in most_common_grid_tuple]
        confidence = _calculate_synthesis_confidence(
            final_grid, valid_grids, vote_counter, method_used
        )
        return final_grid, confidence, method_used

    # No clear majority, proceed to LLM synthesis
    method_used = "llm_synthesis"
    logger.info(
        f"No clear majority. Using LLM synthesizer: {synthesizer_cfg.get('name', 'UnknownLLM')}"
    )

    formatted_predictions_str = ""  # Correctly map solver names to valid_grids (which is a subset of parsed_solver_predictions)
    valid_pred_indices = [
        i
        for i, p_original in enumerate(parsed_solver_predictions)
        if p_original is not None and p_original and p_original[0]
    ]

    for i, grid_idx_in_original_list in enumerate(valid_pred_indices):
        solver_config = (
            solver_cfgs[grid_idx_in_original_list]
            if grid_idx_in_original_list < len(solver_cfgs)
            else {"name": f"UnknownSolver_{i + 1}"}
        )
        solver_name = solver_config.get("name", f"Solver_{i + 1}")
        formatted_predictions_str += f"\n--- Prediction from Solver '{solver_name}' ---\nGrid:\n{json.dumps(valid_grids[i])}\n---\n"

    synthesizer_prompt_user_content = f"""You are an expert AI Synthesizer for ARC solutions.
Given the original ARC task prompt and several candidate output grids from different solver models, your task is to produce the single, best, and most accurate final output grid.
Carefully analyze the original task (examples and test input) and the provided candidate solutions.
You may select one of the candidate grids if you deem it correct and optimal.
If predictions differ, identify the most plausible underlying logic or pattern that fits the original task's examples AND generates one of the candidates.
If no candidate seems perfect but a common underlying pattern can be refined, derive a new grid based on a superior synthesis of the logic.
Focus on strict adherence to ARC principles: precise transformations, pattern consistency, and minimal complexity of the inferred rule.

### Original ARC Task Prompt (This prompt was used to generate the candidate predictions):
{arc_task_full_prompt}

### Candidate Predictions from Solver Models:
{formatted_predictions_str}

### Your Synthesis Task:
Review all the provided information meticulously. Determine the optimal final output grid that best solves the original ARC task.
Output ONLY the final synthesized grid in pure JSON array-of-arrays format. Do not include any explanations, dialogue, apologies, or markdown formatting.
Your entire response should be just the JSON grid.

Final Synthesized Output Grid:
"""

    synth_messages = []
    if use_voxsigil_system_prompt_flag and voxsigil_system_prompt_text:
        synth_messages.append({"role": "system", "content": voxsigil_system_prompt_text})
    synth_messages.append({"role": "user", "content": synthesizer_prompt_user_content})

    try:
        synthesizer_response_text = call_llm_api(
            synthesizer_cfg, synth_messages, LLM_SYNTHESIZER_TEMPERATURE
        )
        if isinstance(synthesizer_response_text, str):
            llm_synthesized_grid = robust_parse_arc_grid_from_llm_text(
                synthesizer_response_text, synthesizer_cfg.get("service", "synthesizer")
            )
        else:
            logger.warning(f"LLM API response was not a string: {type(synthesizer_response_text)}")
            llm_synthesized_grid = None
    except Exception as e:
        logger.error(f"Error calling LLM API: {e}")
        llm_synthesized_grid = None

    if llm_synthesized_grid:
        logger.info(
            f"LLM Synthesizer ({synthesizer_cfg.get('name', 'UnknownLLM')}) produced output: {str(llm_synthesized_grid)[:100]}"
        )
        confidence = _calculate_synthesis_confidence(
            llm_synthesized_grid, valid_grids, vote_counter, method_used
        )
        return llm_synthesized_grid, confidence, method_used
    else:
        # LLM Synthesizer failed, apply fallback strategy (Feature 9)
        logger.warning(
            f"LLM Synthesizer ({synthesizer_cfg.get('name', 'UnknownLLM')}) FAILED to produce a valid grid. Applying fallback strategy: '{failure_fallback_strategy}'."
        )
        if failure_fallback_strategy == "most_frequent_vote":
            final_grid = [list(row) for row in most_common_grid_tuple]
            method_used = "fallback_most_frequent_vote"
            confidence = _calculate_synthesis_confidence(
                final_grid, valid_grids, vote_counter, "majority_vote"
            )  # Confidence of the vote itself
            return final_grid, confidence, method_used
        elif failure_fallback_strategy == "random_valid_prediction" and valid_grids:
            final_grid = random.choice(valid_grids)
            method_used = "fallback_random_valid"
            confidence = 0.2  # Low confidence for random choice
            return final_grid, confidence, method_used
        else:  # Default or error case
            logger.error(
                f"Unknown or unachievable fallback strategy '{failure_fallback_strategy}' or no valid grids for fallback."
            )
            method_used = "fallback_error_default"
            return [[0]], 0.0, method_used  # Default empty grid, zero confidence


# This module is designed to be imported by other scripts.
# It provides ARC reasoning functionality through its API functions.


class ARCReasoner:
    """
    Production-grade ARC Reasoner class for integration with VoxSigil bridges.
    Provides a solve_with_trace method and encapsulates main reasoning logic.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger("ARC.ARCReasoner")

    def solve_with_trace(self, task_data: Dict[str, Any]) -> Tuple[Dict[str, Any], list]:
        """
        Main entry point for ARC LLM bridge: solves an ARC task and returns solution and trace.
        Args:
            task_data: ARC task dict (should have 'train', 'test', etc.)
        Returns:
            solution_dict: dict with at least 'solution_grid' (List[List[int]])
            trace: list of reasoning steps or log strings
        """
        self.logger.info(f"Solving ARC task with id: {task_data.get('id', 'unknown')}")
        # Analyze task properties
        task_id = task_data.get("id", "unknown")
        task_props = analyze_task_properties(task_id, task_data)
        # Load available VoxSigil components (stub: user should provide or load as needed)
        try:
            available_voxsigil_components = self.rag_interface.retrieve_scaffolds(task_id)
        except Exception:
            available_voxsigil_components = []
        # Select strategy (stub: can be extended)
        strategy, component = select_reasoning_strategy(
            task_props,
            available_voxsigil_components,
            force_catengine_env=False,
            force_arc_solver_env=False,
        )
        # Build prompt (stub: can be extended)
        if component:
            prompt, prompt_metadata = build_arc_prompt(component, task_data, task_props)
        else:
            prompt, prompt_metadata = "No suitable component found.", {}
        # Call LLM or solver (placeholder implementation)
        solution_grid = [[0 for _ in range(3)] for _ in range(3)]
        trace = [
            f"Task analyzed: {task_id}",
            f"Strategy selected: {strategy}",
            f"Prompt built: {prompt[:100]}...",
        ]
        solution_dict = {"solution_grid": solution_grid, "metadata": prompt_metadata}
        return solution_dict, trace

    def some_other_reasoning_method(self, *args, **kwargs) -> Any:
        self.logger.info("Called some_other_reasoning_method (stub)")
        return {"result": "mock_reasoning_result"}
