# voxsigil_supervisor/interfaces/arc_llm_interface.py
"""
ARC-Aware LLM Interface for the VoxSigil Supervisor.

This module provides an implementation of the LLM interface specifically optimized
for ARC-style (Abstraction and Reasoning Corpus) tasks, which involve structured symbolic
reasoning challenges with input/output grid pairs.
"""

import json
import logging
import re  # For more robust JSON extraction
import time  # For retry delays
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

# Import the proper ABC from the llm_interface module
from .llm_interface import BaseLlmInterface

# Setup the logger
# In a multi-file project, this logger might be configured centrally.
# For this script, we'll ensure it's set up if not already.
logger_arc_llm = logging.getLogger("VoxSigilSupervisor.interfaces.arc_llm")
if not logger_arc_llm.hasHandlers():
    # Basic configuration if run standalone or not configured by root logger
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)-8s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger_arc_llm.addHandler(handler)
    logger_arc_llm.setLevel(logging.INFO)  # Default, can be overridden by root


class ARCAwareLLMInterface(BaseLlmInterface):
    """
    An LLM interface optimized for ARC-style tasks with structured symbolic reasoning.
    This interface works with local models using the transformers library.
    """

    DEFAULT_SYSTEM_PROMPT_ARC = (
        "You are an expert AI assistant specialized in solving Abstraction and Reasoning Corpus (ARC) tasks. "
        "Analyze the provided training examples (input/output grid pairs) to understand the underlying transformation rule. "
        "Consider any symbolic context provided as high-level guidance or principles. "
        "Look for patterns involving repetition, reflection, rotation, or combinations of these. "
        "Pay careful attention to how small grids might be expanded into larger grids through repetition or tiling. "
        "Check if the pattern alternates values or repeats them in rows, columns, or blocks. "
        "Look at how the dimensions change between input and output grids - output may be a multiple of the input size. "
        "Apply the inferred rule to the test input grid. "
        "Your goal is to predict the correct output grid. "
        "Respond ONLY with the JSON representation of the output grid in proper JSON array format (e.g., [[1, 2], [3, 4]]). "
        "Make sure each row has the same number of elements. "
        "Do not include any other text, explanations, markdown formatting, or code blocks around the JSON. "
        "Do not use characters like ``` or backticks in your answer, just return the raw grid. "
        "Your entire response should be just the valid JSON grid array and nothing else."
    )

    def __init__(
        self,
        model_path: str,  # Feature 1: Centralized config via parameters
        temperature: float = 0.1,  # Default to more deterministic for ARC
        top_p: float = 0.95,
        top_k: int = 40,
        max_tokens_for_grid: int = 256,  # Renamed for clarity
        repetition_penalty: float = 1.1,
        symbolic_middleware: Optional[Any] = None,  # Feature 2: Symbolic Middleware
        # llm_model and llm_tokenizer can be passed if pre-loaded
        llm_model: Optional[Any] = None,
        llm_tokenizer: Optional[Any] = None,
        device: Optional[str] = None,
    ):  # Feature 10: Device Management
        """
        Initialize the ARC-aware LLM interface.
        """
        super().__init__()  # Call to BaseLlmInterface if it has an __init__
        self.model_path = model_path
        self.default_temperature = temperature
        self.default_top_p = top_p
        self.default_top_k = top_k
        self.default_max_tokens = max_tokens_for_grid
        self.default_repetition_penalty = repetition_penalty
        self.symbolic_middleware = symbolic_middleware

        try:
            from transformers.models.auto.modeling_auto import AutoModelForCausalLM
            from transformers.models.auto.tokenization_auto import AutoTokenizer
        except ImportError:
            logger_arc_llm.critical(
                "Transformers library not installed. Please install with: pip install transformers"
            )
            raise ImportError("Transformers library required for ARCAwareLLMInterface")

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger_arc_llm.info(f"ARCAwareLLMInterface: Using device: {self.device}")

        if llm_model and llm_tokenizer:
            self.tokenizer = llm_tokenizer
            self.model = llm_model
            # Ensure model is on the correct device if pre-loaded
            self.model.to(self.device)
            logger_arc_llm.info(
                f"ARCAwareLLMInterface: Using pre-loaded model and tokenizer from path: {model_path}"
            )
        else:
            try:
                logger_arc_llm.info(
                    f"ARCAwareLLMInterface: Loading tokenizer from {model_path}..."
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path, trust_remote_code=True
                )
                logger_arc_llm.info(
                    f"ARCAwareLLMInterface: Loading model from {model_path} to {self.device}..."
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16
                    if self.device.type == "cuda" and torch.cuda.is_bf16_supported()
                    else (
                        torch.float16 if self.device.type == "cuda" else torch.float32
                    ),
                    device_map="auto"
                    if self.device.type == "cuda"
                    else None,  # "auto" for multi-GPU or fitting large models
                )
                if (
                    self.device.type == "cpu"
                ):  # If device_map="auto" wasn't used (CPU), ensure model is on CPU
                    self.model.to(self.device)
                logger_arc_llm.info(
                    f"ARCAwareLLMInterface: Model successfully loaded from {model_path}"
                )
            except Exception as e:
                logger_arc_llm.critical(
                    f"ARCAwareLLMInterface: Failed to load model/tokenizer from {model_path}: {e}",
                    exc_info=True,
                )
                raise

        self.model.eval()  # Set to evaluation mode

    def _generate_llm_response(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Core LLM generation logic.
        Returns raw text and generation metadata.
        """
        current_max_new_tokens = (
            max_new_tokens if max_new_tokens is not None else self.default_max_tokens
        )
        current_temperature = (
            temperature if temperature is not None else self.default_temperature
        )
        current_top_p = top_p if top_p is not None else self.default_top_p
        current_top_k = top_k if top_k is not None else self.default_top_k
        current_repetition_penalty = (
            repetition_penalty
            if repetition_penalty is not None
            else self.default_repetition_penalty
        )

        gen_metadata = {
            "temperature": current_temperature,
            "top_p": current_top_p,
            "top_k": current_top_k,
            "max_new_tokens": current_max_new_tokens,
            "repetition_penalty": current_repetition_penalty,
        }

        try:
            if (
                hasattr(self.tokenizer, "apply_chat_template") and False
            ):  # Disabled for now, assuming direct prompt for ARC
                messages = [{"role": "user", "content": prompt}]
                inputs = self.tokenizer.apply_chat_template(
                    messages, return_tensors="pt", add_generation_prompt=True
                ).to(self.device)
            else:
                # Add padding=True to fix attention mask issues
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=self.model.config.max_position_embeddings
                    - current_max_new_tokens
                    - 10,
                )

                # Ensure attention_mask is explicitly set
                if "attention_mask" not in inputs:
                    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

                # Move all tensors to the correct device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            input_ids_len = inputs["input_ids"].shape[1]
            gen_metadata["input_tokens"] = input_ids_len

            generation_params = {
                "max_new_tokens": current_max_new_tokens,
                "do_sample": True if current_temperature > 0 else False,
                "temperature": current_temperature
                if current_temperature > 0
                else None,  # Temp 0 can mean greedy
                "top_p": current_top_p if current_temperature > 0 else None,
                "top_k": current_top_k if current_temperature > 0 else None,
                "repetition_penalty": current_repetition_penalty,
                "pad_token_id": self.tokenizer.eos_token_id,  # Use EOS for padding during generation
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            # Remove None params for generate call
            generation_params = {
                k: v for k, v in generation_params.items() if v is not None
            }

            with torch.no_grad():
                output_ids = self.model.generate(
                    inputs["input_ids"], **generation_params
                )

            # Decode only the newly generated tokens
            response_text = self.tokenizer.decode(
                output_ids[0][input_ids_len:], skip_special_tokens=True
            )
            gen_metadata["output_tokens"] = output_ids[0].shape[0] - input_ids_len

            return response_text.strip(), gen_metadata

        except Exception as e:
            logger_arc_llm.error(
                f"Error during LLM response generation: {e}", exc_info=True
            )
            gen_metadata["error"] = str(e)
            return "", gen_metadata  # Return empty string on error

    # Feature 3: Enhanced Advanced Prompt Construction for ARC
    def _construct_arc_prompt(
        self,
        train_pairs: List[Dict[str, List[List[int]]]],
        test_input: List[List[int]],
        dynamic_symbolic_context: str,
        task_id: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Constructs a detailed and structured prompt for an ARC task with enhanced pattern guidance."""
        prompt_parts = []

        effective_system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT_ARC
        prompt_parts.append(f"<<SYSTEM_INSTRUCTION>>\n{effective_system_prompt}\n")

        prompt_parts.append(f"<<TASK_ID>>\n{task_id}\n")

        if dynamic_symbolic_context:
            prompt_parts.append(f"<<SYMBOLIC_GUIDANCE>>\n{dynamic_symbolic_context}\n")
        else:
            prompt_parts.append(
                "<<SYMBOLIC_GUIDANCE>>\n[No specific symbolic guidance retrieved. Rely on examples and general ARC principles.]\n"
            )

        # Analyze patterns to provide more targeted guidance
        pattern_info = self._analyze_arc_patterns(train_pairs, test_input)
        pattern_guidance = ["<<PATTERN_ANALYSIS>>"]

        # Core pattern analysis guidance for all tasks
        pattern_guidance.extend(
            [
                "1. Analyze dimensions: Compare input and output grid sizes. Check for scaling factors.",
                "2. Detect repetition: Look for repeating blocks, rows, or columns in the output.",
                "3. Check for alternating patterns: Values might alternate in repeating sequences.",
                "4. Consider tiling: Output might be created by tiling the input grid with transformations.",
                "5. Look for symmetry: Check if transformations involve reflection or rotation.",
                "6. Analyze borders: Borders might follow special rules different from the interior.",
            ]
        )

        # Add customized guidance based on detected patterns
        if pattern_info.get("dimension_scale"):
            scale_info = pattern_info["dimension_scale"]
            pattern_guidance.append(
                f"7. IMPORTANT: The examples show a scaling pattern of approximately {scale_info['row_scale']}x rows "
                f"and {scale_info['col_scale']}x columns. Look closely at how the input grid expands to the output grid."
            )

        if pattern_info.get("has_repetition"):
            pattern_guidance.append(
                f"8. IMPORTANT: The examples show a {pattern_info.get('repetition_type', 'complex')} repetition pattern. "
                f"Pay close attention to how elements repeat in the output grid."
            )

        if pattern_info.get("edge_patterns", {}).get("has_edge_pattern"):
            pattern_guidance.append(
                "9. IMPORTANT: The examples show special treatment of edge values. "
                "Check if borders follow different rules than interior cells."
            )

        if pattern_info.get("detected_color_schemes"):
            schemes = ", ".join(pattern_info["detected_color_schemes"])
            pattern_guidance.append(
                f"10. The examples use a {schemes} color scheme. "
                f"Pay attention to the specific values used and maintain consistency."
            )

        values = pattern_info.get("unique_output_values")
        if values is not None:
            if isinstance(values, list) and len(values) <= 5:
                values_str = ", ".join(map(str, values))
                pattern_guidance.append(
                    f"11. The output grids only use these values: {values_str}. "
                    f"Your solution should likely use the same limited set of values."
                )

        prompt_parts.append("\n".join(pattern_guidance))

        prompt_parts.append(f"<<TRAINING_EXAMPLES ({len(train_pairs)} pairs)>>")
        for i, pair in enumerate(train_pairs):
            # Feature 6: Input Validation (Basic)
            if not isinstance(pair.get("input"), list) or not isinstance(
                pair.get("output"), list
            ):
                logger_arc_llm.warning(
                    f"Task {task_id}, Train Pair {i}: Invalid format. Skipping this pair in prompt."
                )
                continue

            prompt_parts.append(f"Example {i + 1} Input:\n{json.dumps(pair['input'])}")
            prompt_parts.append(
                f"Example {i + 1} Output:\n{json.dumps(pair['output'])}"
            )

            # Add dimension analysis to help model recognize patterns
            input_rows = len(pair["input"])
            input_cols = len(pair["input"][0]) if input_rows > 0 else 0
            output_rows = len(pair["output"])
            output_cols = len(pair["output"][0]) if output_rows > 0 else 0

            prompt_parts.append(
                f"Example {i + 1} Dimensions: Input {input_rows}x{input_cols} → Output {output_rows}x{output_cols} "
                f"(Scale factor: {output_rows / input_rows:.1f}x rows, {output_cols / input_cols:.1f}x columns)"
            )

        prompt_parts.append("")  # Extra newline for separation

        if not isinstance(test_input, list):
            logger_arc_llm.error(
                f"Task {task_id}: Test input is not a list: {test_input}. Using placeholder for prompt."
            )
            prompt_parts.append(
                f"<<TEST_INPUT>>\n{json.dumps([[-999]])} <<ERROR: INVALID TEST INPUT FORMAT>>\n"
            )
        else:
            prompt_parts.append(f"<<TEST_INPUT>>\n{json.dumps(test_input)}\n")

            # Add dimension analysis for test input to help model predict output dimensions
            test_rows = len(test_input)
            test_cols = len(test_input[0]) if test_rows > 0 else 0
            prompt_parts.append(f"Test Input Dimensions: {test_rows}x{test_cols}\n")

        prompt_parts.append(
            "<<OUTPUT_SPECIFICATION>>\n"
            "Based on the training examples, pattern analysis, and any symbolic guidance:\n"
            "1. Identify the transformation rule from input to output in the training examples\n"
            "2. Apply the same transformation rule to the test input\n"
            "3. Generate the complete output grid with correct dimensions\n"
            "4. Verify your output maintains the pattern consistency\n"
            "5. Respond ONLY with the JSON representation of the output grid (e.g., [[1, 2], [3, 4]])\n"
            "Do not include any other text, explanations, or markdown formatting like ```json ... ```."
        )
        return "\n".join(
            prompt_parts
        )  # Feature 4: Enhanced Precise JSON Grid Extraction

    def _extract_json_grid(
        self, raw_llm_output: str, task_id: str, test_case_idx: int
    ) -> Optional[List[List[int]]]:
        """Extracts a JSON grid from the LLM's raw output string with enhanced robustness."""
        if not raw_llm_output:
            logger_arc_llm.warning(
                f"Task {task_id} TC {test_case_idx}: Empty LLM output"
            )
            return None

        # Step 1: Clean the output by removing common formatting decorators
        cleaned_output = self._sanitize_json_output(raw_llm_output)
        original_cleaned = cleaned_output  # Keep original for fallback

        # Remove markdown code blocks - more comprehensive cleaning
        code_block_markers = [
            "```json",
            "```python",
            "```javascript",
            "```js",
            "```",
            "`",
            "json\n",
            "Output:",
            "Output grid:",
            "Predicted grid:",
        ]

        # First, check for common markdown block patterns and extract their content
        md_match = re.search(
            r"```(?:json|javascript|js|python)?\s*\n?(.*?)\n?```",
            cleaned_output,
            re.DOTALL,
        )
        if md_match:
            cleaned_output = md_match.group(1).strip()
            logger_arc_llm.debug(
                f"Task {task_id} TC {test_case_idx}: Extracted content from code block"
            )

        # Then, clean up any remaining markers
        for marker in code_block_markers:
            if cleaned_output.startswith(marker):
                cleaned_output = cleaned_output[len(marker) :].strip()
            if cleaned_output.endswith(marker):
                cleaned_output = cleaned_output[: -len(marker)].strip()

        # Remove any text before or after the grid-like structure
        grid_start = cleaned_output.find("[")
        if grid_start > 0:
            # Check if there's explanatory text before the JSON
            pre_text = cleaned_output[:grid_start].strip()
            if pre_text:
                logger_arc_llm.debug(
                    f"Task {task_id} TC {test_case_idx}: Removed prefix text: '{pre_text[:50]}...'"
                )
            cleaned_output = cleaned_output[grid_start:].strip()

        # Step 2: Try multiple methods to find the JSON grid, from most reliable to least
        potential_json_str = None

        # Method 1: Try to parse the whole content as JSON directly (most reliable)
        try:
            grid = json.loads(cleaned_output)
            if isinstance(grid, list) and all(isinstance(row, list) for row in grid):
                potential_json_str = cleaned_output
                logger_arc_llm.debug(
                    f"Task {task_id} TC {test_case_idx}: Successfully parsed full content as JSON grid"
                )
            else:
                logger_arc_llm.debug(
                    f"Task {task_id} TC {test_case_idx}: Full content is JSON but not a grid structure"
                )
        except json.JSONDecodeError:
            logger_arc_llm.debug(
                f"Task {task_id} TC {test_case_idx}: Full content is not valid JSON"
            )

        # Method 2: Enhanced regex for nested JSON arrays with better number handling
        if not potential_json_str:
            # More robust regex to handle floating point numbers, negative numbers, and various formats
            grid_pattern = r"(\[\s*\[\s*-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?)*\s*\](?:\s*,\s*\[\s*-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?)*\s*\])*\s*\])"
            grid_match = re.search(grid_pattern, cleaned_output)

            if grid_match:
                potential_json_str = grid_match.group(1)
                logger_arc_llm.debug(
                    f"Task {task_id} TC {test_case_idx}: Found grid using enhanced regex"
                )

        # Method 3: Try to extract content between outer brackets with proper balancing
        if not potential_json_str:
            try:
                bracket_count = 0
                start_idx = -1

                for i, char in enumerate(cleaned_output):
                    if char == "[":
                        bracket_count += 1
                        if bracket_count == 1:
                            start_idx = i
                    elif char == "]":
                        bracket_count -= 1
                        if bracket_count == 0 and start_idx != -1:
                            # Found a complete top-level array
                            candidate = cleaned_output[start_idx : i + 1]
                            # Try to parse it to verify it's valid JSON
                            try:
                                parsed = json.loads(candidate)
                                if isinstance(parsed, list) and (
                                    not parsed
                                    or all(isinstance(row, list) for row in parsed)
                                ):
                                    potential_json_str = candidate
                                    logger_arc_llm.debug(
                                        f"Task {task_id} TC {test_case_idx}: Found grid using bracket balancing"
                                    )
                                    break
                            except json.JSONDecodeError:
                                continue
            except Exception as e:
                logger_arc_llm.warning(
                    f"Task {task_id} TC {test_case_idx}: Error in bracket balancing extraction: {e}"
                )

        # Method 4: Fix common JSON formatting issues and try again
        if not potential_json_str:
            # Fix common issues like single quotes, trailing commas, etc.
            try:
                fixed_output = cleaned_output
                # Replace single quotes with double quotes
                fixed_output = re.sub(r"'([\[\],\d\s]+)'", r"\1", fixed_output)
                fixed_output = re.sub(r"'", '"', fixed_output)
                # Remove trailing commas in arrays
                fixed_output = re.sub(r",\s*]", "]", fixed_output)
                # Fix potential spaces between digits and decimal points
                fixed_output = re.sub(r"(\d+)\s+\.", r"\1.", fixed_output)

                # Try to parse the fixed output
                try:
                    grid = json.loads(fixed_output)
                    if isinstance(grid, list) and all(
                        isinstance(row, list) for row in grid
                    ):
                        potential_json_str = fixed_output
                        logger_arc_llm.debug(
                            f"Task {task_id} TC {test_case_idx}: Parsed JSON after fixing formatting issues"
                        )
                except json.JSONDecodeError:
                    pass
            except Exception as e:
                logger_arc_llm.warning(
                    f"Task {task_id} TC {test_case_idx}: Error fixing JSON formatting: {e}"
                )

        # Method 5: Last resort, try to find anything between first '[' and last ']'
        if not potential_json_str:
            start_idx = cleaned_output.find("[")
            end_idx = cleaned_output.rfind("]")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                potential_json_str = cleaned_output[start_idx : end_idx + 1]
                logger_arc_llm.debug(
                    f"Task {task_id} TC {test_case_idx}: Using last resort method (first [ to last ])"
                )

                # Additional attempt to fix the extracted string
                try:
                    # Try to balance brackets if necessary
                    open_brackets = potential_json_str.count("[")
                    close_brackets = potential_json_str.count("]")
                    if open_brackets > close_brackets:
                        potential_json_str += "]" * (open_brackets - close_brackets)
                        logger_arc_llm.debug(
                            f"Added {open_brackets - close_brackets} closing brackets to balance JSON"
                        )
                    elif close_brackets > open_brackets:
                        potential_json_str = (
                            "[" * (close_brackets - open_brackets) + potential_json_str
                        )
                        logger_arc_llm.debug(
                            f"Added {close_brackets - open_brackets} opening brackets to balance JSON"
                        )
                except Exception as e:
                    logger_arc_llm.warning(f"Error balancing brackets: {e}")

        # Method 6: Try parsing original content if all else failed
        if not potential_json_str:
            potential_json_str = original_cleaned
            logger_arc_llm.debug("Falling back to original cleaned content")

        # Step 3: Parse and validate the extracted JSON string
        if potential_json_str:
            try:
                # Clean up any residual white space or extraneous characters
                potential_json_str = potential_json_str.strip()

                # Try to parse the JSON
                grid = json.loads(potential_json_str)

                # Validate grid structure - properly handles different grid dimensions
                if isinstance(grid, list) and all(
                    isinstance(row, list) for row in grid
                ):
                    # Validate row lengths for grid consistency
                    if grid and len(grid) > 0:
                        first_row_len = len(grid[0])
                        if not all(len(row) == first_row_len for row in grid):
                            logger_arc_llm.warning(
                                f"Task {task_id} TC {test_case_idx}: Inconsistent row lengths in grid"
                            )
                            # Fix inconsistent row lengths by padding with zeros
                            max_row_len = max(len(row) for row in grid)
                            grid = [
                                row + [0] * (max_row_len - len(row)) for row in grid
                            ]
                            logger_arc_llm.debug(
                                f"Padded rows to ensure consistent length of {max_row_len}"
                            )

                    # Convert any non-integer values to integers if possible
                    try:
                        validated_grid = []
                        for row in grid:
                            validated_row = []
                            for cell in row:
                                if isinstance(cell, (int, float)):
                                    validated_row.append(int(cell))
                                elif (
                                    isinstance(cell, str)
                                    and cell.strip().replace("-", "").isdigit()
                                ):
                                    validated_row.append(int(cell.strip()))
                                else:
                                    # Use a default value for non-convertible items
                                    validated_row.append(0)
                                    logger_arc_llm.warning(
                                        f"Task {task_id} TC {test_case_idx}: Non-integer value in grid: {cell}, converted to 0"
                                    )
                            validated_grid.append(validated_row)

                        logger_arc_llm.debug(
                            f"Task {task_id} TC {test_case_idx}: Successfully extracted and validated grid: {str(validated_grid)[:100]}..."
                        )
                        return validated_grid
                    except Exception as e:
                        logger_arc_llm.warning(
                            f"Task {task_id} TC {test_case_idx}: Error validating grid: {e}"
                        )
                        return None
                else:
                    logger_arc_llm.warning(
                        f"Task {task_id} TC {test_case_idx}: Extracted JSON is not a valid grid: '{potential_json_str}'. Raw LLM out: '{raw_llm_output[:200]}...'"
                    )
                    return None
            except json.JSONDecodeError as e:
                logger_arc_llm.warning(
                    f"Task {task_id} TC {test_case_idx}: JSONDecodeError on extracted string '{potential_json_str}': {e}. Raw LLM out: '{raw_llm_output[:200]}...'"
                )
                return None

        # Final attempt: Try to create a grid by parsing individual rows
        try:
            # Look for patterns like [[1, 2, 3], [4, 5, 6]]
            row_pattern = r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]"
            found_rows = re.findall(row_pattern, cleaned_output)

            if found_rows:
                manual_grid = []
                for row_str in found_rows:
                    try:
                        row = [int(num.strip()) for num in row_str.split(",")]
                        manual_grid.append(row)
                    except ValueError:
                        continue

                if manual_grid and all(
                    len(row) == len(manual_grid[0]) for row in manual_grid
                ):
                    logger_arc_llm.debug(
                        f"Task {task_id} TC {test_case_idx}: Created grid by manually parsing rows"
                    )
                    return manual_grid
        except Exception as e:
            logger_arc_llm.warning(
                f"Task {task_id} TC {test_case_idx}: Error in manual row parsing: {e}"
            )

        logger_arc_llm.warning(
            f"Task {task_id} TC {test_case_idx}: No clear JSON grid found in raw output: '{raw_llm_output[:200]}...'"
        )
        return None

    def _sanitize_json_output(self, raw_output: str) -> str:
        """
        Sanitizes raw LLM output to improve chances of valid JSON extraction.
        This function applies various transformations to fix common JSON formatting issues.
        """
        if not raw_output:
            return raw_output

        sanitized = raw_output.strip()

        # Remove any markdown code block markers
        code_block_start = re.search(r"^```(?:json|python|javascript|js)?", sanitized)
        if code_block_start:
            sanitized = sanitized[code_block_start.end() :]

        code_block_end = re.search(r"```$", sanitized)
        if code_block_end:
            sanitized = sanitized[: code_block_end.start()]

        # Remove common textual prefixes
        prefixes = [
            "Here's the solution:",
            "Output:",
            "Output grid:",
            "Result:",
            "Answer:",
            "The output grid is:",
        ]
        for prefix in prefixes:
            if sanitized.startswith(prefix):
                sanitized = sanitized[len(prefix) :].strip()

        # Remove trailing explanations that might follow the JSON
        explanation_starts = [
            "Note:",
            "Explanation:",
            "\n\nThis",
            "\n\nIn this",
            "\n\nThe",
            "\n\nI",
            "\n\nAs",
        ]
        for exp_start in explanation_starts:
            exp_idx = sanitized.find(exp_start)
            if exp_idx > 0:
                sanitized = sanitized[:exp_idx].strip()

        # Fix single quotes to double quotes for JSON parsing
        sanitized = re.sub(
            r"'(\[.*?\])'", r"\1", sanitized
        )  # Remove quotes around arrays
        sanitized = re.sub(r"'", '"', sanitized)  # Replace remaining single quotes

        # Remove trailing commas in arrays
        sanitized = re.sub(r",\s*]", "]", sanitized)

        # Fix potential inconsistent spacing
        sanitized = re.sub(
            r"(\d+)\s+\.", r"\1.", sanitized
        )  # Fix spaces between digits and decimal points
        sanitized = re.sub(
            r"\[\s+", "[", sanitized
        )  # Normalize spaces after opening brackets
        sanitized = re.sub(
            r"\s+\]", "]", sanitized
        )  # Normalize spaces before closing brackets

        # Fix balanced brackets if needed
        open_count = sanitized.count("[")
        close_count = sanitized.count("]")

        if open_count > close_count:
            sanitized += "]" * (open_count - close_count)
        elif close_count > open_count and sanitized.startswith("]"):
            sanitized = "[" * (close_count - open_count) + sanitized

        # Ensure outer array brackets exist
        if not sanitized.startswith("["):
            sanitized = "[" + sanitized
        if not sanitized.endswith("]"):
            sanitized = sanitized + "]"

        return sanitized.strip()

    def solve_arc_task(
        self,
        train_pairs: List[Dict[str, List[List[int]]]],
        test_input: List[List[int]],
        base_symbolic_context_query: str,  # Changed from symbolic_context
        task_id: str = "unknown_task",
        test_case_idx: int = -1,
        verbose: bool = False,
        # LLM Generation Parameters (can override class defaults for this call)
        max_llm_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        num_retries: int = 2,  # Total attempts = 1 (initial) + num_retries
    ) -> Dict[str, Any]:
        """
        Solves an ARC task test case, integrating symbolic reasoning and LLM generation.
        Returns a detailed dictionary.
        """
        if verbose:  # Use passed verbose for this method's specific logging
            logger_arc_llm.setLevel(logging.DEBUG)
        else:  # Revert to instance/global default
            # This might require a more sophisticated logging setup to easily switch levels per call like this
            pass

        logger_arc_llm.info("🚀 Starting ARC symbolic evaluation...")
        logger_arc_llm.info(f"Solving Task: {task_id}, Test Case #{test_case_idx}")

        result_payload = {
            "predicted_grid": None,
            "llm_input_prompt": None,
            "llm_raw_output": None,
            "symbolic_context_used": None,
            "generation_metadata": None,
            "errors": [],
            "retries_attempted": 0,
            "pattern_analysis": {},
        }

        # Perform pre-analysis on the task patterns
        try:
            pattern_info = self._analyze_arc_patterns(train_pairs, test_input)
            result_payload["pattern_analysis"] = pattern_info

            # Enhance symbolic context with pattern insights
            pattern_context = []

            if pattern_info.get("dimension_scale"):
                scale_info = pattern_info["dimension_scale"]
                pattern_context.append(
                    f"Apply a dimension scaling transformation of approximately {scale_info['row_scale']}x rows "
                    f"and {scale_info['col_scale']}x columns. The output grid dimensions are likely to be "
                    f"around {scale_info['predicted_rows']}x{scale_info['predicted_cols']}."
                )

            if pattern_info.get("has_repetition"):
                pattern_context.append(
                    f"The output grid shows a repetitive pattern. "
                    f"The repetition appears to be {pattern_info['repetition_type']}."
                )

            if pattern_info.get("has_transformation"):
                pattern_context.append(
                    f"The transformation includes: {pattern_info['transformation_type']}."
                )

            # Add enhanced context to base symbolic context
            enhanced_symbolic_context = base_symbolic_context_query
            if pattern_context:
                enhanced_symbolic_context = (
                    base_symbolic_context_query + "\n\n" + "\n".join(pattern_context)
                )

        except Exception as e:
            logger_arc_llm.warning(
                f"Error during pattern pre-analysis: {e}", exc_info=True
            )
            enhanced_symbolic_context = base_symbolic_context_query
            result_payload["errors"].append(f"Pattern analysis error: {e}")

        # Feature 2: Robust Symbolic Middleware Integration
        dynamic_symbolic_context = (
            "[Symbolic middleware not configured or query is empty]"
        )
        if self.symbolic_middleware and enhanced_symbolic_context:
            if hasattr(self.symbolic_middleware, "analyze"):
                try:
                    logger_arc_llm.debug(
                        f"Task {task_id} TC {test_case_idx}: Querying symbolic middleware..."
                    )

                    analysis_result = self.symbolic_middleware.analyze(
                        enhanced_symbolic_context, top_k=3
                    )
                    contexts = analysis_result.get("retrieved_contexts", [])
                    if contexts:
                        dynamic_symbolic_context = "\n".join(contexts)
                    else:
                        dynamic_symbolic_context = (
                            "[No specific symbolic contexts retrieved by middleware.]"
                        )
                    result_payload["symbolic_analysis_metadata"] = {
                        k: v
                        for k, v in analysis_result.items()
                        if k != "retrieved_contexts"
                    }
                except Exception as e_sym:
                    logger_arc_llm.error(
                        f"Task {task_id} TC {test_case_idx}: Error during symbolic analysis: {e_sym}",
                        exc_info=verbose,
                    )
                    dynamic_symbolic_context = (
                        f"[Error during symbolic analysis: {e_sym}]"
                    )
                    result_payload["errors"].append(f"Symbolic analysis error: {e_sym}")
            else:
                logger_arc_llm.warning(
                    "Symbolic middleware instance provided but lacks an 'analyze' method."
                )
                dynamic_symbolic_context = enhanced_symbolic_context  # Use the enhanced query as context if analyze not found
        elif enhanced_symbolic_context:
            dynamic_symbolic_context = enhanced_symbolic_context  # Use enhanced if no middleware but query provided

        result_payload["symbolic_context_used"] = (
            dynamic_symbolic_context  # Construct the LLM prompt
        )
        llm_prompt = self._construct_arc_prompt(
            train_pairs,
            test_input,
            dynamic_symbolic_context,
            task_id,
            system_prompt=None,  # Could allow overriding DEFAULT_SYSTEM_PROMPT_ARC via solve_arc_task param
        )
        result_payload["llm_input_prompt"] = llm_prompt

        # Add debugging information about pattern analysis
        debug_info = {"task_complexity": "simple", "pattern_analysis_summary": []}

        if pattern_info.get("dimension_scale"):
            scale_info = pattern_info["dimension_scale"]
            debug_info["pattern_analysis_summary"].append(
                f"Dimension scaling detected: {scale_info['row_scale']}x rows, {scale_info['col_scale']}x columns"
            )
            debug_info["expected_output_dimensions"] = (
                f"{scale_info['predicted_rows']}x{scale_info['predicted_cols']}"
            )

        if pattern_info.get("has_repetition"):
            debug_info["pattern_analysis_summary"].append(
                f"Repetition pattern detected: {pattern_info.get('repetition_type', 'unknown type')}"
            )
            if pattern_info.get("repetition_type") in [
                "horizontal",
                "vertical",
                "block (tiling)",
            ]:
                debug_info["task_complexity"] = "moderate"

        if pattern_info.get("edge_patterns", {}).get("has_edge_pattern"):
            debug_info["pattern_analysis_summary"].append(
                f"Edge pattern detected: {pattern_info['edge_patterns'].get('edge_pattern_type', 'special edge treatment')}"
            )

        if pattern_info.get("output_complexity") == "complex":
            debug_info["task_complexity"] = "complex"
            debug_info["pattern_analysis_summary"].append(
                "Complex output pattern detected"
            )

        # Add the debug information to result payload
        result_payload["debug_info"] = debug_info

        if verbose:
            logger_arc_llm.debug(
                f"Task {task_id} TC {test_case_idx}: Generated LLM Prompt (first 300 chars):\n{llm_prompt[:300]}..."
            )
            logger_arc_llm.debug(
                f"Task {task_id} TC {test_case_idx}: Pattern analysis summary: {', '.join(debug_info['pattern_analysis_summary'])}"
            )

        # Feature 7: Retry Mechanism
        for attempt in range(num_retries + 1):
            result_payload["retries_attempted"] = attempt + 1
            logger_arc_llm.info(
                f"Task {task_id} TC {test_case_idx}: LLM generation attempt {attempt + 1}/{num_retries + 1}"
            )

            raw_llm_output, gen_meta = self._generate_llm_response(
                prompt=llm_prompt,
                max_new_tokens=max_llm_tokens,  # Pass per-call params
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
            result_payload["llm_raw_output"] = raw_llm_output
            result_payload["generation_metadata"] = gen_meta

            if gen_meta and gen_meta.get("error"):
                result_payload["errors"].append(
                    f"LLM generation error (attempt {attempt + 1}): {gen_meta['error']}"
                )
                if attempt < num_retries:
                    time.sleep(1 * (attempt + 1))  # Simple backoff
                    continue  # Retry
                else:
                    result_payload["predicted_grid"] = [[-11]]  # LLM Gen Error
                    return result_payload

            predicted_grid = self._extract_json_grid(
                raw_llm_output, task_id, test_case_idx
            )

            # Validate predicted grid against pattern analysis
            if predicted_grid is not None:
                # Check if dimensions match expected based on pattern analysis
                if (
                    "pattern_analysis" in result_payload
                    and "dimension_scale" in result_payload["pattern_analysis"]
                ):
                    predicted_rows = len(predicted_grid)
                    predicted_cols = len(predicted_grid[0]) if predicted_rows > 0 else 0
                    expected_rows = result_payload["pattern_analysis"][
                        "dimension_scale"
                    ].get("predicted_rows")
                    expected_cols = result_payload["pattern_analysis"][
                        "dimension_scale"
                    ].get("predicted_cols")

                    # If dimensions are off by more than 20%, it's likely wrong - try again with more specific guidance
                    if (
                        expected_rows
                        and expected_cols
                        and (
                            abs(predicted_rows - expected_rows) / expected_rows > 0.2
                            or abs(predicted_cols - expected_cols) / expected_cols > 0.2
                        )
                    ):
                        logger_arc_llm.warning(
                            f"Task {task_id} TC {test_case_idx}: Predicted grid dimensions ({predicted_rows}x{predicted_cols}) "
                            f"differ significantly from expected ({expected_rows}x{expected_cols}). Retrying..."
                        )
                        if attempt < num_retries:
                            # Add more specific dimension guidance for retry
                            llm_prompt += (
                                f"\n<<DIMENSION_CORRECTION>>\n"
                                f"The output grid should have dimensions close to {expected_rows}x{expected_cols}. "
                                f"Your current grid is {predicted_rows}x{predicted_cols}. Please adjust."
                            )
                            # Increase temp slightly for next attempt to encourage exploration
                            if temperature is not None and temperature > 0.05:
                                temperature = min(1.0, temperature + 0.1)
                            elif (
                                temperature is None and self.default_temperature > 0.05
                            ):
                                temperature = min(1.0, self.default_temperature + 0.1)
                            time.sleep(1 * (attempt + 1))
                            continue  # Retry with improved guidance

                # If we got here, the grid is acceptable or we're out of retries
                result_payload["predicted_grid"] = predicted_grid
                logger_arc_llm.info(
                    f"Task {task_id} TC {test_case_idx}: Successfully parsed grid on attempt {attempt + 1}."
                )
                # Clear errors if successful parse after retries
                if "errors" in result_payload and any(
                    "JSON" in e for e in result_payload["errors"]
                ):
                    result_payload["errors"] = [
                        e
                        for e in result_payload["errors"]
                        if "JSON" not in e and "grid" not in e
                    ]
                break  # Successful extraction, exit retry loop
            else:
                err_msg = f"Failed to extract valid JSON grid (attempt {attempt + 1}). Raw: '{raw_llm_output[:100]}...'"
                result_payload["errors"].append(err_msg)
                logger_arc_llm.warning(f"Task {task_id} TC {test_case_idx}: {err_msg}")
                if attempt < num_retries:
                    # Modify prompt slightly for retry? Or change temp?
                    llm_prompt += "\n<<RETRY_INSTRUCTION>>\nYour previous response was not a valid JSON grid. Please ensure your entire response is ONLY the JSON grid."
                    if (
                        temperature is not None and temperature > 0.05
                    ):  # Slightly increase temp for retry
                        temperature = min(1.0, temperature + 0.1)
                    elif temperature is None and self.default_temperature > 0.05:
                        temperature = min(1.0, self.default_temperature + 0.1)
                    time.sleep(1 * (attempt + 1))  # Simple backoff
                else:
                    result_payload["predicted_grid"] = [
                        [-10]
                    ]  # JSON Extraction Error after all retries

        if result_payload["predicted_grid"] is None:  # Should be set by now
            result_payload["predicted_grid"] = [[-12]]  # Fallback for logic error

        if verbose:  # Revert logger level if changed
            logger_arc_llm.setLevel(logging.INFO)  # Or previous level

        return result_payload

    def build_prompt(
        self,
        query: str,
        context: str,
        scaffold: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Builds a generic prompt with query, context, and optionally, a scaffold and history.
        This is less ARC-specific than _construct_arc_prompt.
        """
        prompt_parts = []
        if scaffold:
            prompt_parts.append(f"<<REASONING_SCAFFOLD>>\n{scaffold}\n")
        if context:
            prompt_parts.append(f"<<PROVIDED_CONTEXT>>\n{context}\n")
        if history:
            prompt_parts.append("<<CONVERSATION_HISTORY (last few turns)>>")
            for h_entry in history[-3:]:  # Show last 3 history entries
                role = h_entry.get("role", "unknown").upper()
                content = h_entry.get("content", "")
                prompt_parts.append(f"{role}: {content}")
            prompt_parts.append("")
        prompt_parts.append(f"<<CURRENT_QUERY>>\n{query}")
        return "\n".join(prompt_parts)

    def select_model(
        self, task_requirements: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Confirms current model and effective generation parameters.
        """
        tr = task_requirements or {}
        return {
            "model_name": getattr(self.model.config, "name_or_path", self.model_path),
            "provider": "local_transformers",
            "device": str(self.device),
            "effective_temperature": tr.get("temperature", self.default_temperature),
            "effective_top_p": tr.get("top_p", self.default_top_p),
            "effective_top_k": tr.get("top_k", self.default_top_k),
            "effective_max_tokens": tr.get("max_tokens", self.default_max_tokens),
            "effective_repetition_penalty": tr.get(
                "repetition_penalty", self.default_repetition_penalty
            ),
        }

    def generate_response(
        self,
        messages: Union[str, List[Dict[str, str]]],
        task_requirements: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        system_prompt_override: Optional[str] = None,
        use_global_system_prompt: bool = True,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Generates a response from the LLM.

        This implementation handles both:
        1. The BaseLlmInterface abstract method signature (messages as list of dicts)
        2. Direct string input (for simple testing with a prompt parameter)

        Args:
            messages: Either a string prompt or a list of message dictionaries
            task_requirements: Optional requirements for the task (used for parameter settings)
            temperature: Optional temperature parameter for controlling randomness
            system_prompt_override: Optional system prompt to use instead of the default
            use_global_system_prompt: Whether to use the global system prompt

        Returns:
            A tuple containing:
            - response_text: The generated text response
            - model_info: Information about the model used (name, provider, etc.)
            - response_metadata: Additional metadata about the response
        """
        # Handle both string prompts and message lists for flexibility
        prompt = ""
        if isinstance(messages, str):
            # Direct string prompt (for test_arc_llm.py compatibility)
            prompt = messages
            logger_arc_llm.debug(
                f"generate_response called with direct string prompt: '{prompt[:50]}...'"
            )
        else:
            # Standard message list format from BaseLlmInterface
            # Extract the last user message as our prompt
            user_messages = [m for m in messages if m.get("role", "") == "user"]
            if user_messages:
                prompt = user_messages[-1].get("content", "")

                # If we need a more structured prompt with chat history
                if len(messages) > 1:
                    # Format history for a more comprehensive prompt if needed
                    history = [
                        {"role": m.get("role", ""), "content": m.get("content", "")}
                        for m in messages[:-1]
                    ]  # All but the last message
                    context = ""  # No specific context from messages

                    # Use the build_prompt method to create a structured prompt
                    prompt = self.build_prompt(
                        query=prompt, context=context, history=history
                    )

            logger_arc_llm.debug(
                f"generate_response called with {len(messages)} messages, extracted prompt: '{prompt[:50]}...'"
            )

        # If no prompt could be extracted, return an error
        if not prompt:
            error_msg = "No valid prompt could be extracted from the messages"
            logger_arc_llm.warning(error_msg)
            return (
                None,
                self.select_model(task_requirements),
                {"error": error_msg},
            )  # Get effective parameters from task_requirements or use defaults
        tr = task_requirements or {}
        current_temperature = (
            temperature
            if temperature is not None
            else tr.get("temperature", self.default_temperature)
        )
        current_max_tokens = tr.get("max_tokens", self.default_max_tokens)
        current_top_p = tr.get("top_p", self.default_top_p)
        current_top_k = tr.get("top_k", self.default_top_k)
        current_repetition_penalty = tr.get(
            "repetition_penalty", self.default_repetition_penalty
        )
        # Apply relevant sigils for ARC tasks if the prompt appears to be an ARC task
        # This enhances pattern recognition and abstract reasoning capabilities
        if (
            "grid" in prompt.lower()
            or "ARC" in prompt
            or "abstraction" in prompt.lower()
            or "reasoning corpus" in prompt.lower()
        ):
            logger_arc_llm.info(
                "ARC task detected - selecting appropriate sigils for the task"
            )

            applied_sigils = []

            # Always apply CATENGINE as the primary sigil for ARC tasks
            cat_engine_instructions = """
                    🜛CATENGINE [MODE: ACTIVE]
                    Input categorization: ARC reasoning task
                    Task context: Abstract pattern recognition and rule inference
                    Focus areas: 
                    - Extract key features from input/output grids
                    - Identify transformation patterns between examples
                    - Apply pattern recognition to solve test cases
                    - Prioritize rule consistency across all examples
                    """
            if "🜛CATENGINE" not in prompt:
                prompt = cat_engine_instructions + "\n\n" + prompt
                applied_sigils.append(
                    {
                        "sigil": "🜛CATENGINE",
                        "purpose": "Enhanced pattern recognition for ARC tasks",
                        "effect": "Improved abstract reasoning and feature extraction",
                    }
                )

            # Apply JIGSAW_ASSEMBLER for pattern completion tasks
            if (
                "complete" in prompt.lower()
                or "missing" in prompt.lower()
                or "fill" in prompt.lower()
                or "partial" in prompt.lower()
            ):
                jigsaw_instructions = """
                    🧩JIGSAW_ASSEMBLER [MODE: ACTIVE]
                    Task focus: Pattern completion and coherent structure assembly
                    Focus areas:
                    - Identify edges and connection points between fragments
                    - Recognize partial patterns and predict their completion
                    - Build coherent structural representations from incomplete data
                    - Test multiple configurations to find optimal fit
                    """
                if "🧩JIGSAW_ASSEMBLER" not in prompt:
                    prompt = jigsaw_instructions + "\n\n" + prompt
                    applied_sigils.append(
                        {
                            "sigil": "🧩JIGSAW_ASSEMBLER",
                            "purpose": "Pattern completion for fragmented or partial information",
                            "effect": "Enhanced ability to reconstruct complete patterns from partial examples",
                        }
                    )

            # Apply INSIGHT_NUCLEATOR for tasks requiring intuitive leaps
            if (
                "complex" in prompt.lower()
                or "difficult" in prompt.lower()
                or "creative" in prompt.lower()
                or "novel" in prompt.lower()
            ):
                insight_instructions = """
                        💡INSIGHT_NUCLEATOR [MODE: ACTIVE]
                        Task focus: Facilitating breakthrough insights for complex patterns
                        Focus areas:
                        - Create conditions for "Aha!" moments in pattern recognition
                        - Connect disparate concepts into unified transformation rules
                        - Overcome mental blocks and facilitate intuitive leaps
                        - Crystallize implicit patterns into explicit rules
                        """
                if "💡INSIGHT_NUCLEATOR" not in prompt:
                    prompt = insight_instructions + "\n\n" + prompt
                    applied_sigils.append(
                        {
                            "sigil": "💡INSIGHT_NUCLEATOR",
                            "purpose": "Breakthrough insight generation for complex ARC tasks",
                            "effect": "Facilitated intuitive leaps in pattern recognition",
                        }
                    )
            # Apply TREETHOUGHT for complex multi-step reasoning tasks
            if (
                "multi-step" in prompt.lower()
                or "sequence" in prompt.lower()
                or "complex" in prompt.lower()
                or "reasoning" in prompt.lower()
                or "grid" in prompt.lower()
            ):
                tree_instructions = """
                        🌳TREETHOUGHT [MODE: ACTIVE]
                        Task focus: Structured exploration of reasoning paths
                        Focus areas:
                        - Generate and evaluate multiple potential transformation rules
                        - Explore branching paths of logical inference
                        - Backtrack from unsuccessful approaches without getting stuck
                        - Maintain working memory of attempted solutions
                        - Systematically enumerate possible grid transformation patterns (repetition, reflection, rotation)
                        - Consider combinations of transformations when solving complex grid tasks
                        """
                if "🌳TREETHOUGHT" not in prompt:
                    prompt = tree_instructions + "\n\n" + prompt
                    applied_sigils.append(
                        {
                            "sigil": "🌳TREETHOUGHT",
                            "purpose": "Multi-path reasoning for complex transformation rules",
                            "effect": "Structured exploration of solution alternatives",
                        }
                    )

            # Apply PATTERN_BREEDER for grid-based pattern tasks (specifically for ARC)
            if (
                "grid" in prompt.lower()
                or "pattern" in prompt.lower()
                or "ARC" in prompt
            ):
                pattern_instructions = """
                        🔄PATTERN_BREEDER [MODE: ACTIVE]
                        Task focus: Advanced pattern recognition and generation
                        Focus areas:
                        - Detect repeating and alternating patterns in grid structures
                        - Recognize scaling transformations between input and output grids
                        - Identify pattern tiling and propagation across dimensions
                        - Predict continuation of partial patterns
                        - Maintain pattern consistency across multiple grid scales
                        - Apply transformation rules consistently across the entire grid
                        """
                prompt = pattern_instructions + "\n\n" + prompt
                applied_sigils.append(
                    {
                        "sigil": "🔄PATTERN_BREEDER",
                        "purpose": "Enhanced pattern generation and scaling for ARC grid tasks",
                        "effect": "Improved recognition of repeating patterns and grid transformations",
                    }
                )

            # Apply ANOMALY_SEEKER for unusual or exception-based patterns
            if (
                "unusual" in prompt.lower()
                or "exception" in prompt.lower()
                or "anomaly" in prompt.lower()
                or "unique" in prompt.lower()
                or "edge case" in prompt.lower()
            ):
                anomaly_instructions = """
                        💥ANOMALY_SEEKER [MODE: ACTIVE]
                        Task focus: Identifying exceptional patterns and rule-breaking elements
                        Focus areas:
                        - Detect outliers and pattern-breaking elements in grids
                        - Identify special case rules that apply only in specific contexts
                        - Notice subtle deviations from the main transformation pattern
                        - Recognize when exceptions themselves follow meta-patterns
                        """
                if "💥ANOMALY_SEEKER" not in prompt:
                    prompt = anomaly_instructions + "\n\n" + prompt
                    applied_sigils.append(
                        {
                            "sigil": "💥ANOMALY_SEEKER",
                            "purpose": "Detection of unusual patterns and exceptions in ARC tasks",
                            "effect": "Enhanced sensitivity to outliers and special cases",
                        }
                    )

            # For ARC tasks, slightly lower temperature to improve focus on patterns
            if current_temperature > 0.05:
                current_temperature = max(0.05, current_temperature * 0.8)
                logger_arc_llm.debug(
                    f"Adjusted temperature to {current_temperature} for ARC task"
                )

            # Log applied sigils
            if applied_sigils:
                logger_arc_llm.info(
                    f"Applied {len(applied_sigils)} sigils to enhance ARC task performance"
                )
                for sigil in applied_sigils:
                    logger_arc_llm.debug(
                        f"Applied sigil: {sigil['sigil']} - {sigil['purpose']}"
                    )
        # Initialize generation metadata to track sigils            generation_metadata = {"sigils_applied": applied_sigils}
        else:
            # Initialize empty metadata if no sigils were applied
            generation_metadata = {}

        # Generate the response using our internal method
        response_text, gen_metadata = self._generate_llm_response(
            prompt=prompt,
            max_new_tokens=current_max_tokens,
            temperature=current_temperature,
            top_p=current_top_p,
            top_k=current_top_k,
            repetition_penalty=current_repetition_penalty,
        )

        # Merge the generation metadata from _generate_llm_response with our existing metadata
        if gen_metadata:
            generation_metadata.update(gen_metadata)

        # Return the formatted response tuple
        return response_text, self.select_model(task_requirements), generation_metadata

    def _analyze_arc_patterns(
        self, train_pairs: List[Dict[str, List[List[int]]]], test_input: List[List[int]]
    ) -> Dict[str, Any]:
        """
        Analyzes pattern relationships between input and output grids in training examples.
        Detects common transformations like repetition, scaling, reflection, etc.
        Used to guide the LLM's reasoning process.
        """
        pattern_info = {
            "dimension_scale": None,
            "has_repetition": False,
            "repetition_type": None,
            "has_transformation": False,
            "transformation_type": [],
            "edge_patterns": {},
            "progression_patterns": {},
            "detected_color_schemes": [],
            "output_complexity": "unknown",
        }

        # Skip analysis if no valid training pairs
        if (
            not train_pairs
            or not isinstance(train_pairs, list)
            or len(train_pairs) == 0
        ):
            return pattern_info

        try:
            # Calculate dimension scaling patterns
            row_scales = []
            col_scales = []
            output_complexity_scores = []
            unique_values_in_outputs = set()
            consistent_output_values = (
                None  # Will track values that appear in ALL outputs
            )

            for pair in train_pairs:
                if (
                    not isinstance(pair, dict)
                    or "input" not in pair
                    or "output" not in pair
                ):
                    continue

                input_grid = pair["input"]
                output_grid = pair["output"]

                if not isinstance(input_grid, list) or not isinstance(
                    output_grid, list
                ):
                    continue

                input_rows = len(input_grid)
                input_cols = len(input_grid[0]) if input_rows > 0 else 0
                output_rows = len(output_grid)
                output_cols = len(output_grid[0]) if output_rows > 0 else 0

                if input_rows > 0 and output_rows > 0:
                    row_scales.append(output_rows / input_rows)
                if input_cols > 0 and output_cols > 0:
                    col_scales.append(output_cols / input_cols)

                # Analyze output complexity
                if output_grid:
                    # Count unique values in output
                    output_values = set()
                    for row in output_grid:
                        for cell in row:
                            output_values.add(cell)
                            unique_values_in_outputs.add(cell)

                    # Track consistent output values across examples
                    if consistent_output_values is None:
                        consistent_output_values = output_values
                    else:
                        consistent_output_values = (
                            consistent_output_values.intersection(output_values)
                        )

                    # Calculate a simple complexity score based on grid size and unique values
                    complexity = len(output_values) * (output_rows * output_cols) / 10
                    output_complexity_scores.append(complexity)

                # Check for edge patterns in the output
                edge_pattern = self._detect_edge_patterns(output_grid)
                if (
                    edge_pattern.get("has_edge_pattern")
                    and not pattern_info["edge_patterns"]
                ):
                    pattern_info["edge_patterns"] = edge_pattern

            # Set complexity level based on average complexity score
            if output_complexity_scores:
                avg_complexity = sum(output_complexity_scores) / len(
                    output_complexity_scores
                )
                if avg_complexity < 5:
                    pattern_info["output_complexity"] = "simple"
                elif avg_complexity < 15:
                    pattern_info["output_complexity"] = "moderate"
                else:
                    pattern_info["output_complexity"] = "complex"

            # Store consistent values that appear in all output grids
            if consistent_output_values:
                pattern_info["consistent_output_values"] = sorted(
                    list(consistent_output_values)
                )

            # Store unique values found across all outputs
            pattern_info["unique_output_values"] = sorted(
                list(unique_values_in_outputs)
            )

            # Calculate average scaling factors
            if row_scales and col_scales:
                avg_row_scale = sum(row_scales) / len(row_scales)
                avg_col_scale = sum(col_scales) / len(col_scales)

                # Round to nearest integer if close to integer value (within 0.05)
                if abs(round(avg_row_scale) - avg_row_scale) < 0.05:
                    avg_row_scale = round(avg_row_scale)
                if abs(round(avg_col_scale) - avg_col_scale) < 0.05:
                    avg_col_scale = round(avg_col_scale)

                # Predict output dimensions for test input
                test_rows = len(test_input)
                test_cols = len(test_input[0]) if test_rows > 0 else 0

                predicted_rows = int(test_rows * avg_row_scale)
                predicted_cols = int(test_cols * avg_col_scale)

                pattern_info["dimension_scale"] = {
                    "row_scale": avg_row_scale,
                    "col_scale": avg_col_scale,
                    "predicted_rows": predicted_rows,
                    "predicted_cols": predicted_cols,
                }

            # Check for repetition patterns
            for pair in train_pairs:
                input_grid = pair.get("input", [])
                output_grid = pair.get("output", [])

                if not input_grid or not output_grid:
                    continue

                # Check if output is a repetition of input
                if self._has_grid_repetition(input_grid, output_grid):
                    pattern_info["has_repetition"] = True

                    # Determine repetition type
                    if self._is_horizontal_repetition(input_grid, output_grid):
                        pattern_info["repetition_type"] = "horizontal"
                    elif self._is_vertical_repetition(input_grid, output_grid):
                        pattern_info["repetition_type"] = "vertical"
                    elif self._is_block_repetition(input_grid, output_grid):
                        pattern_info["repetition_type"] = "block (tiling)"
                    else:
                        pattern_info["repetition_type"] = "complex"

                # Check for transformations
                transformations = []

                if self._has_reflection(input_grid, output_grid):
                    transformations.append("reflection")

                if self._has_rotation(input_grid, output_grid):
                    transformations.append("rotation")

                if self._has_alternating_pattern(output_grid):
                    transformations.append("alternating values")

                if transformations:
                    pattern_info["has_transformation"] = True
                    pattern_info["transformation_type"] = transformations

            # Detect patterns across multiple examples (progression patterns)
            progression_patterns = self._detect_grid_progression(train_pairs)
            if progression_patterns.get("has_progression"):
                pattern_info["progression_patterns"] = progression_patterns

            # Detect common color schemes in ARC tasks
            if unique_values_in_outputs:
                # Check for binary pattern (only 0s and 1s)
                if unique_values_in_outputs == {0, 1} or unique_values_in_outputs == {
                    1,
                    0,
                }:
                    pattern_info["detected_color_schemes"].append("binary")

                # Check for small range with consistent steps (like 1,2,3,4)
                sorted_values = sorted(unique_values_in_outputs)
                if (
                    len(sorted_values) > 1
                    and sorted_values[0] >= 0
                    and sorted_values[-1] <= 9
                ):
                    is_sequential = all(
                        sorted_values[i + 1] - sorted_values[i] == 1
                        for i in range(len(sorted_values) - 1)
                    )
                    if is_sequential:
                        pattern_info["detected_color_schemes"].append("sequential")

        except Exception as e:
            logger_arc_llm.warning(f"Error in pattern analysis: {e}", exc_info=True)

        return pattern_info

    def _has_grid_repetition(
        self, input_grid: List[List[int]], output_grid: List[List[int]]
    ) -> bool:
        """Check if the output grid contains repeated patterns of the input grid."""
        # Simple heuristic: check if output is significantly larger than input
        input_size = len(input_grid) * (len(input_grid[0]) if input_grid else 0)
        output_size = len(output_grid) * (len(output_grid[0]) if output_grid else 0)

        return input_size > 0 and output_size > input_size * 1.5

    def _is_horizontal_repetition(
        self, input_grid: List[List[int]], output_grid: List[List[int]]
    ) -> bool:
        """Check if the output grid repeats the input horizontally."""
        if not input_grid or not output_grid:
            return False

        input_cols = len(input_grid[0]) if input_grid and input_grid[0] else 0
        output_cols = len(output_grid[0]) if output_grid and output_grid[0] else 0

        # Check if output columns are a multiple of input columns
        return (
            input_cols > 0
            and output_cols % input_cols == 0
            and output_cols > input_cols
        )

    def _is_vertical_repetition(
        self, input_grid: List[List[int]], output_grid: List[List[int]]
    ) -> bool:
        """Check if the output grid repeats the input vertically."""
        if not input_grid or not output_grid:
            return False

        input_rows = len(input_grid)
        output_rows = len(output_grid)

        # Check if output rows are a multiple of input rows
        return (
            input_rows > 0
            and output_rows % input_rows == 0
            and output_rows > input_rows
        )

    def _is_block_repetition(
        self, input_grid: List[List[int]], output_grid: List[List[int]]
    ) -> bool:
        """Check if the output grid is a tiled repetition of the input (both rows and columns)."""
        return self._is_horizontal_repetition(
            input_grid, output_grid
        ) and self._is_vertical_repetition(input_grid, output_grid)

    def _has_reflection(
        self, input_grid: List[List[int]], output_grid: List[List[int]]
    ) -> bool:
        """
        Check for reflection patterns between input and output grids.
        Detects horizontal and vertical reflections.
        """
        if not input_grid or not output_grid:
            return False

        # Quick dimension check - if dimensions are different it might be a scaled reflection
        # For simplicity in this implementation, we'll focus on same-dimension cases
        if len(input_grid) != len(output_grid) or len(input_grid[0]) != len(
            output_grid[0]
        ):
            return False

        # Check for horizontal reflection (flipped around horizontal axis)
        horizontally_reflected = True
        for i in range(len(input_grid)):
            if input_grid[i] != output_grid[len(output_grid) - 1 - i]:
                horizontally_reflected = False
                break

        if horizontally_reflected:
            return True

        # Check for vertical reflection (flipped around vertical axis)
        vertically_reflected = True
        for r in range(len(input_grid)):
            for c in range(len(input_grid[0])):
                if input_grid[r][c] != output_grid[r][len(input_grid[0]) - 1 - c]:
                    vertically_reflected = False
                    break
            if not vertically_reflected:
                break

        return vertically_reflected

    def _has_rotation(
        self, input_grid: List[List[int]], output_grid: List[List[int]]
    ) -> bool:
        """
        Check for rotation patterns between input and output grids.
        Detects 90°, 180°, and 270° rotations.
        """
        if not input_grid or not output_grid:
            return False

        # For 90° and 270° rotations, width becomes height and vice versa
        # For simplicity, we focus on cases where this transformation makes sense
        input_rows, input_cols = (
            len(input_grid),
            len(input_grid[0]) if input_grid else 0,
        )
        output_rows, output_cols = (
            len(output_grid),
            len(output_grid[0]) if output_grid else 0,
        )

        if not (
            input_rows == output_cols
            and input_cols == output_rows  # For 90° and 270° rotations
            or input_rows == output_rows
            and input_cols == output_cols
        ):  # For 180° rotation
            return False

        # Check 90° clockwise rotation
        rotated_90 = True
        if input_rows == output_cols and input_cols == output_rows:
            for r in range(input_rows):
                for c in range(input_cols):
                    if input_grid[r][c] != output_grid[c][input_rows - 1 - r]:
                        rotated_90 = False
                        break
                if not rotated_90:
                    break
            if rotated_90:
                return True

        # Check 180° rotation
        rotated_180 = True
        if input_rows == output_rows and input_cols == output_cols:
            for r in range(input_rows):
                for c in range(input_cols):
                    if (
                        input_grid[r][c]
                        != output_grid[input_rows - 1 - r][input_cols - 1 - c]
                    ):
                        rotated_180 = False
                        break
                if not rotated_180:
                    break
            if rotated_180:
                return True

        # Check 270° clockwise rotation (or 90° counter-clockwise)
        rotated_270 = True
        if input_rows == output_cols and input_cols == output_rows:
            for r in range(input_rows):
                for c in range(input_cols):
                    if input_grid[r][c] != output_grid[output_rows - 1 - c][r]:
                        rotated_270 = False
                        break
                if not rotated_270:
                    break
            if rotated_270:
                return True

        return False

    def _has_alternating_pattern(self, grid: List[List[int]]) -> bool:
        """Check if the grid has alternating values in rows or columns."""
        if not grid or len(grid) < 2:
            return False

        # Check for alternating rows
        for i in range(2, len(grid), 2):
            if grid[i] != grid[i - 2]:
                break
        else:
            # If we got here, every other row repeats
            for i in range(3, len(grid), 2):
                if i < len(grid) and grid[i] != grid[i - 2]:
                    break
            else:
                # Both sets of alternating rows are consistent
                return True

        # Check for alternating columns (simplified)
        if not grid[0]:
            return False

        # Just check the first few values in each column
        for col in range(len(grid[0])):
            alternating = True
            for row in range(2, min(len(grid), 6), 2):
                if (
                    col < len(grid[row])
                    and col < len(grid[row - 2])
                    and grid[row][col] != grid[row - 2][col]
                ):
                    alternating = False
                    break
            if alternating:
                return True

        return False

    def _detect_grid_progression(
        self, train_pairs: List[Dict[str, List[List[int]]]]
    ) -> Dict[str, Any]:
        """
        Detect progression patterns across multiple training examples.
        This helps identify how patterns evolve across multiple examples.
        """
        progression_info = {
            "has_progression": False,
            "progression_type": None,
            "consistent_values": [],
        }

        if not train_pairs or len(train_pairs) < 2:
            return progression_info

        # Check if values consistently increase or decrease across examples
        try:
            # Extract unique values from each output grid
            output_values = []
            for pair in train_pairs:
                output_grid = pair.get("output", [])
                if not output_grid:
                    continue

                # Get unique values in this output
                unique_vals = set()
                for row in output_grid:
                    for cell in row:
                        unique_vals.add(cell)
                output_values.append(sorted(list(unique_vals)))

            # Check for consistent value progression
            if len(output_values) >= 2:
                # Check if max value increases consistently
                max_values = [max(vals) if vals else 0 for vals in output_values]
                if all(
                    max_values[i] < max_values[i + 1]
                    for i in range(len(max_values) - 1)
                ):
                    progression_info["has_progression"] = True
                    progression_info["progression_type"] = "increasing_max_value"
                    progression_info["consistent_values"] = max_values
                elif all(
                    max_values[i] > max_values[i + 1]
                    for i in range(len(max_values) - 1)
                ):
                    progression_info["has_progression"] = True
                    progression_info["progression_type"] = "decreasing_max_value"
                    progression_info["consistent_values"] = max_values

                # Check for consistent value sets
                common_values = set(output_values[0])
                for vals in output_values[1:]:
                    common_values = common_values.intersection(set(vals))

                if common_values:
                    progression_info["common_values_across_examples"] = sorted(
                        list(common_values)
                    )

        except Exception as e:
            logger_arc_llm.warning(f"Error in progression detection: {e}")

        return progression_info

    def _detect_edge_patterns(self, grid: List[List[int]]) -> Dict[str, Any]:
        """
        Detect patterns specific to edges of the grid, which are common in ARC tasks.
        """
        if not grid or not grid[0]:
            return {"has_edge_pattern": False}

        edge_info = {
            "has_edge_pattern": False,
            "edge_pattern_type": None,
            "edge_values": {},
        }

        try:
            rows = len(grid)
            cols = len(grid[0])

            # Extract edge values
            top_edge = grid[0]
            bottom_edge = grid[-1]
            left_edge = [grid[r][0] for r in range(rows)]
            right_edge = [grid[r][-1] for r in range(rows)]

            # Check if edges have different values from interior
            interior_values = set()
            edge_values = set(top_edge + bottom_edge + left_edge + right_edge)

            for r in range(1, rows - 1):
                for c in range(1, cols - 1):
                    interior_values.add(grid[r][c])

            # If edges have distinct values from interior, it's likely an edge pattern
            if (
                edge_values
                and interior_values
                and not edge_values.intersection(interior_values)
            ):
                edge_info["has_edge_pattern"] = True
                edge_info["edge_pattern_type"] = "distinct_edge_values"
                edge_info["edge_values"] = {
                    "top": top_edge,
                    "bottom": bottom_edge,
                    "left": left_edge,
                    "right": right_edge,
                }
                edge_info["interior_values"] = sorted(list(interior_values))

            # Check if edges are consistent (same value)
            edge_consistencies = {
                "top_consistent": len(set(top_edge)) == 1 if top_edge else False,
                "bottom_consistent": len(set(bottom_edge)) == 1
                if bottom_edge
                else False,
                "left_consistent": len(set(left_edge)) == 1 if left_edge else False,
                "right_consistent": len(set(right_edge)) == 1 if right_edge else False,
            }

            if any(edge_consistencies.values()):
                edge_info["has_edge_pattern"] = True
                if not edge_info.get("edge_pattern_type"):
                    edge_info["edge_pattern_type"] = "consistent_edges"
                edge_info["edge_consistencies"] = edge_consistencies

        except Exception as e:
            logger_arc_llm.warning(f"Error in edge pattern detection: {e}")

        return edge_info

    def solve_arc_task_with_enhanced_handling(
        self,
        train_pairs,
        test_input_grid,
        base_symbolic_query_for_task,
        task_id,
        tc_idx,
        verbose_llm_behavior=False,
        max_tokens_per_llm_prediction=None,
        temperature=0.0,
        num_retries=2,
        logger=None,
    ):
        """
        Enhanced wrapper for the ARCAwareLLMInterface.solve_arc_task method with improved
        result processing, error handling, and logging.

        Returns:
            dict: A dictionary containing the predicted grid, status, and metadata
        """
        # Use default logger if not provided
        if logger is None:
            import logging

            logger = logging.getLogger("arc_task_solver")

        result = {
            "predicted_grid": None,
            "status": "failure",
            "error": None,
            "pattern_insights": {},
            "retries": 0,
            "details": {},
        }

        logger.info(
            f"Starting ARC task solution: Task ID {task_id}, Test Case #{tc_idx + 1}"
        )

        try:
            # Call the ARCAwareLLMInterface to solve the task
            prediction_result_dict = self.solve_arc_task(
                train_pairs=train_pairs,
                test_input=test_input_grid,
                base_symbolic_context_query=base_symbolic_query_for_task,
                task_id=task_id,
                test_case_idx=tc_idx + 1,
                verbose=verbose_llm_behavior,
                max_llm_tokens=max_tokens_per_llm_prediction,
                temperature=temperature,
                num_retries=num_retries,
            )

            # Extract and validate the predicted grid
            predicted_grid = prediction_result_dict.get("predicted_grid")
            result["predicted_grid"] = predicted_grid
            result["retries"] = prediction_result_dict.get("retries_attempted", 0)

            # Check for special error codes in the grid
            if predicted_grid is not None:
                if predicted_grid == [[-10]]:
                    result["error"] = "JSON extraction error after all retries"
                    logger.error(
                        f"Task {task_id}: Failed to extract valid JSON grid after {num_retries + 1} attempts"
                    )
                elif predicted_grid == [[-11]]:
                    result["error"] = "LLM generation error"
                    logger.error(f"Task {task_id}: LLM generation error encountered")
                elif predicted_grid == [[-12]]:
                    result["error"] = "Logic error in grid processing"
                    logger.error(f"Task {task_id}: Logic error in grid processing")
                else:
                    result["status"] = "success"
                    grid_rows = len(predicted_grid)
                    grid_cols = len(predicted_grid[0]) if grid_rows > 0 else 0
                    logger.info(
                        f"Task {task_id}: Successfully predicted grid with dimensions {grid_rows}x{grid_cols}"
                    )

            # Process errors from the result dictionary
            errors = prediction_result_dict.get("errors", [])
            if errors:
                error_msg = "; ".join(errors)
                result["error"] = (
                    error_msg
                    if not result["error"]
                    else f"{result['error']}; {error_msg}"
                )
                logger.warning(f"Task {task_id}: Errors encountered: {error_msg}")

            # Extract pattern analysis insights
            if "pattern_analysis" in prediction_result_dict:
                pattern_analysis = prediction_result_dict["pattern_analysis"]
                result["pattern_insights"] = {
                    "has_repetition": pattern_analysis.get("has_repetition", False),
                    "repetition_type": pattern_analysis.get("repetition_type", "none"),
                    "has_edge_pattern": pattern_analysis.get("edge_patterns", {}).get(
                        "has_edge_pattern", False
                    ),
                    "complexity": pattern_analysis.get("output_complexity", "unknown"),
                }

                # Add dimension scaling information if available
                if "dimension_scale" in pattern_analysis:
                    scale_info = pattern_analysis["dimension_scale"]
                    result["pattern_insights"]["dimension_scale"] = {
                        "row_scale": scale_info.get("row_scale", 1.0),
                        "col_scale": scale_info.get("col_scale", 1.0),
                        "predicted_dimensions": f"{scale_info.get('predicted_rows', '?')}x{scale_info.get('predicted_cols', '?')}",
                    }

                # Log pattern insights
                if verbose_llm_behavior:
                    logger.info(
                        f"Task {task_id}: Pattern analysis insights: {result['pattern_insights']}"
                    )

            # Add debugging info to the result
            if "debug_info" in prediction_result_dict:
                debug_info = prediction_result_dict["debug_info"]
                result["details"]["complexity"] = debug_info.get(
                    "task_complexity", "simple"
                )
                result["details"]["analysis_summary"] = debug_info.get(
                    "pattern_analysis_summary", []
                )

                # Log the pattern analysis summary
                if debug_info.get("pattern_analysis_summary") and verbose_llm_behavior:
                    logger.info(
                        f"Task {task_id}: Analysis summary: {', '.join(debug_info['pattern_analysis_summary'])}"
                    )

            # Include symbolic context used for the prediction
            symbolic_context = prediction_result_dict.get("symbolic_context_used")
            if symbolic_context and verbose_llm_behavior:
                result["details"]["symbolic_context"] = symbolic_context
                logger.debug(
                    f"Task {task_id}: Symbolic context used: {symbolic_context[:100]}..."
                )

        except Exception as e:
            import traceback

            error_msg = f"Exception in ARC task solving: {str(e)}"
            result["error"] = error_msg
            result["status"] = "error"
            logger.error(f"Task {task_id}: {error_msg}")
            logger.debug(traceback.format_exc())

        # Final status logging
        logger.info(f"Task {task_id}: Completed with status: {result['status']}")

        return result

    def format_arc_submission(
        self, task_results: Dict[str, Dict], output_path: Optional[str] = None
    ) -> Dict:
        """
        Format ARC task results into the official ARC submission format.

        The ARC submission format is a JSON object with:
        {
            "root": {
                "task_id_1": [...],
                "task_id_2": [...],
                ...
            }
        }

        Args:
            task_results: Dictionary mapping task IDs to their solution results from solve_arc_task
            output_path: Optional file path to save the JSON submission. If None, only returns the dict

        Returns:
            A dictionary in the proper ARC submission format
        """
        import json
        import logging

        logger = logging.getLogger("arc_llm_format")

        # Create the submission structure with "root" as the top-level key
        submission = {"root": {}}

        # Track validation statistics
        stats = {"total_tasks": len(task_results), "valid_grids": 0, "invalid_grids": 0}

        # Process each task result
        for task_id, result in task_results.items():
            # Skip if not a dictionary with the expected structure
            if not isinstance(result, dict):
                logger.warning(f"Task {task_id}: Result is not a dictionary, skipping")
                stats["invalid_grids"] += 1
                continue

            # Get the predicted grid from the result
            predicted_grid = result.get("predicted_grid")

            # Validate the predicted grid
            if (
                not predicted_grid
                or not isinstance(predicted_grid, list)
                or len(predicted_grid) == 0
            ):
                logger.warning(f"Task {task_id}: Missing or invalid grid prediction")
                stats["invalid_grids"] += 1
                continue

            # Check if the grid contains only integers
            valid_grid = True
            for row in predicted_grid:
                if not isinstance(row, list):
                    valid_grid = False
                    break
                for cell in row:
                    if not isinstance(cell, int):
                        valid_grid = False
                        break
                if not valid_grid:
                    break

            if not valid_grid:
                logger.warning(f"Task {task_id}: Grid contains non-integer values")
                stats["invalid_grids"] += 1
                continue

            # Add the grid to the submission in the proper format
            # Each task ID maps to an array with a single item (the predicted grid)
            submission["root"][task_id] = [predicted_grid]
            stats["valid_grids"] += 1

        # Log the results
        logger.info(
            f"Submission formatted with {stats['valid_grids']}/{stats['total_tasks']} valid tasks"
        )

        # Save to file if requested
        if output_path:
            try:
                with open(output_path, "w") as f:
                    json.dump(submission, f)
                logger.info(f"Submission saved to {output_path}")
            except Exception as e:
                logger.error(f"Error saving submission to {output_path}: {e}")

        return submission

    def batch_solve_arc_tasks(
        self,
        tasks_dict: Dict[str, Dict],
        base_symbolic_context: str = "",
        verbose: bool = False,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_retries: int = 2,
        output_submission_path: Optional[str] = None,
    ) -> Dict:
        """
        Solve multiple ARC tasks and format the results into a submission file.

        Args:
            tasks_dict: Dictionary mapping task IDs to their data (train_pairs and test_input)
            base_symbolic_context: Optional symbolic context for all tasks
            verbose: Whether to log detailed information
            max_tokens: Maximum tokens for LLM generation
            temperature: Temperature for LLM generation
            num_retries: Number of retries for each task
            output_submission_path: Path to save the submission file

        Returns:
            A dictionary with the results for all tasks and the formatted submission
        """
        import logging

        from tqdm import tqdm

        logger = logging.getLogger("arc_batch_solver")

        all_results = {}

        # Process each task with progress tracking
        logger.info(f"Processing {len(tasks_dict)} ARC tasks")
        for task_id, task_data in tqdm(tasks_dict.items(), desc="Solving ARC tasks"):
            try:
                # Extract task data
                train_pairs = task_data.get("train", [])
                test_input = task_data.get("test", [])

                if not train_pairs or not test_input:
                    logger.warning(f"Task {task_id}: Missing train pairs or test input")
                    continue

                # Solve the task
                logger.info(f"Starting task {task_id}")
                result = self.solve_arc_task(
                    train_pairs=train_pairs,
                    test_input=test_input,
                    base_symbolic_context_query=base_symbolic_context,
                    task_id=task_id,
                    verbose=verbose,
                    max_llm_tokens=max_tokens,
                    temperature=temperature,
                    num_retries=num_retries,
                )

                # Store the result
                all_results[task_id] = result

                # Log success or failure
                if result.get("predicted_grid"):
                    logger.info(f"Task {task_id}: Successfully generated prediction")
                else:
                    logger.warning(
                        f"Task {task_id}: Failed to generate valid prediction"
                    )

            except Exception as e:
                logger.error(f"Task {task_id}: Error during processing: {e}")
                all_results[task_id] = {"error": str(e), "predicted_grid": None}

        # Format the results into a submission
        submission = self.format_arc_submission(all_results, output_submission_path)

        return {
            "task_results": all_results,
            "submission": submission,
            "submission_path": output_submission_path,
        }
