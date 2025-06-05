#!/usr/bin/env python
"""
ARC VoxSigil Loader Module (arc_voxsigil_loader.py)

Handles loading of VoxSigil system prompts and configurations for ARC tasks.
Provides utilities for loading prompts from schema files, templates, and fallbacks.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

# --- Logger Setup ---
logger = logging.getLogger("ARC.VoxSigilLoader")


class VoxSigilComponent:
    """
    Represents a VoxSigil component with associated metadata.
    Used for strategy selection and prompt management in the ARC system.
    """

    def __init__(
        self,
        name: Union[str, Dict[str, Any]],
        capabilities: Optional[List[str]] = None,
        description: Optional[str] = None,
    ):
        # Handle the case when first parameter is a dictionary
        if isinstance(name, dict):
            config_dict = name
            self.name = config_dict.get("name", "unnamed_component")
            self.capabilities = config_dict.get("capabilities", [])
            self.description = config_dict.get("description", None)
            self.tags = config_dict.get("tags", [])
            self.sigil = config_dict.get("sigil", "general")
            self.prompt_template_content = config_dict.get("prompt_template", None)
            self.properties = config_dict.get("properties", {})
            self.execution_mode = config_dict.get("execution_mode", "standard")
            self.expected_placeholders = config_dict.get("expected_placeholders", [])
            self.parameterization_schema = config_dict.get(
                "parameterization_schema", {"parameters": []}
            )
            self.task_description_template = config_dict.get(
                "task_description_template", None
            )
        else:
            self.name = name
            self.capabilities = capabilities or []
            self.description = description
            self.tags = []
            self.sigil = "general"
            self.prompt_template_content = None
            self.properties = {}
            self.execution_mode = "standard"
            self.expected_placeholders = []
            self.parameterization_schema = {"parameters": []}
            self.task_description_template = None

    def __repr__(self):
        return f"VoxSigilComponent(name={self.name}, sigil={self.sigil}, capabilities={self.capabilities}, tags={self.tags})"

    def has_capability(self, capability: str) -> bool:
        """Check if component has a specific capability."""
        return capability in self.capabilities

    def has_tag(self, tag: str) -> bool:
        """Check if component has a specific tag."""
        return tag in self.tags

    def get_property(self, key: str, default: Any = None) -> Any:
        """Get a component property with optional default value."""
        return self.properties.get(key, default)

    def get_task_description(self, task_id: str, task_props=None) -> str:
        """Get a description for the task using this component."""
        if self.task_description_template:
            try:
                return self.task_description_template.replace("{{task_id}}", task_id)
            except:
                pass
        return f"Solve task '{task_id}' using '{self.sigil}' strategy."


# --- Default System Prompts ---
DEFAULT_ARC_SYSTEM_PROMPT = """You are VoxSigil, an advanced AI assistant specialized in solving Abstract Reasoning Corpus (ARC) tasks.

Your primary capabilities include:
- Analyzing visual patterns in grid-based puzzles
- Identifying transformation rules and logical relationships
- Generating precise grid-based outputs in the correct format
- Reasoning through complex spatial and temporal relationships

When working with ARC tasks:
1. Carefully analyze input grids to identify patterns
2. Consider transformations like rotation, reflection, scaling, color changes, and shape operations
3. Look for consistent rules across training examples
4. Apply discovered rules to generate test outputs
5. Always output grids in valid JSON format as arrays of arrays

Response Format:
- Provide your reasoning process step by step
- Include your final answer as a JSON grid: [[row1], [row2], ...]
- Use integers 0-9 for grid values
- Ensure grid dimensions are consistent and logical

Focus on accuracy, pattern recognition, and systematic reasoning."""

DEFAULT_GENERAL_SYSTEM_PROMPT = """You are VoxSigil, a helpful and intelligent AI assistant.

You excel at:
- Clear and accurate reasoning
- Structured problem-solving
- Providing well-formatted responses
- Following specific output requirements

Always strive for accuracy, clarity, and helpfulness in your responses."""

# --- Configuration ---
VOXSIGIL_LIBRARY_PATH = Path(
    os.getenv(
        "VOXSIGIL_LIBRARY_PATH",
        Path(__file__).parent.parent.parent / "VoxSigil_Library",
    )
)

SCHEMA_SEARCH_PATHS = ["schema", "prompts", "config", "system_prompts", "."]

SCHEMA_FILENAMES = [
    "voxsigil-schema-current.yaml",
    "voxsigil_schema.yaml",
    "system_prompt.yaml",
    "voxsigil.yaml",
    "config.yaml",
]


def find_schema_file(library_path: Optional[Path] = None) -> Optional[Path]:
    """
    Find VoxSigil schema file in common locations.

    Args:
        library_path: Optional override for library path

    Returns:
        Path to schema file if found, None otherwise
    """
    search_root = library_path or VOXSIGIL_LIBRARY_PATH

    if not search_root.exists():
        logger.debug(f"Library path does not exist: {search_root}")
        return None

    # Search in subdirectories
    for subdir in SCHEMA_SEARCH_PATHS:
        search_dir = search_root / subdir
        if not search_dir.exists():
            continue

        for filename in SCHEMA_FILENAMES:
            schema_file = search_dir / filename
            if schema_file.exists():
                logger.info(f"Found schema file: {schema_file}")
                return schema_file

    # Search in root directory
    for filename in SCHEMA_FILENAMES:
        schema_file = search_root / filename
        if schema_file.exists():
            logger.info(f"Found schema file in root: {schema_file}")
            return schema_file

    logger.debug(f"No schema file found in {search_root}")
    return None


def load_yaml_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load and parse a YAML file safely.

    Args:
        file_path: Path to YAML file

    Returns:
        Parsed YAML data or None if failed
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        logger.debug(f"Successfully loaded YAML: {file_path}")
        return data
    except Exception as e:
        logger.warning(f"Failed to load YAML file {file_path}: {e}")
        return None


def load_json_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load and parse a JSON file safely.

    Args:
        file_path: Path to JSON file

    Returns:
        Parsed JSON data or None if failed
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.debug(f"Successfully loaded JSON: {file_path}")
        return data
    except Exception as e:
        logger.warning(f"Failed to load JSON file {file_path}: {e}")
        return None


def extract_system_prompt_from_schema(schema_data: Dict[str, Any]) -> Optional[str]:
    """
    Extract system prompt from loaded schema data.

    Args:
        schema_data: Parsed schema data

    Returns:
        System prompt string or None if not found
    """
    if not isinstance(schema_data, dict):
        return None

    # Common keys where system prompt might be stored
    prompt_keys = [
        "system_prompt",
        "systemPrompt",
        "prompt",
        "default_prompt",
        "arc_system_prompt",
        "voxsigil_prompt",
    ]

    for key in prompt_keys:
        if key in schema_data:
            prompt = schema_data[key]
            if isinstance(prompt, str) and prompt.strip():
                logger.debug(f"Found system prompt under key: {key}")
                return prompt.strip()

    # Look for nested structures
    for key, value in schema_data.items():
        if isinstance(value, dict):
            nested_prompt = extract_system_prompt_from_schema(value)
            if nested_prompt:
                return nested_prompt

    return None


def load_voxsigil_system_prompt(library_path_override: Optional[Path] = None) -> str:
    """
    Load VoxSigil system prompt from schema files or return fallback.

    This is the main function expected by arc_llm_handler.py.

    Args:
        library_path_override: Optional path override for VoxSigil library

    Returns:
        System prompt string (never None, falls back to default)
    """
    logger.info("Loading VoxSigil system prompt...")

    # Try to find and load schema file
    schema_file = find_schema_file(library_path_override)

    if schema_file:
        # Try loading as YAML first, then JSON
        schema_data = load_yaml_file(schema_file)
        if not schema_data and schema_file.suffix.lower() == ".json":
            schema_data = load_json_file(schema_file)

        if schema_data:
            system_prompt = extract_system_prompt_from_schema(schema_data)
            if system_prompt:
                logger.info("Successfully loaded system prompt from schema")
                return system_prompt
            else:
                logger.warning("Schema file found but no system prompt extracted")
        else:
            logger.warning(f"Failed to parse schema file: {schema_file}")

    # Check environment variable override
    env_prompt = os.getenv("VOXSIGIL_SYSTEM_PROMPT")
    if env_prompt and env_prompt.strip():
        logger.info("Using system prompt from environment variable")
        return env_prompt.strip()

    # Check for ARC-specific prompt if this is an ARC context
    if "arc" in str(library_path_override or "").lower():
        logger.info("Using default ARC system prompt")
        return DEFAULT_ARC_SYSTEM_PROMPT

    # Fall back to general prompt
    logger.info("Using default general system prompt")
    return DEFAULT_GENERAL_SYSTEM_PROMPT


def create_sample_schema(output_path: Path, prompt_type: str = "arc") -> bool:
    """
    Create a sample VoxSigil schema file for testing.

    Args:
        output_path: Where to save the schema file
        prompt_type: Type of prompt ("arc" or "general")

    Returns:
        True if created successfully, False otherwise
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if prompt_type.lower() == "arc":
            prompt = DEFAULT_ARC_SYSTEM_PROMPT
            description = "ARC task specialized prompt"
        else:
            prompt = DEFAULT_GENERAL_SYSTEM_PROMPT
            description = "General purpose prompt"

        schema_data = {
            "voxsigil_schema_version": "1.4",
            "description": f"VoxSigil {description}",
            "system_prompt": prompt,
            "capabilities": ["reasoning", "analysis", "structured_output"],
            "created_by": "arc_voxsigil_loader.py",
            "created_at": "auto-generated",
        }

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(schema_data, f, default_flow_style=False, indent=2)

        logger.info(f"Created sample schema at: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to create sample schema: {e}")
        return False


def validate_system_prompt(prompt: str) -> bool:
    """
    Validate that a system prompt is reasonable.

    Args:
        prompt: System prompt to validate

    Returns:
        True if prompt passes basic validation
    """
    if not isinstance(prompt, str):
        return False

    prompt = prompt.strip()

    # Basic checks
    if len(prompt) < 10:
        logger.warning("System prompt is very short")
        return False

    if len(prompt) > 10000:
        logger.warning("System prompt is very long")
        return False

    # Check for common required elements
    required_indicators = [
        "you are",
        "assistant",
        "ai",
        "voxsigil",
        "help",
        "task",
        "respond",
        "answer",
    ]

    prompt_lower = prompt.lower()
    if not any(indicator in prompt_lower for indicator in required_indicators):
        logger.warning("System prompt may not be properly formatted")
        return False

    return True


def get_prompt_info(library_path_override: Optional[Path] = None) -> Dict[str, Any]:
    """
    Get information about the current prompt configuration.

    Args:
        library_path_override: Optional path override

    Returns:
        Dictionary with prompt configuration info
    """
    info = {
        "library_path": str(library_path_override or VOXSIGIL_LIBRARY_PATH),
        "schema_file": None,
        "prompt_source": "fallback",
        "prompt_valid": False,
        "prompt_length": 0,
    }

    # Try to find schema file
    schema_file = find_schema_file(library_path_override)
    if schema_file:
        info["schema_file"] = str(schema_file)
        info["prompt_source"] = "schema_file"

    # Check environment variable
    env_prompt = os.getenv("VOXSIGIL_SYSTEM_PROMPT")
    if env_prompt:
        info["prompt_source"] = "environment_variable"

    # Load the actual prompt
    prompt = load_voxsigil_system_prompt(library_path_override)
    info["prompt_length"] = len(prompt)
    info["prompt_valid"] = validate_system_prompt(prompt)

    return info


# --- Testing and Debug Functions ---
if __name__ == "__main__":
    # Simple test when run directly
    print("=== VoxSigil Loader Test ===")

    # Test prompt loading
    prompt = load_voxsigil_system_prompt()
    print(f"Loaded prompt ({len(prompt)} chars):")
    print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
    print()

    # Test prompt info
    info = get_prompt_info()
    print("Prompt Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    print()

    # Test schema creation
    test_schema_path = Path("test_voxsigil_schema.yaml")
    if create_sample_schema(test_schema_path, "arc"):
        print(f"Created test schema: {test_schema_path}")

        # Test loading the created schema
        test_prompt = load_voxsigil_system_prompt(test_schema_path.parent)
        print(f"Test prompt loaded: {len(test_prompt)} chars")

        # Clean up
        try:
            test_schema_path.unlink()
            print("Cleaned up test file")
        except:
            pass

# --- Initialization ---
logger.info("ARC VoxSigil loader module initialized")
