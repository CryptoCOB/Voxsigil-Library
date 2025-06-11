# arc_config.py
import os
import sys
import logging
from pathlib import Path
import time
import json

# === Directory and Path Configuration ===
SCRIPT_DIR = Path(__file__).resolve().parent
VOXSIGIL_LIBRARY_PATH = Path(
    os.getenv("VOXSIGIL_LIBRARY_PATH", SCRIPT_DIR / "VoxSigil-Library")
)
ARC_DATA_DIR = Path(os.getenv("ARC_DATA_DIR", SCRIPT_DIR))
ARC_TASKS_FILENAME = os.getenv("ARC_TASKS_FILENAME", "arc-agi_training_challenges.json")
ARC_SOLUTIONS_FILENAME = os.getenv(
    "ARC_SOLUTIONS_FILENAME", "arc-agi_training_solutions.json"
)

RESULTS_OUTPUT_DIR = SCRIPT_DIR / "voxsigil_arc_run_outputs"
RESULTS_OUTPUT_DIR.mkdir(
    parents=True, exist_ok=True
)  # Ensure parent dirs are also created

# === API Configuration ===
OLLAMA_API_BASE_URL = os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434")
LMSTUDIO_API_BASE_URL = os.getenv("LMSTUDIO_API_BASE_URL", "http://127.0.0.1:1234")

OLLAMA_ENDPOINT_CHAT = f"{OLLAMA_API_BASE_URL.rstrip('/')}/v1/chat/completions"
LMSTUDIO_ENDPOINT_CHAT = f"{LMSTUDIO_API_BASE_URL.rstrip('/')}/v1/chat/completions"
OLLAMA_ENDPOINT_MODELS_TAGS = f"{OLLAMA_API_BASE_URL.rstrip('/')}/api/tags"
LMSTUDIO_ENDPOINT_MODELS = f"{LMSTUDIO_API_BASE_URL.rstrip('/')}/v1/models"


# === Default Model Configurations ===
# Model Tiers: 1 (fastest/cheapest) to 5 (most capable/expensive)
DEFAULT_MODELS_CONFIG = [
    # Ollama models
    {
        "name": "mistral:latest",
        "service": "ollama",
        "role": "solver",
        "capabilities": ["reasoning", "json", "MultiAgentSystems"],
        "strength_tier": 3,
        "notes": "Good all-rounder",
    },
    {
        "name": "granite3.2-vision",
        "service": "ollama",
        "role": "solver",
        "capabilities": [
            "reasoning",
            "json",
            "synthesis",
            "vision",
            "MultiAgentSystems",
        ],
        "strength_tier": 3,
        "notes": "Strong reasoning & synthesis with vision capabilities",
    },
    {
        "name": "llava-phi3:latest",
        "service": "ollama",
        "role": "solver",
        "capabilities": ["reasoning", "json", "vision", "MultiAgentSystems"],
        "strength_tier": 4,
        "notes": "LLaVA Phi-3 model with vision capabilities",
    },
    {
        "name": "qwen3:30b-a3b",
        "service": "ollama",
        "role": "solver",
        "capabilities": ["reasoning", "logic", "json", "MultiAgentSystems"],
        "strength_tier": 5,
        "notes": "Good for logic/code-like tasks",
    },
    {
        "name": "cogito:8b",
        "service": "ollama",
        "role": "solver",
        "capabilities": ["general", "json"],
        "strength_tier": 2,
        "notes": "General purpose text",
    },
    # LM Studio models - matching exact model names in LM Studio
    {
        "name": "deepseek-r1-distill-qwen-7b",
        "service": "lmstudio",
        "role": "solver",
        "capabilities": ["reasoning", "json", "synthesis", "MultiAgentSystems"],
        "strength_tier": 5,
        "notes": "DeepSeek R1 Distill model for VoxSigil reasoning",
    },
    {
        "name": "phi-4",
        "service": "lmstudio",
        "role": "solver",
        "capabilities": ["reasoning", "json", "structured_output", "MultiAgentSystems"],
        "strength_tier": 5,
        "notes": "Microsoft Phi-4 model, preferred for synthesis operations",
    },
    {
        "name": "qwen3-30b-a3b",
        "service": "lmstudio",
        "role": "solver",
        "capabilities": [
            "reasoning",
            "json",
            "synthesis",
            "structured_output",
            "MultiAgentSystems",
        ],
        "strength_tier": 5,
        "notes": "Large advanced reasoning model",
    },
    {
        "name": "llama-3.2-1b-instruct",
        "service": "lmstudio",
        "role": "solver",
        "capabilities": ["reasoning", "json"],
        "strength_tier": 2,
        "notes": "Fast small model",
    },
    {
        "name": "granite-vision-3.2-2b",
        "service": "lmstudio",
        "role": "solver",
        "capabilities": [
            "reasoning",
            "json",
            "structured_output",
            "vision",
            "MultiAgentSystems",
        ],
        "strength_tier": 3,
        "notes": "granite-vision-3.2-2b model with vision capabilities",
    },
    {
        "name": "qwen2.5-3b-instruct",
        "service": "lmstudio",
        "role": "solver",
        "capabilities": ["reasoning", "json"],
        "strength_tier": 3,
        "notes": "Fast Qwen model",
    },
    {
        "name": "qwen2.5-coder-14b-instruct",
        "service": "lmstudio",
        "role": "solver",
        "capabilities": [
            "reasoning",
            "code",
            "json",
            "structured_output",
            "MultiAgentSystems",
        ],
        "strength_tier": 4,
        "notes": "Advanced code-oriented model",
    },
]
DEFAULT_SYNTHESIZER_MODEL_CONFIG = {
    "name": "phi-4",
    "service": "lmstudio",
    "role": "synthesizer",
    "capabilities": ["reasoning", "json", "structured_output", "MultiAgentSystems"],
    "strength_tier": 5,
    "notes": "Large model for synthesis",
}

# === LLM Request Parameters ===
LLM_REQUEST_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT", "1800"))
LLM_RETRY_DELAY_SECONDS = int(os.getenv("LLM_RETRY_DELAY", "30"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_SOLVER_TEMPERATURE = float(os.getenv("LLM_SOLVER_TEMP", "0.05"))
LLM_SYNTHESIZER_TEMPERATURE = float(os.getenv("LLM_SYNTHESIZER_TEMP", "0.2"))
LLM_CLEANER_TEMPERATURE = float(os.getenv("LLM_CLEANER_TEMP", "0.0"))

# === Run Control Parameters ===
MAX_ARC_TASKS_TO_PROCESS = int(os.getenv("MAX_ARC_TASKS", "5"))  # 0 for all
USE_LLM_CACHE = os.getenv("USE_LLM_CACHE", "True").lower() == "true"
USE_VOXSIGIL_SYSTEM_PROMPT = (
    os.getenv("USE_VOXSIGIL_SYSTEM_PROMPT", "True").lower() == "true"
)
# === Cache Files ===
LLM_RESPONSE_CACHE_FILE = RESULTS_OUTPUT_DIR / "llm_response_cache.json"


def load_llm_response_cache(cache_file=LLM_RESPONSE_CACHE_FILE):
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception:
                return {}
    return {}


def save_llm_response_cache(cache, cache_file=LLM_RESPONSE_CACHE_FILE):
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def initialize_and_validate_models_config(models_config=DEFAULT_MODELS_CONFIG):
    # Ensure required keys exist and types are correct
    required_keys = {"name", "service", "role", "capabilities", "strength_tier"}
    validated = []
    for model in models_config:
        if not required_keys.issubset(model):
            raise ValueError(f"Model config missing required keys: {model}")
        validated.append(model)
    return validated


def load_voxsigil_entries(library_path=VOXSIGIL_LIBRARY_PATH):
    entries = []
    if library_path.exists() and library_path.is_dir():
        for entry_file in library_path.glob("*.json"):
            try:
                with open(entry_file, "r", encoding="utf-8") as f:
                    entries.append(json.load(f))
            except Exception:
                continue
    return entries


class VoxSigilComponent:
    def __init__(self, name, capabilities, description=None):
        self.name = name
        self.capabilities = capabilities
        self.description = description

    def __repr__(self):
        return f"VoxSigilComponent(name={self.name}, capabilities={self.capabilities})"


def analyze_task_for_categorization_needs(task):
    # Simple heuristic: check for presence of certain keys or patterns
    needs = []
    if "vision" in task.get("capabilities", []):
        needs.append("vision")
    if "reasoning" in task.get("capabilities", []):
        needs.append("reasoning")
    if "synthesis" in task.get("capabilities", []):
        needs.append("synthesis")
    return needs


# === ARC Grid JSON Schema ===
ARC_GRID_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "predicted_grid": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0, "maximum": 9},
            },
            "description": "The predicted output grid for the ARC task. Each inner array is a row.",
        }
    },
    "required": ["predicted_grid"],
}

# === Logging Configuration ===
RUN_TIMESTAMP_STR = time.strftime("%Y%m%d-%H%M%S")
LOG_FILE_PATH = RESULTS_OUTPUT_DIR / f"arc_run_log_{RUN_TIMESTAMP_STR}.log"


def setup_logging():
    logging.basicConfig(
        level=logging.getLevelName(os.getenv("LOG_LEVEL", "INFO").upper()),
        format="%(asctime)s.%(msecs)03d - %(levelname)s - [%(name)s_%(module)s.%(funcName)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE_PATH, encoding="utf-8"),
        ],
    )
    # For cleaner logs, you might want to get a specific logger instance
    # logger = logging.getLogger("ARC_ENGINE") # and use this logger elsewhere


# === Cache Files ===
LLM_RESPONSE_CACHE_FILE = RESULTS_OUTPUT_DIR / "llm_response_cache.json"

# === Special Sigil Names for Strategy ===
ARC_SOLVER_SIGIL_NAME = "🔢🧩ARC_SOLVER"
CATENGINE_SIGIL_NAME = "🜛CATENGINE"

# === Cache Configuration ===
MAX_CACHE_AGE_DAYS = int(os.getenv("MAX_CACHE_AGE_DAYS", "30"))  # Default: 30 days
MAX_CACHE_SIZE_ITEMS = int(
    os.getenv("MAX_CACHE_SIZE_ITEMS", "10000")
)  # Default: 10,000 items
DEFAULT_FILE_HASH_ALGO = os.getenv(
    "DEFAULT_FILE_HASH_ALGO", "md5"
)  # Default: MD5 hashing algorithm

# === Strategy Configuration ===
DEFAULT_STRATEGY_PRIORITY = [
    "solver",
    "synthesizer",
    "cleaner",
]  # Default priority order for strategies
SYNTHESIS_FAILURE_FALLBACK_STRATEGY = (
    "retry_with_alternate_model"  # Fallback strategy on synthesis failure
)
DETAILED_PROMPT_METADATA = (
    os.getenv("DETAILED_PROMPT_METADATA", "True").lower() == "true"
)  # Include detailed metadata in prompts
DETAILED_PROMPT_METADATA_IN_LOG = (
    os.getenv("DETAILED_PROMPT_METADATA_IN_LOG", "True").lower() == "true"
)  # Include detailed metadata in logs
