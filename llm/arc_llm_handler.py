#!/usr/bin/env python
"""
VoxSigil Dynamic LLM Handler (voxsigil_llm_handler.py)

Manages interaction with various LLM services, including dynamic model discovery,
performance tracking, and intelligent model routing for ARC tasks.
This module assumes prompts (including any RAG enhancements) are provided to it.
It imports and uses functionality from other VoxSigil modules if needed (e.g., system prompt loading).
"""

# --- Standard Library Imports ---
import hashlib
import json
import logging
import os
import random
import re
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Local import from same directory instead of package import
try:
    from .arc_utils import LLM_RESPONSE_CACHE
except ImportError:
    try:
        from arc_utils import LLM_RESPONSE_CACHE
    except ImportError:
        # Create fallback cache if import fails
        LLM_RESPONSE_CACHE = {}

# --- Third-Party Library Imports (with conditional fallbacks) ---
try:
    import requests

    HAVE_REQUESTS = True
except ImportError:
    requests = None  # type: ignore
    HAVE_REQUESTS = False
    # Critical, will be checked

try:
    import tiktoken

    HAVE_TIKTOKEN = True
except ImportError:
    tiktoken = None  # type: ignore
    HAVE_TIKTOKEN = False

# --- VoxSigil System Imports (assuming these files are in python path or same directory) ---
try:
    # If you have a central config in another file:
    # from voxsigil_config import VS_LLM_CONFIG, VS_CONFIG (adjust as per your config structure)
    # For this standalone handler, we'll define config internally or rely on its own.
    # If voxsigil_loader provides system prompt:
    try:
        # Local import from same directory instead of package import
        from .arc_voxsigil_loader import load_voxsigil_system_prompt
    except ImportError:
        try:
            from arc_voxsigil_loader import load_voxsigil_system_prompt
        except ImportError:
            # Try the absolute path as well
            from ARC.llm.arc_voxsigil_loader import load_voxsigil_system_prompt
    HAVE_VOXSIGIL_LOADER = True
except ImportError:
    import logging

    logging.getLogger("VoxSigilLLMHandler").warning(
        "voxsigil_loader.py not found or load_voxsigil_system_prompt unavailable. "
        "System prompt loading will use a fallback."
    )
    HAVE_VOXSIGIL_LOADER = False

    # Define a fallback if the loader isn't available
    def load_voxsigil_system_prompt(library_path_override=None) -> str:
        return "Fallback System Prompt: You are a helpful AI assistant."


# --- Configuration Class (specific to LLM Handler) ---
class VoxSigilLLMHandlerConfig:
    def __init__(self):
        self.CACHE_DIR = Path(
            os.getenv(
                "VOXSIGIL_CACHE_DIR",
                Path(__file__).resolve().parent / ".voxsigil_cache",
            )
        )
        self.STATS_DB_PATH = self.CACHE_DIR / "voxsigil_llm_performance_stats.db"
        self.DISCOVERED_MODELS_CACHE_FILE = (
            self.CACHE_DIR / "voxsigil_discovered_llms_cache.json"
        )

        self.LOG_LEVEL_STR = os.getenv("VOXSIGIL_LOG_LEVEL", "INFO").upper()
        self.LOG_LEVEL = logging._nameToLevel.get(self.LOG_LEVEL_STR, logging.INFO)
        self.LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        self.OLLAMA_API_BASE_URLS = [
            url.strip()
            for url in os.getenv(
                "VOXSIGIL_OLLAMA_URLS", "http://localhost:11434"
            ).split(",")
            if url.strip()
        ]
        # Disable LM Studio for RAG and embedding
        self.LMSTUDIO_API_BASE_URLS = []  # Removed default LM Studio URLs to disable it
        self.OTHER_OPENAI_COMPATIBLE_URLS = [
            url.strip()
            for url in os.getenv("VOXSIGIL_OTHER_LLM_URLS", "").split(",")
            if url.strip()
        ]

        self.DEFAULT_SERVICE_ENDPOINTS = {
            "ollama": {
                "chat": "/api/chat",
                "models": "/api/tags",
                "generate": "/api/generate",
            },  # generate for ping/load
            "lmstudio": {"chat": "/v1/chat/completions", "models": "/v1/models"},
            "openai_compatible": {
                "chat": "/v1/chat/completions",
                "models": "/v1/models",
            },
        }

        self.MODEL_PREFERENCES = json.loads(
            os.getenv("VOXSIGIL_MODEL_PREFERENCES", "[]")
        )
        self.DEFAULT_MODEL_STRENGTH_TIER = int(
            os.getenv("VOXSIGIL_DEFAULT_MODEL_TIER", "3")
        )

        self.LLM_REQUEST_TIMEOUT_SECONDS = int(os.getenv("VOXSIGIL_LLM_TIMEOUT", "180"))
        self.LLM_MAX_RETRIES = int(os.getenv("VOXSIGIL_LLM_MAX_RETRIES", "2"))
        self.LLM_RETRY_DELAY_SECONDS = int(os.getenv("VOXSIGIL_LLM_RETRY_DELAY", "10"))
        self.USE_LLM_RESPONSE_CACHE = (
            os.getenv("VOXSIGIL_USE_LLM_RESPONSE_CACHE", "True").lower() == "true"
        )
        self.DISCOVERED_MODELS_CACHE_TTL_HOURS = int(
            os.getenv("VOXSIGIL_MODEL_CACHE_TTL_HOURS", "1")
        )

        self.TIKTOKEN_DEFAULT_ENCODING = os.getenv(
            "VOXSIGIL_TIKTOKEN_ENCODING", "cl100k_base"
        )
        self.ARC_GRID_JSON_SCHEMA = {
            "type": "array",
            "items": {"type": "array", "items": {"type": "integer"}},
        }

        # Path to the main VoxSigil library if system prompt relies on it.
        # This assumes that if voxsigil_loader is present, it knows where its library is.
        # If voxsigil_loader itself needs this path, it should also get it from env or have a default.
        self.VOXSIGIL_LIBRARY_PATH_FOR_PROMPT = Path(
            os.getenv(
                "VOXSIGIL_LIBRARY_PATH",
                Path(__file__).resolve().parent / "VoxSigil-Library",
            )
        )


VS_LLM_CONFIG = VoxSigilLLMHandlerConfig()
VS_LLM_CONFIG.CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- Logging Setup ---
_llm_logger_initialized_flag = False


def get_llm_handler_logger(name: str) -> logging.Logger:
    global _llm_logger_initialized_flag
    logger_instance = logging.getLogger(name)
    if not _llm_logger_initialized_flag or not logger_instance.handlers:
        logging.basicConfig(
            level=VS_LLM_CONFIG.LOG_LEVEL, format=VS_LLM_CONFIG.LOG_FORMAT, force=True
        )
        for lib_name in ["requests", "urllib3", "httpx"]:
            logging.getLogger(lib_name).setLevel(logging.WARNING)
        _llm_logger_initialized_flag = True
    logger_instance.setLevel(VS_LLM_CONFIG.LOG_LEVEL)
    return logger_instance


logger = get_llm_handler_logger("VoxSigilLLMHandler")

# --- Tokenizer Instance ---
_tokenizer_instance: Optional[Any] = None
if HAVE_TIKTOKEN and tiktoken:
    try:
        _tokenizer_instance = tiktoken.get_encoding(
            VS_LLM_CONFIG.TIKTOKEN_DEFAULT_ENCODING
        )
        logger.info(
            f"tiktoken tokenizer '{VS_LLM_CONFIG.TIKTOKEN_DEFAULT_ENCODING}' initialized."
        )
    except Exception as e_tok:
        logger.warning(
            f"Failed to initialize tiktoken '{VS_LLM_CONFIG.TIKTOKEN_DEFAULT_ENCODING}': {e_tok}. Using fallback."
        )
else:
    logger.info(
        "tiktoken library not available or disabled. Using character-based token estimation."
    )


def count_tokens(text: str, model_name_for_encoding: Optional[str] = None) -> int:
    # (Implementation from previous full script)
    if not text:
        return 0
    if HAVE_TIKTOKEN and tiktoken:
        tokenizer_to_use = _tokenizer_instance
        # Try model-specific encoding if no global one or if requested
        if model_name_for_encoding and (
            not tokenizer_to_use or model_name_for_encoding
        ):
            try:
                tokenizer_to_use = tiktoken.encoding_for_model(model_name_for_encoding)
            except Exception:
                tokenizer_to_use = tiktoken.get_encoding(
                    VS_LLM_CONFIG.TIKTOKEN_DEFAULT_ENCODING
                )

        if tokenizer_to_use and hasattr(tokenizer_to_use, "encode"):
            try:
                return len(tokenizer_to_use.encode(text))
            except Exception:
                pass
    return max(1, len(text) // 4)


# --- Performance Stats Database Functions ---
# setup_stats_db, log_performance_stat, get_model_performance_summary
# (These implementations are taken from the previous fully combined script, they are good as is)
def setup_stats_db():
    if (
        not Path(VS_LLM_CONFIG.STATS_DB_PATH).exists()
        or os.path.getsize(VS_LLM_CONFIG.STATS_DB_PATH) == 0
    ):
        logger.info(
            f"Initializing LLM performance stats database at: {VS_LLM_CONFIG.STATS_DB_PATH}"
        )
    with sqlite3.connect(VS_LLM_CONFIG.STATS_DB_PATH) as conn:  # Use a context manager
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT, model_id TEXT NOT NULL, service TEXT NOT NULL, base_url TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, request_duration_ms INTEGER,
            input_tokens INTEGER, output_tokens INTEGER, tokens_per_second REAL,
            success BOOLEAN, error_type TEXT, http_status_code INTEGER,
            temperature REAL, retry_attempt INTEGER, time_to_first_token_ms INTEGER
        )""")
        # Add indexes for faster queries
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_model_service_time ON model_performance (model_id, service, timestamp)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_model_baseurl_time ON model_performance (model_id, base_url, timestamp)"
        )
        conn.commit()


setup_stats_db()


def log_performance_stat(stat_data: Dict[str, Any]):
    stat_data.setdefault(
        "timestamp", datetime.now(timezone.utc).isoformat(timespec="milliseconds") + "Z"
    )  # ISO 8601 UTC
    stat_data = {
        k: (round(v, 3) if isinstance(v, float) else v) for k, v in stat_data.items()
    }  # Round floats
    with sqlite3.connect(VS_LLM_CONFIG.STATS_DB_PATH) as conn:
        cursor = conn.cursor()
        # Robust insert or update based on a unique combination? For now, just insert.
        columns = ", ".join(stat_data.keys())
        placeholders = ", ".join("?" * len(stat_data))
        sql = f"INSERT INTO model_performance ({columns}) VALUES ({placeholders})"
        try:
            cursor.execute(sql, tuple(stat_data.values()))
            conn.commit()
        except sqlite3.Error as e_sql:
            logger.error(
                f"SQLite error logging performance stat: {e_sql} - Data: {stat_data}"
            )


def get_model_performance_summary(
    model_id: str,
    service: str,
    base_url: Optional[str] = None,
    lookback_hours: int = 24,
) -> Dict[str, Any]:
    summary = {
        "avg_latency_ms": None,
        "avg_tps": None,
        "success_rate": None,
        "error_count": 0,
        "total_requests": 0,
        "last_used_utc": None,
    }
    with sqlite3.connect(VS_LLM_CONFIG.STATS_DB_PATH) as conn:
        cursor = conn.cursor()
        # Ensure timestamp is in UTC for comparison
        since_time_utc = (
            datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        ).isoformat(timespec="milliseconds") + "Z"

        query_params: List[Any] = [model_id, service, since_time_utc]
        base_url_condition = ""
        if base_url:
            base_url_condition = "AND base_url = ? "
            query_params.append(base_url)

        query = f"""
        SELECT 
            AVG(CASE WHEN success = 1 THEN request_duration_ms ELSE NULL END), /* Avg latency of successful calls */
            AVG(CASE WHEN success = 1 THEN tokens_per_second ELSE NULL END), /* Avg TPS of successful calls */
            SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*), /* Success rate */
            SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END), /* Error count */
            COUNT(*), /* Total requests */
            MAX(timestamp) /* Last used time */
        FROM model_performance 
        WHERE model_id = ? AND service = ? {base_url_condition} AND timestamp >= ?
        """
        try:
            cursor.execute(query, tuple(query_params))
            row = cursor.fetchone()
            if row and row[4] is not None and row[4] > 0:  # row[4] is COUNT(*)
                summary.update(
                    {
                        "avg_latency_ms": round(row[0], 0)
                        if row[0] is not None
                        else None,
                        "avg_tps": round(row[1], 2) if row[1] is not None else None,
                        "success_rate": round(row[2], 3)
                        if row[2] is not None
                        else None,
                        "error_count": row[3] or 0,
                        "total_requests": row[4],
                        "last_used_utc": row[5],
                    }
                )
        except sqlite3.Error as e_sql_sum:
            logger.error(
                f"SQLite error fetching performance summary for {model_id}/{service}: {e_sql_sum}"
            )
    return summary


# --- LLM Response Cache Key ---
def get_llm_cache_key(
    content_str: str, model_id: str, temp: float, service: str
) -> str:
    # (Implementation from previous)
    key_string = f"model:{model_id}|svc:{service}|temp:{temp:.2f}|content_hash:{hashlib.sha256(content_str.encode('utf-8')).hexdigest()}"
    return hashlib.sha256(key_string.encode("utf-8")).hexdigest()


# --- Model Discovery (implementations from previous full script, slightly refined) ---
# _parse_ollama_models, _parse_openai_compatible_models, _probe_model_basic,
# query_all_services_for_models, _load_cached_discovered_models, _save_discovered_models_to_cache
# ensure_ollama_model_active
# (These are extensive but crucial for this module. They will be included but perhaps slightly condensed in explanation)


def _parse_discovered_model_info(
    raw_info: Dict[str, Any], service_name: str, base_url: str
) -> Optional[Dict[str, Any]]:
    # (Implementation as in previous LLM handler focused script)
    model_id: Optional[str] = None
    display_name: Optional[str] = None
    details: Dict[str, Any] = {}
    param_size_gb: Optional[float] = None
    family_name: Optional[str] = None

    if service_name == "ollama":
        model_id = raw_info.get("name")
        display_name = model_id
        details = raw_info.get("details", {})
        param_size_str = details.get("parameter_size", "0B")
        family_name = details.get("family")
        if "B" in param_size_str.upper():
            try:
                param_size_gb = float(re.sub(r"[^\d.]", "", param_size_str))
            except Exception:
                pass
    elif service_name in ["lmstudio", "openai_compatible"]:
        model_id = raw_info.get("id")
        display_name = model_id

    if not model_id:
        return None

    tier = VS_LLM_CONFIG.DEFAULT_MODEL_STRENGTH_TIER
    model_id_lower = model_id.lower()
    if param_size_gb is not None:  # Tier by params if known
        if param_size_gb >= 65:
            tier = 5
        elif param_size_gb >= 25:
            tier = 4
        elif param_size_gb >= 7:
            tier = 3
        else:
            tier = 2
    else:  # Fallback to name heuristics
        if any(
            s in model_id_lower
            for s in ["gpt-4", "claude-3", "command-r-plus", "70b", "qwen2:72b"]
        ):
            tier = 5
        elif any(
            s in model_id_lower
            for s in ["gpt-3.5", "mixtral", "llama3:8b", "phi3:14b", "30b"]
        ):
            tier = 4
        elif any(
            s in model_id_lower for s in ["phi3", "llama2:7b", "gemma:7b", "mistral:7b"]
        ):
            tier = 3

    caps = ["general"]
    if tier >= 3 or "json" in model_id_lower:
        caps.append("json")
    if (
        "vision" in model_id_lower
        or "llava" in model_id_lower
        or "bakllava" in model_id_lower
        or "moondream" in model_id_lower
    ):
        caps.append("vision")
    if family_name and family_name.lower() not in caps:
        caps.append(family_name.lower())  # Add Ollama family as a capability

    return {
        "id": model_id,
        "name": display_name,
        "service": service_name,
        "base_url": base_url,
        "capabilities": sorted(list(set(caps))),
        "strength_tier": tier,
        "raw_details": details,
        "source": "discovery",
        "last_seen_utc": datetime.now(timezone.utc).isoformat(),
    }


def _probe_model_basic(model_info: Dict[str, Any]) -> Dict[str, Any]:
    # (Implementation as in previous LLM handler focused script)
    if not HAVE_REQUESTS or requests is None:
        return model_info
    probed_info = model_info.copy()
    probed_info.setdefault(
        "probed_capabilities", list(probed_info.get("capabilities", []))
    )
    chat_ep_path = VS_LLM_CONFIG.DEFAULT_SERVICE_ENDPOINTS.get(
        model_info["service"], {}
    ).get("chat")
    if not chat_ep_path:
        return probed_info

    chat_ep = f"{model_info['base_url'].rstrip('/')}{chat_ep_path}"
    payload_chat = {
        "model": model_info["id"],
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 2,
        "temperature": 0.0,
    }

    try:
        resp = requests.post(
            chat_ep, json=payload_chat, timeout=15
        )  # Faster timeout for probe
        if resp.ok:
            probed_info["status"] = "responsive"
        else:
            probed_info["status"] = f"error_{resp.status_code}"
            return probed_info
    except Exception:
        probed_info["status"] = "unreachable"
        return probed_info

    if (
        "json" in model_info.get("capabilities", [])
        or model_info.get("strength_tier", 1) >= 3
    ):
        # ... (JSON probe logic from previous, condensed for brevity)
        payload_json = {
            "model": model_info["id"],
            "messages": [{"role": "user", "content": '{"key":"val"}'}],
            "max_tokens": 5,
        }
        if model_info["service"] == "ollama":
            payload_json["format"] = "json"
        elif model_info["service"] in ["lmstudio", "openai_compatible"]:
            payload_json["response_format"] = {"type": "json_object"}

        try:
            resp_json = requests.post(chat_ep, json=payload_json, timeout=15)
            if resp_json.ok:  # And check if content is valid JSON
                if model_info["service"] == "ollama":
                    content = resp_json.json().get("message", {}).get("content")
                else:
                    content = (
                        resp_json.json()
                        .get("choices", [{}])[0]
                        .get("message", {})
                        .get("content")
                    )
                if content:
                    try:
                        json.loads(content)
                        probed_info.setdefault("probed_capabilities", []).append(
                            "json_confirmed"
                        )
                    except Exception as e:
                        logger.debug(f"Failed JSON probe decode: {e}")
        except Exception as e:
            logger.debug(f"JSON probing failed: {e}")

    probed_info["probed_capabilities"] = sorted(
        list(set(probed_info["probed_capabilities"]))
    )
    return probed_info


def query_all_services_for_models() -> List[Dict[str, Any]]:
    # (Implementation from previous, ensure robust requests usage)
    if not HAVE_REQUESTS or requests is None:
        logger.critical("Requests library not installed. Cannot discover models.")
        return []

    cached_models = _load_cached_discovered_models()
    if cached_models:
        logger.info(f"Loaded {len(cached_models)} models from discovery cache.")
        return cached_models

    all_probed_models: List[Dict[str, Any]] = []
    # ... (Loop through OLLAMA_URLS, LMSTUDIO_URLS, OTHER_URLS as in previous script) ...
    service_map = {
        "ollama": VS_LLM_CONFIG.OLLAMA_API_BASE_URLS,
        "lmstudio": VS_LLM_CONFIG.LMSTUDIO_API_BASE_URLS,
        "openai_compatible": VS_LLM_CONFIG.OTHER_OPENAI_COMPATIBLE_URLS,
    }
    for service_name, base_urls_list in service_map.items():
        for base_url_str in base_urls_list:
            base_url_str = base_url_str.strip()
            if not base_url_str:
                continue

            ep_config = VS_LLM_CONFIG.DEFAULT_SERVICE_ENDPOINTS.get(service_name, {})
            models_ep_path = ep_config.get("models")
            if not models_ep_path:
                continue
            full_models_ep = f"{base_url_str.rstrip('/')}{models_ep_path}"

            logger.info(
                f"Querying {service_name} @ {base_url_str} (Endpoint: {full_models_ep})"
            )
            try:
                response = requests.get(
                    full_models_ep, headers={"Accept": "application/json"}, timeout=10
                )
                response.raise_for_status()
                raw_data = response.json()

                raw_model_list = []
                if service_name == "ollama" and "models" in raw_data:
                    raw_model_list = raw_data["models"]
                elif "data" in raw_data:
                    raw_model_list = raw_data["data"]  # OpenAI / LMStudio style

                for raw_model_item in raw_model_list:
                    parsed_info = _parse_discovered_model_info(
                        raw_model_item, service_name, base_url_str
                    )
                    if parsed_info:
                        probed_info = _probe_model_basic(parsed_info)
                        if probed_info.get("status") == "responsive":
                            all_probed_models.append(probed_info)
            except Exception as e:
                logger.warning(f"Failed discovering from {full_models_ep}: {e}")

    # Deduplicate: (id, service, base_url) should be unique
    unique_models_dict: Dict[Tuple[str, str, str], Dict[str, Any]] = {
        (m["id"], m["service"], m["base_url"]): m for m in all_probed_models
    }
    final_list = list(unique_models_dict.values())
    _save_discovered_models_to_cache(final_list)
    return final_list


def _load_cached_discovered_models() -> Optional[List[Dict[str, Any]]]:
    # (Implementation from previous)
    cache_file = VS_LLM_CONFIG.DISCOVERED_MODELS_CACHE_FILE
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if (
                datetime.now(timezone.utc)
                - datetime.fromisoformat(data.get("timestamp"))
            ) < timedelta(hours=VS_LLM_CONFIG.DISCOVERED_MODELS_CACHE_TTL_HOURS):
                return data.get("models")
        except Exception as e:
            logger.warning(f"Error loading model cache: {e}")
    return None


def _save_discovered_models_to_cache(models: List[Dict[str, Any]]):
    # (Implementation from previous)
    VS_LLM_CONFIG.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(
            VS_LLM_CONFIG.DISCOVERED_MODELS_CACHE_FILE, "w", encoding="utf-8"
        ) as f:
            json.dump(
                {"timestamp": datetime.now(timezone.utc).isoformat(), "models": models},
                f,
                indent=2,
            )
        logger.info(f"Saved {len(models)} models to discovery cache.")
    except Exception as e:
        logger.warning(f"Error saving model cache: {e}")


def ensure_ollama_model_active(model_name: str, base_url: str) -> bool:
    """Attempts to make an Ollama model active/loaded via a minimal generate call."""
    if not HAVE_REQUESTS or requests is None:
        return False
    # (Implementation from previous LLM handler focused script)
    ep_path = VS_LLM_CONFIG.DEFAULT_SERVICE_ENDPOINTS.get("ollama", {}).get("generate")
    if not ep_path:
        return False
    endpoint = f"{base_url.rstrip('/')}{ep_path}"
    payload = {
        "model": model_name,
        "prompt": " ",
        "stream": False,
        "options": {"num_predict": 0},
    }  # num_predict 0 if supported, or 1
    logger.debug(f"Activating Ollama model '{model_name}' at {endpoint}...")
    try:
        response = requests.post(
            endpoint, json=payload, timeout=45
        )  # Shorter timeout for activation check
        if response.ok:
            logger.info(f"Ollama model '{model_name}' active/loaded.")
            return True
        elif response.status_code == 404:  # Model not found by Ollama service
            logger.warning(
                f"Ollama model '{model_name}' not found by service at {base_url}. "
                f"Consider `ollama pull {model_name}` or `ollama run {model_name}` manually."
            )
        else:
            logger.warning(
                f"Ollama activate ping for '{model_name}' failed: {response.status_code} {response.text[:100]}"
            )
    except Exception as e:
        logger.error(f"Error activating Ollama model '{model_name}': {e}")
    return False


# --- ModelSelector Class ---
class ModelSelector:
    # (Implementation from previous LLM handler focused script)
    # Key methods: __init__, _update_model_ratings_and_status, select_model, get_all_models_summary
    def __init__(
        self,
        discovered_models: List[Dict[str, Any]],
        preferences: Optional[List[Dict[str, Any]]] = None,
    ):
        self.all_discovered_models = discovered_models
        self.preferences = (
            preferences if preferences else VS_LLM_CONFIG.MODEL_PREFERENCES
        )
        self._update_model_ratings_and_status()  # Initial rating

    def _update_model_ratings_and_status(self):
        for model_info in self.all_discovered_models:
            if model_info.get("status") != "responsive":
                model_info["dynamic_score"] = -1000
                continue

            perf = get_model_performance_summary(
                model_info["id"], model_info["service"], model_info["base_url"]
            )
            model_info["performance"] = perf
            score = (
                model_info.get(
                    "strength_tier", VS_LLM_CONFIG.DEFAULT_MODEL_STRENGTH_TIER
                )
                * 20.0
            )
            probed_caps = model_info.get(
                "probed_capabilities", model_info.get("capabilities", [])
            )
            if "json_confirmed" in probed_caps:
                score += 20  # Higher bonus for confirmed JSON
            elif "json" in probed_caps:
                score += 10
            if "vision" in probed_caps:
                score += 25

            if perf.get("success_rate") is not None:
                score *= 0.6 + 0.4 * perf["success_rate"]
            if perf.get("avg_tps") is not None and perf["avg_tps"] > 0:
                score += min(20, perf["avg_tps"] * 0.4)
            if perf.get("avg_latency_ms") is not None and perf["avg_latency_ms"] > 0:
                score -= min(25, perf["avg_latency_ms"] / 750.0)
            if (
                perf.get("error_count", 0) > perf.get("total_requests", 0) * 0.25
                and perf.get("total_requests", 0) > 4
            ):
                score *= 0.6

            for pref_item in self.preferences:  # Apply user preferences
                name_contains = pref_item.get("name_contains", "")
                service_pref = pref_item.get("service")
                if name_contains and name_contains.lower() in model_info["id"].lower():
                    if not service_pref or (
                        isinstance(service_pref, str)
                        and service_pref.lower() == model_info["service"].lower()
                    ):
                        score *= pref_item.get("boost", 1.0)
            model_info["dynamic_score"] = round(max(0, score), 2)

    def select_model(
        self, task_requirements: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        self._update_model_ratings_and_status()  # Refresh ratings
        min_tier = task_requirements.get("min_strength_tier", 1)
        req_caps = set(task_requirements.get("required_capabilities", []))
        # ... (Filtering logic from previous) ...
        candidates = [
            m
            for m in self.all_discovered_models
            if m.get("dynamic_score", -1) >= 0
            and m.get("strength_tier", 0) >= min_tier
            and req_caps.issubset(
                set(m.get("probed_capabilities", m.get("capabilities", [])))
            )
            and (
                not task_requirements.get("preferred_service")
                or m.get("service") == task_requirements.get("preferred_service")
            )
        ]
        if not candidates:
            logger.warning(f"No models meet requirements: {task_requirements}")
            return None

        candidates.sort(key=lambda x: x.get("dynamic_score", 0), reverse=True)
        selected = candidates[0]  # Ollama activation attempt
        if selected["service"] == "ollama" and not ensure_ollama_model_active(
            selected["id"], selected["base_url"]
        ):
            logger.warning(
                f"Failed to activate selected Ollama model {selected['id']}. Trying next best if available."
            )
            if len(candidates) > 1:
                selected = candidates[1]
                logger.info(f"Fell back to: {selected['id']}")
            else:
                logger.error(
                    f"No fallback model for failed Ollama activation of {selected['id']}"
                )
                return None
            if selected["service"] == "ollama" and not ensure_ollama_model_active(
                selected["id"], selected["base_url"]
            ):  # Try again
                logger.error(
                    f"Fallback Ollama model {selected['id']} also failed activation."
                )
                return None

        logger.info(
            f"Selected model: {selected['id']} ({selected['service']}) Score: {selected['dynamic_score']:.2f}"
        )
        return selected

    def get_all_models_summary(self) -> List[Dict[str, Any]]:
        self._update_model_ratings_and_status()
        return sorted(
            self.all_discovered_models,
            key=lambda m: m.get("dynamic_score", 0),
            reverse=True,
        )


# --- Global State and Initialization ---
_GLOBAL_MODEL_SELECTOR: Optional[ModelSelector] = None
_VOXSIGIL_SYSTEM_PROMPT_TEXT_GLOBAL: str = (
    "Default VoxSigil System Prompt: Be a helpful AI assistant."
)


def initialize_llm_handler(
    use_voxsigil_system_prompt: bool = True, force_discover_models: bool = False
) -> bool:
    """Initializes the LLM Handler's global state."""
    global _GLOBAL_MODEL_SELECTOR, _VOXSIGIL_SYSTEM_PROMPT_TEXT_GLOBAL
    if not HAVE_REQUESTS or requests is None:
        logger.critical(
            "LLM Handler requires 'requests' library. Initialization failed."
        )
        return False

    if use_voxsigil_system_prompt and HAVE_VOXSIGIL_LOADER:
        try:
            _VOXSIGIL_SYSTEM_PROMPT_TEXT_GLOBAL = load_voxsigil_system_prompt(
                VS_LLM_CONFIG.VOXSIGIL_LIBRARY_PATH_FOR_PROMPT
            )
            logger.info(
                f"VoxSigil system prompt loaded ({count_tokens(_VOXSIGIL_SYSTEM_PROMPT_TEXT_GLOBAL)} tokens)."
            )
        except Exception as e_sp:
            logger.error(f"Failed to load VoxSigil system prompt: {e_sp}")

    if force_discover_models and VS_LLM_CONFIG.DISCOVERED_MODELS_CACHE_FILE.exists():
        try:
            os.remove(VS_LLM_CONFIG.DISCOVERED_MODELS_CACHE_FILE)
            logger.info("Forced model re-discovery.")
        except OSError as e_rm:
            logger.error(f"Could not remove model discovery cache: {e_rm}")

    discovered_models_list = query_all_services_for_models()
    if not discovered_models_list:
        logger.error("CRITICAL: No LLMs were discovered. LLM calls will likely fail.")

    _GLOBAL_MODEL_SELECTOR = ModelSelector(
        discovered_models_list
    )  # Preferences are in VS_LLM_CONFIG

    summary_list = _GLOBAL_MODEL_SELECTOR.get_all_models_summary()
    logger.info(
        f"LLM Handler initialized. {len(summary_list)} models processed for selection."
    )
    if summary_list:
        logger.info("Top 3 models by current dynamic score:")
        for m in summary_list[:3]:
            logger.info(
                f"  - {m['id']} ({m['service']}@{m['base_url']}), Tier: {m.get('strength_tier', '?')}, Score: {m.get('dynamic_score', 0):.1f}, Caps: {m.get('probed_capabilities', m.get('capabilities'))}"
            )
    return bool(summary_list)


# --- Main External API Function ---
def llm_chat_completion(
    user_prompt: str,
    task_requirements: Optional[Dict[str, Any]] = None,
    system_prompt_override: Optional[str] = None,
    use_global_voxsigil_system_prompt: bool = True,
    temperature: float = 0.2,  # Sensible default temperature
    # This function assumes the user_prompt might already be RAG-enhanced by middleware.
    # It does not perform RAG itself.
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    High-level function for LLM chat completion. Selects a model dynamically.
    Returns: (llm_content_string, selected_model_config_dict, full_raw_api_response_dict)
    """
    if not _GLOBAL_MODEL_SELECTOR:
        logger.warning("LLM Handler not initialized. Attempting auto-initialization...")
        if not initialize_llm_handler(
            use_voxsigil_system_prompt=use_global_voxsigil_system_prompt
        ):
            logger.error("Auto-initialization of LLM Handler failed. Cannot proceed.")
            return None, None, {"error": "LLM_HANDLER_NOT_INITIALIZED"}

    current_task_reqs = (
        task_requirements
        if task_requirements is not None
        else {"required_capabilities": ["general"]}
    )  # Minimal default

    # Type ignore because _GLOBAL_MODEL_SELECTOR check above should ensure it's not None
    selected_model_config = _GLOBAL_MODEL_SELECTOR.select_model(current_task_reqs)  # type: ignore

    if not selected_model_config:
        logger.error(
            f"Failed to select a model for task: {current_task_reqs}. Trying with relaxed requirements."
        )
        # Attempt relaxation: remove min_strength_tier if present, or lower it
        relaxed_reqs = current_task_reqs.copy()
        if "min_strength_tier" in relaxed_reqs:
            del relaxed_reqs["min_strength_tier"]
        else:
            relaxed_reqs["min_strength_tier"] = [
                str(1)
            ]  # Add it as a list containing a string
        selected_model_config = _GLOBAL_MODEL_SELECTOR.select_model(relaxed_reqs)  # type: ignore
        if not selected_model_config:
            logger.error(
                f"Model selection failed even with relaxed requirements: {relaxed_reqs}"
            )
            return (
                None,
                None,
                {
                    "error": "NO_MODEL_SELECTED",
                    "requirements_tried": [current_task_reqs, relaxed_reqs],
                },
            )

    messages: List[Dict[str, str]] = []
    final_system_prompt_text = system_prompt_override
    if final_system_prompt_text is None and use_global_voxsigil_system_prompt:
        final_system_prompt_text = _VOXSIGIL_SYSTEM_PROMPT_TEXT_GLOBAL

    if final_system_prompt_text:
        messages.append({"role": "system", "content": final_system_prompt_text})

    messages.append(
        {"role": "user", "content": user_prompt}
    )  # user_prompt is expected to be final (e.g. RAG enhanced)

    llm_content, raw_api_response = _llm_call_api_internal(
        selected_model_config, messages, temperature
    )  # Use internal name

    # Augment raw_api_response with handler metadata for transparency
    if isinstance(raw_api_response, dict):
        raw_api_response.setdefault("voxsigil_handler_metadata", {}).update(
            {
                "selected_model_id": selected_model_config.get("id"),
                "selected_model_service": selected_model_config.get("service"),
                "selected_model_base_url": selected_model_config.get("base_url"),
                "final_user_prompt_token_count": count_tokens(
                    user_prompt
                ),  # Tokens of the prompt given to this handler
            }
        )

    return llm_content, selected_model_config, raw_api_response


# --- Internal `_llm_call_api_internal` (was `call_llm_api` before) ---
def _llm_call_api_internal(  # Renamed to distinguish from a potentially more abstract public API
    model_config: Dict[str, Any],
    messages_payload: List[Dict[str, str]],
    temperature: float,
    retry_attempt: int = 1,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Internal core function to call an LLM API."""
    # (Full implementation of the original call_llm_api including caching, retries, performance logging)
    if not HAVE_REQUESTS or requests is None:
        return None, {"error": "requests_library_missing"}

    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    model_id_for_api = model_config["id"]
    service = model_config["service"].lower()
    base_url = model_config["base_url"]
    model_timeout = VS_LLM_CONFIG.LLM_REQUEST_TIMEOUT_SECONDS
    if model_config.get("strength_tier", 0) >= 5:
        model_timeout = max(
            model_timeout, 480
        )  # Increased from 300 to 480 seconds for large models
    elif model_config.get("strength_tier", 0) == 4:
        model_timeout = max(
            model_timeout, 360
        )  # Increased from 240 to 360 seconds for medium-large models

    # Also check model name for size indicators (32b, 65b, 70b, 100b, etc.)
    if any(size in model_id_for_api.lower() for size in ["32b", "65b", "70b", "100b"]):
        model_timeout = max(
            model_timeout, 480
        )  # Ensure at least 8 minutes for large models

    chat_ep_path = VS_LLM_CONFIG.DEFAULT_SERVICE_ENDPOINTS.get(service, {}).get("chat")
    if not chat_ep_path:
        return None, {"error": f"No chat endpoint for {service}"}
    endpoint = f"{base_url.rstrip('/')}{chat_ep_path}"

    payload: Dict[str, Any] = {
        "model": model_id_for_api,
        "messages": messages_payload,
        "stream": False,
        "temperature": temperature,
    }
    if temperature > 0.01 and service in ["ollama", "openai_compatible", "lmstudio"]:
        seed_val = random.randint(1, 2**31 - 1)
        if service == "ollama":
            payload.setdefault("options", {})["seed"] = seed_val
        else:
            payload["seed"] = seed_val

    probed_caps = model_config.get(
        "probed_capabilities", model_config.get("capabilities", [])
    )
    if "json_confirmed" in probed_caps or (
        "json" in probed_caps and model_config.get("strength_tier", 0) > 2
    ):  # Prefer confirmed json
        is_vision_model = "vision" in probed_caps or any(
            vm_kw in model_id_for_api.lower()
            for vm_kw in ["llava", "bakllava", "moondream"]
        )
        if service == "ollama":
            payload["format"] = "json"
        elif service in ["lmstudio", "openai_compatible"] and not is_vision_model:
            payload["response_format"] = {"type": "json_object"}

    content_hash_for_cache = hashlib.sha256(
        json.dumps(messages_payload, sort_keys=True).encode()
    ).hexdigest()
    resp_format_hash_for_cache = hashlib.sha256(
        json.dumps(payload.get("response_format", {}), sort_keys=True).encode()
    ).hexdigest()[:8]

    cache_key = get_llm_cache_key(
        f"{content_hash_for_cache}|{resp_format_hash_for_cache}",
        model_id_for_api,
        temperature,
        service,
    )

    if VS_LLM_CONFIG.USE_LLM_RESPONSE_CACHE:
        from .arc_utils import get_cached_response

        cached_response = get_cached_response(cache_key)
        if cached_response is not None:
            logger.info(
                f"LLM CACHE HIT for {model_id_for_api}@{base_url} (KeyHash:{cache_key[:8]})"
            )
            return cached_response[0], cached_response[1]

    logger.debug(
        f"API Call: {service.upper()}/{model_id_for_api}@{base_url} (Attempt {retry_attempt}, T {temperature:.2f})"
    )

    start_mono = time.monotonic()
    start_wall = datetime.now(timezone.utc)
    http_status: Optional[int] = None
    error_type: Optional[str] = None
    raw_resp: Dict[str, Any] = {}
    llm_content: Optional[str] = None

    try:
        r = requests.post(
            endpoint, json=payload, headers=headers, timeout=model_timeout
        )
        http_status = r.status_code
        r.raise_for_status()
        raw_resp = r.json()
        if service == "ollama":
            llm_content = (
                json.dumps(raw_resp)
                if payload.get("format") == "json"
                else raw_resp.get("message", {}).get("content")
            )
        else:
            llm_content = (
                raw_resp.get("choices", [{}])[0].get("message", {}).get("content")
            )
        if llm_content is None:
            error_type = "UnexpectedResponseStructure"
    except requests.exceptions.Timeout:
        error_type = "Timeout"
    except requests.exceptions.HTTPError as e:
        http_status = e.response.status_code
        error_type = f"HTTPError_{http_status}"
    except requests.exceptions.RequestException:
        error_type = "RequestException"
    except json.JSONDecodeError:
        error_type = "JSONDecodeError"
    except Exception as e:
        error_type = type(e).__name__
        logger.error(f"LLM GenErr: {e}", exc_info=True)

    duration_ms = int((time.monotonic() - start_mono) * 1000)
    input_toks = sum(
        count_tokens(m["content"], model_id_for_api) for m in messages_payload
    )
    output_toks = count_tokens(llm_content, model_id_for_api) if llm_content else 0
    api_usage = raw_resp.get("usage", {})
    final_in_toks = api_usage.get("prompt_tokens", input_toks)
    final_out_toks = api_usage.get("completion_tokens", output_toks)
    tps_val = (
        (final_out_toks / (duration_ms / 1000.0))
        if final_out_toks > 0 and duration_ms > 0
        else 0.0
    )

    log_performance_stat(
        {
            "model_id": model_id_for_api,
            "service": service,
            "base_url": base_url,
            "timestamp": start_wall.isoformat(),
            "request_duration_ms": duration_ms,
            "input_tokens": final_in_toks,
            "output_tokens": final_out_toks,
            "tokens_per_second": tps_val,
            "success": llm_content is not None and error_type is None,
            "error_type": error_type,
            "http_status_code": http_status,
            "temperature": temperature,
            "retry_attempt": retry_attempt,
        }
    )

    if llm_content is not None:
        if VS_LLM_CONFIG.USE_LLM_RESPONSE_CACHE:
            from .arc_utils import cache_response

            cache_response(cache_key, (llm_content, raw_resp))
        return llm_content, raw_resp

    if retry_attempt < VS_LLM_CONFIG.LLM_MAX_RETRIES:
        delay = (
            VS_LLM_CONFIG.LLM_RETRY_DELAY_SECONDS * (2 ** (retry_attempt - 1))
        ) + random.uniform(0, 1)
        logger.warning(
            f"LLM call {model_id_for_api} err: {error_type}. Retrying in {delay:.1f}s ({retry_attempt + 1}/{VS_LLM_CONFIG.LLM_MAX_RETRIES})"
        )
        time.sleep(delay)
        return _llm_call_api_internal(
            model_config, messages_payload, temperature, retry_attempt + 1
        )

    logger.error(
        f"LLM call {model_id_for_api} failed after {VS_LLM_CONFIG.LLM_MAX_RETRIES} attempts. Last error: {error_type}"
    )
    return None, raw_resp if raw_resp else {
        "error": error_type or "MaxRetriesReached",
        "status_code": http_status,
    }


def call_llm_api(
    model_config: Dict[str, Any],
    messages_payload: List[Dict[str, str]],
    temperature: float,
    retry_attempt: int = 1,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """External API function to call an LLM API. Wraps the internal implementation.

    This is provided for backward compatibility with existing code.

    Args:
        model_config: Configuration for the model to call
        messages_payload: List of message dictionaries to send to the API
        temperature: Temperature parameter to control randomness
        retry_attempt: Current retry attempt number

    Returns:
        Tuple containing the LLM response content and the raw API response
    """
    return _llm_call_api_internal(
        model_config=model_config,
        messages_payload=messages_payload,
        temperature=temperature,
        retry_attempt=retry_attempt,
    )


# --- ARC Grid Parsing (simplified stubs, assume full implementation exists if needed) ---
def robust_parse_arc_grid_from_llm_text(
    llm_text_output: Optional[str], service: str = "unknown"
) -> Optional[List[List[int]]]:
    # This is where the robust parsing logic from your original script would go.
    # For now, a very simple placeholder:
    if not llm_text_output:
        return None
    try:
        # A real implementation would use the multi-stage regex and JSON cleaning.
        # This placeholder just tries direct JSON load or looks for a simple list of lists.
        data = json.loads(llm_text_output)
        if isinstance(data, dict) and "predicted_grid" in data:
            grid_candidate = data["predicted_grid"]
        elif isinstance(data, list):
            grid_candidate = data
        else:
            grid_candidate = None

        if (
            grid_candidate
            and isinstance(grid_candidate, list)
            and all(isinstance(row, list) for row in grid_candidate)
        ):
            return [[int(c) for c in row] for row in grid_candidate]  # Basic conversion
    except json.JSONDecodeError as e:  # Broad except for placeholder
        logger.debug(f"JSONDecodeError: {e}")
        # Fallback to regex for [[...],[...]] pattern
        match = re.search(
            r"(\[(?:\s*\[[^\]]*\],?\s*)*\s*\[[^\]]*\]\s*\])", llm_text_output
        )
        if match:
            try:
                grid_from_regex = json.loads(match.group(1))
                if isinstance(grid_from_regex, list) and all(
                    isinstance(row, list) for row in grid_from_regex
                ):
                    return [[int(c) for c in row] for row in grid_from_regex]
            except json.JSONDecodeError as e:
                logger.debug(f"JSONDecodeError during regex parsing: {e}")
                pass
    logger.debug(
        f"Robust ARC grid parsing did not find a grid in output from {service}."
    )
    return None


# --- Main Demo Block ---
if __name__ == "__main__":
    # This demo assumes an Ollama server is running on http://localhost:11434
    # and has at least one model like 'mistral' or 'phi3' available.
    logger.info("--- VoxSigil LLM Handler Standalone Demo ---")

    # Setup a dummy VoxSigil library for system prompt loading by the imported loader
    demo_sys_prompt_lib_path = VS_LLM_CONFIG.CACHE_DIR / "DemoSysPromptLib"
    (demo_sys_prompt_lib_path / "schema").mkdir(parents=True, exist_ok=True)
    with open(
        demo_sys_prompt_lib_path / "schema" / "voxsigil-schema-current.yaml", "w"
    ) as f:
        yaml.dump(
            {
                "system_prompt": "DEMO SYSTEM PROMPT: Focus on clarity and conciseness for VoxSigil tasks."
            },
            f,
        )

    # Point the imported loader (if available) to this dummy lib for the demo
    if HAVE_VOXSIGIL_LOADER:
        # This relies on VOXSIGIL_LIBRARY_PATH being a module-level var in voxsigil_loader
        # If it's an instance var, this approach needs adjustment.
        # For a cleaner approach, load_voxsigil_system_prompt should accept a path.
        # The function `load_voxsigil_system_prompt` already takes library_path_override.
        # So, we pass it to initialize_llm_handler which then passes it to loader.
        library_path = VS_LLM_CONFIG.VOXSIGIL_LIBRARY_PATH_FOR_PROMPT

    # Initialize the handler (discovers models, loads system prompt etc.)
    # Use force_discover_models=True to bypass cache for first run of demo.
    init_ok = initialize_llm_handler(
        use_voxsigil_system_prompt=True, force_discover_models=True
    )

    if (
        not init_ok
        or not _GLOBAL_MODEL_SELECTOR
        or not _GLOBAL_MODEL_SELECTOR.get_all_models_summary()
    ):
        logger.critical(
            "LLM Handler failed to initialize or no models found. Demo cannot proceed."
        )
        if HAVE_REQUESTS:
            logger.info(
                "Please ensure an LLM service (like Ollama) is running and accessible."
            )
        else:
            logger.critical(
                "'requests' library is missing. Please install it: pip install requests"
            )
        exit(1)

    logger.info("\n--- Test 1: General Chat Completion (ModelSelector chooses) ---")
    task_req1 = {"min_strength_tier": 2, "required_capabilities": ["general", "json"]}
    prompt1 = 'Explain the concept of object-oriented programming in simple terms, providing a small JSON example like {"class": "Animal", "properties": ["name", "sound"] }.'

    content1, model_cfg1, raw_resp1 = llm_chat_completion(
        user_prompt=prompt1, task_requirements=task_req1, temperature=0.3
    )

    if content1 and model_cfg1:
        logger.info(
            f"Response from [{model_cfg1['id']}] ({model_cfg1['service']}@{model_cfg1['base_url']}):"
        )
        logger.info(content1[:400] + ("..." if len(content1) > 400 else ""))
        # logger.info(f"Full Raw API Response Keys: {list(raw_resp1.keys()) if raw_resp1 else 'N/A'}")
        handler_meta1 = (
            raw_resp1.get("voxsigil_handler_metadata", {}) if raw_resp1 else {}
        )
        logger.info(f"Handler metadata: {handler_meta1}")

        # Attempt to parse JSON if the model was supposed to output it
        if "json" in model_cfg1.get(
            "probed_capabilities", model_cfg1.get("capabilities", [])
        ):
            try:
                # Basic clean and parse (a more robust cleaner might be needed for some models)
                json_like_part = content1
                if "```json" in content1:
                    json_like_part = content1.split("```json")[1].split("```")[0]
                elif "```" in content1:
                    json_like_part = content1.split("```")[1].split("```")[
                        0
                    ]  # Generic code block
                parsed_json = json.loads(json_like_part.strip())
                logger.info(f"Successfully parsed JSON from response: {parsed_json}")
            except json.JSONDecodeError:
                logger.warning(
                    "Response was expected to be JSON or contain JSON, but failed to parse."
                )
    else:
        logger.warning(
            "Test 1: Simple chat completion call failed or returned no content."
        )

    logger.info("\n--- Test 2: Another call, check for cache or performance update ---")
    # (This would ideally be with different params to avoid cache, or check perf stats update)
    prompt2 = "What is the capital of France?"
    task_req2 = {"min_strength_tier": 1, "required_capabilities": ["general"]}
    content2, model_cfg2, raw_resp2 = llm_chat_completion(
        user_prompt=prompt2, task_requirements=task_req2, temperature=0.0
    )
    if content2 and model_cfg2:
        logger.info(f"Response from [{model_cfg2['id']}]: {content2.strip()}")

    logger.info(
        "\n--- Test 3: Task requiring specific capability (e.g., vision if a model supports it) ---"
    )
    task_req_vision = {
        "required_capabilities": ["vision", "general"],
        "min_strength_tier": 3,
    }
    prompt_vision = "Describe this image (imagine an image was provided)."  # Placeholder for actual vision call

    content_vision, model_cfg_vision, _ = llm_chat_completion(
        user_prompt=prompt_vision, task_requirements=task_req_vision
    )
    if (
        model_cfg_vision
    ):  # Even if content is None due to no actual image, selection might succeed
        logger.info(
            f"Model selected for vision task: {model_cfg_vision['id']}. (Actual vision call not performed in this demo)."
        )
        if content_vision:
            logger.info(f"  Response: {content_vision.strip()}")
    else:
        logger.warning("No suitable model found/selected for vision task.")

    logger.info("\n--- Displaying current model performance summaries (last 24h) ---")
    if _GLOBAL_MODEL_SELECTOR:
        sorted_models = _GLOBAL_MODEL_SELECTOR.get_all_models_summary()
        for m_summary in sorted_models[:5]:  # Show top 5
            perf = m_summary.get("performance", {})
            logger.info(
                f"Model: {m_summary['id']} ({m_summary['service']}@{m_summary['base_url']}) "
                f"Score: {m_summary.get('dynamic_score', 0):.1f} Tier: {m_summary.get('strength_tier', '?')} "
                f"Status: {m_summary.get('status', 'unknown')} "
                f"Perf (Requests: {perf.get('total_requests', 0)}, "
                f"Success: {perf.get('success_rate', 'N/A')}, "
                f"AvgLat: {perf.get('avg_latency_ms', 'N/A')}ms, "
                f"AvgTPS: {perf.get('avg_tps', 'N/A')})"
            )

    # Note: RAG injection itself is now assumed to be handled by an upstream component
    # (e.g., VoxSigilMiddleware) which would then pass the RAG-enhanced prompt
    # to `llm_chat_completion`'s `user_prompt` argument.
    # The `voxsigil_rag_instance` and `rag_params` in `llm_chat_completion` are
    # for convenience if this handler is used in a simpler setup where it *also*
    # needs to trigger RAG.

    # Cleanup dummy lib for system prompt
    try:
        import shutil

        if demo_sys_prompt_lib_path.exists():
            shutil.rmtree(demo_sys_prompt_lib_path)
        logger.info(
            f"Cleaned up demo system prompt library: {demo_sys_prompt_lib_path}"
        )
    except Exception as e_clean:
        logger.error(f"Error cleaning up demo lib: {e_clean}")

    logger.info("--- VoxSigil LLM Handler Standalone Demo Finished ---")


def initialize_and_validate_models_config(
    force_discover_models: bool = False,
    use_voxsigil_system_prompt: bool = True,
    min_required_model_tier: int = 2,
) -> Dict[str, Any]:
    """
    Initializes and validates LLM model configurations for ARC tasks.
    This function ensures the LLM Handler is properly set up with valid models.

    Args:
        force_discover_models: If True, forces re-discovery of models instead of using cache
        use_voxsigil_system_prompt: If True, loads and uses the VoxSigil system prompt
        min_required_model_tier: Minimum model tier required for ARC tasks

    Returns:
        Dict containing configuration status and validated models information
    """
    global _GLOBAL_MODEL_SELECTOR

    logger.info(
        f"Initializing and validating models configuration for ARC (min tier: {min_required_model_tier})"
    )

    result = {
        "success": False,
        "available_models": [],
        "suitable_models": [],
        "strongest_model": None,
        "error": None,
    }

    # Initialize the LLM handler
    init_success = initialize_llm_handler(
        use_voxsigil_system_prompt=use_voxsigil_system_prompt,
        force_discover_models=force_discover_models,
    )

    if not init_success or not _GLOBAL_MODEL_SELECTOR:
        result["error"] = (
            "Failed to initialize LLM handler or no models were discovered."
        )
        logger.error(result["error"])
        return result

    # Get all discovered models
    all_models = _GLOBAL_MODEL_SELECTOR.get_all_models_summary()
    result["available_models"] = [
        {"id": m["id"], "service": m["service"], "tier": m.get("strength_tier", 0)}
        for m in all_models
    ]

    # Filter for suitable models that meet minimum requirements for ARC
    suitable_models = [
        m
        for m in all_models
        if m.get("status") == "responsive"
        and m.get("strength_tier", 0) >= min_required_model_tier
        and all(
            cap in m.get("probed_capabilities", m.get("capabilities", []))
            for cap in ["general", "json"]
        )
    ]

    result["suitable_models"] = [
        {"id": m["id"], "service": m["service"], "tier": m.get("strength_tier", 0)}
        for m in suitable_models
    ]

    if not suitable_models:
        result["error"] = (
            f"No suitable models found meeting minimum tier {min_required_model_tier} requirement."
        )
        logger.warning(result["error"])
        return result

    # Sort by dynamic score and get the strongest model
    suitable_models.sort(key=lambda x: x.get("dynamic_score", 0), reverse=True)
    strongest = suitable_models[0]
    result["strongest_model"] = {
        "id": strongest["id"],
        "service": strongest["service"],
        "tier": strongest.get("strength_tier", 0),
        "score": strongest.get("dynamic_score", 0),
    }

    # If strongest model is Ollama, ensure it's active
    if strongest["service"] == "ollama":
        if not ensure_ollama_model_active(strongest["id"], strongest["base_url"]):
            logger.warning(
                f"Failed to activate Ollama model {strongest['id']}. Will use next best if available."
            )
            if len(suitable_models) > 1:
                second_best = suitable_models[1]
                result["strongest_model"] = {
                    "id": second_best["id"],
                    "service": second_best["service"],
                    "tier": second_best.get("strength_tier", 0),
                    "score": second_best.get("dynamic_score", 0),
                }
                logger.info(f"Fallback to next best model: {second_best['id']}")
            else:
                result["error"] = (
                    "Failed to activate strongest model and no fallbacks available."
                )
                logger.error(result["error"])
                return result

    logger.info(
        f"Model validation successful. Strongest model: {result['strongest_model']['id']} "
        f"(Tier: {result['strongest_model']['tier']}, "
        f"Service: {result['strongest_model']['service']})"
    )

    result["success"] = True
    return result
