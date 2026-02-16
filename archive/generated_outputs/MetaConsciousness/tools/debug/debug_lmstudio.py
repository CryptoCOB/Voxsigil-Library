"""
Debug tool for LM Studio connection issues.

This script provides diagnostic functions and fixes for LM Studio integration.
"""

import os
import sys
import json
import time
from typing import Any, Dict, List, Tuple
import requests
import argparse
from pathlib import Path

# Add the parent directory to sys.path to enable imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try importing the LMStudioAdapter
try:
    from MetaConsciousness.interface.lmstudio_adapter import LMStudioAdapter
    lmstudio_available = True
except ImportError:
    from MetaConsciousness.models.lmstudio.adapter import LMStudioAdapter
    lmstudio_available = True
    print("Imported LMStudioAdapter from direct path")
except Exception as e:
    lmstudio_available = False
    print(f"Error importing LMStudioAdapter: {e}")

def check_lmstudio_service(url="http://localhost:1234/v1") -> Tuple[bool, str]:
    """
    Check if LM Studio API is accessible.

    Args:
        url: API URL to check

    Returns:
        tuple: (success, message)
    """
    print(f"Checking LM Studio service at {url}...")

    try:
        # Try to connect to the models endpoint
        response = requests.get(f"{url}/models", timeout=5)

        if response.status_code == 200:
            models_data = response.json()
            if isinstance(models_data, dict) and "data" in models_data:
                models = [model.get("id", "") for model in models_data["data"]]
                if models:
                    return True, f"✅ Connected successfully. Found {len(models)} models."
                else:
                    return False, "⚠️ Connected to API but no models are available. Start a model in LM Studio."
            else:
                return False, "⚠️ Connected but received unexpected data format. Check API compatibility."
        else:
            return False, f"❌ API endpoint returned status code {response.status_code}. Check if API server is running."
    except requests.exceptions.ConnectionError:
        return False, "❌ Connection refused. LM Studio API server is not running."
    except requests.exceptions.Timeout:
        return False, "❌ Connection timeout. LM Studio API server is not responding."
    except Exception as e:
        return False, f"❌ Error checking LM Studio service: {e}"

def test_lmstudio_completion(url="http://localhost:1234/v1", model="default") -> None:
    """
    Test a basic completion with LM Studio.

    Args:
        url: API URL
        model: Model name

    Returns:
        tuple: (success, message, response_text)
    """
    print(f"Testing completion with model: {model}...")

    try:
        headers = {"Content-Type": "application/json"}
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"}
            ],
            "temperature": 0.7,
            "max_tokens": 50
        }

        response = requests.post(
            f"{url}/chat/completions",
            headers=headers,
            json=data,
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            response_text = result["choices"][0]["message"]["content"]
            return True, "✅ Completion test successful", response_text
        else:
            return False, f"❌ Completion failed with status code: {response.status_code}", response.text
    except Exception as e:
        return False, f"❌ Error during completion test: {e}", None

def test_adapter() -> None:
    """
    Test the LMStudioAdapter functionality.

    Returns:
        bool: True if test passed
    """
    if not lmstudio_available:
        print("❌ LMStudioAdapter not available. Cannot run adapter test.")
        return False

    print("Testing LMStudioAdapter...")

    try:
        # Initialize adapter with fallback mode
        adapter = LMStudioAdapter(fallback_mode=True)

        # Test connection
        print(f"Connection status: {adapter._is_connected}")

        # Test fallback mode
        fallback_response = adapter._fallback_response("Test prompt", "Test system prompt")
        print(f"Fallback response working: {'✅' if fallback_response else '❌'}")

        # Test vigilance suggestion
        vigilance = adapter.suggest_vigilance("checkerboard", "high")
        print(f"Vigilance suggestion: {vigilance}")

        # Test trace explanation
        explanation = adapter.explain_trace([{"test": "data"}])
        print(f"Trace explanation length: {len(explanation)} chars")

        print("✅ Adapter tests completed successfully")
        return True
    except Exception as e:
        print(f"❌ Error testing adapter: {e}")
        return False

def test_adapter_details() -> None:
    """Test the LMStudioAdapter functionality with detailed status."""
    if not lmstudio_available:
        print("❌ LMStudioAdapter not available. Cannot run adapter test.")
        return False

    print("Testing LMStudioAdapter with detailed status...")

    try:
        # Initialize adapter with fallback mode and shorter timeout
        adapter = LMStudioAdapter(fallback_mode=True, timeout=3)

        # Get detailed connection status
        if hasattr(adapter, 'get_connection_status'):
            status = adapter.get_connection_status()
            print("\nConnection Details:")
            for key, value in status.items():
                print(f"  {key}: {value}")
        else:
            print(f"Connection status: {adapter._is_connected}")

        # Test fallback mode
        fallback_response = adapter._fallback_response("Test prompt", "Test system prompt")
        print(f"Fallback response working: {'✅' if fallback_response else '❌'}")

        # Test vigilance suggestion with timing
        start_time = time.time()
        vigilance = adapter.suggest_vigilance("checkerboard", "high")
        suggestion_time = time.time() - start_time
        print(f"Vigilance suggestion: {vigilance} (took {suggestion_time:.2f}s)")

        # Test trace explanation with timing
        start_time = time.time()
        explanation = adapter.explain_trace([{"test": "data"}])
        explanation_time = time.time() - start_time
        print(f"Trace explanation length: {len(explanation)} chars (took {explanation_time:.2f}s)")

        print("✅ Adapter tests completed successfully")
        return True
    except Exception as e:
        print(f"❌ Error testing adapter: {e}")
        return False

def fix_adapter_import() -> None:
    """
    Fix the LMStudioAdapter import issue.

    Returns:
        bool: True if fix was successful
    """
    # Check if the adapter file exists in the expected location
    sdk_path = Path(os.path.dirname(os.path.abspath(__file__))) / "MetaConsciousness"
    models_path = sdk_path / "models" / "lmstudio"
    adapter_path = models_path / "adapter.py"

    # Check if bridge module exists
    bridge_path = sdk_path / "lmstudio_adapter.py"

    print(f"Checking adapter at: {adapter_path}")
    print(f"Checking bridge at: {bridge_path}")

    # If adapter exists but bridge doesn't, create the bridge
    if adapter_path.exists() and not bridge_path.exists():
        print("Creating bridge module for LMStudioAdapter...")

        # Ensure directory exists
        if not bridge_path.parent.exists():
            bridge_path.parent.mkdir(parents=True, exist_ok=True)

        # Create the bridge module
        with open(bridge_path, 'w', encoding='utf-8') as f:
            f.write('''"""
LM Studio Adapter for the MetaConsciousness SDK.

This module provides integration with LM Studio for enhanced metacognitive reasoning.
"""

# Re-export the LMStudioAdapter from its actual location
from .models.lmstudio.adapter import LMStudioAdapter

# Export only the adapter class
__all__ = ['LMStudioAdapter']
''')
        print(f"✅ Created bridge module at {bridge_path}")
        return True

    # If adapter doesn't exist, warning
    elif not adapter_path.exists():
        print(f"❌ Adapter file not found at {adapter_path}")
        return False

    # If bridge already exists
    elif bridge_path.exists():
        print(f"✅ Bridge module already exists at {bridge_path}")
        return True

    return False

def check_models_info(url="http://localhost:1234/v1") -> Tuple[bool, str, List[Dict[str, Any]]]:
    """
    Check model information and recommend timeouts.

    Args:
        url: API URL to check

    Returns:
        tuple: (success, message, model_list)
    """
    print(f"Checking for available models...")

    try:
        # Try to connect to the models endpoint
        response = requests.get(f"{url}/models", timeout=5)

        if response.status_code == 200:
            models_data = response.json()
            if isinstance(models_data, dict) and "data" in models_data:
                models = models_data["data"]
                model_list = []

                # Process model info
                for model in models:
                    if "id" in model:
                        model_id = model["id"]
                        model_size = "large" if any(kw in model_id.lower() for kw in ["32b", "65b", "70b"]) else \
                                    "medium" if any(kw in model_id.lower() for kw in ["7b", "13b", "14b"]) else "small"

                        # Recommended timeout
                        rec_timeout = 60 if model_size == "large" else 30 if model_size == "medium" else 10

                        model_list.append({
                            "id": model_id,
                            "size": model_size,
                            "recommended_timeout": rec_timeout
                        })

                # Format message
                message = f"✅ Found {len(model_list)} models. Showing size and recommended timeouts:\n"
                for model in model_list:
                    message += f"  - {model['id']} ({model['size']}, timeout: {model['recommended_timeout']}s)\n"

                return True, message, model_list

            return True, "✅ Models endpoint returned unexpected format.", []
        else:
            return False, f"❌ Models endpoint returned status code: {response.status_code}", []
    except Exception as e:
        return False, f"❌ Error checking models: {e}", []

# New function to configure task-specific models
def setup_task_specific_models(url="http://localhost:1234/v1") -> Tuple[bool, str, Dict[str, str]]:
    """
    Configure task-specific models based on available models in LM Studio.
    
    Args:
        url: API URL to check
        
    Returns:
        tuple: (success, message, task_model_mapping)
    """
    print("Setting up task-specific models...")
    
    # Get available models
    success, message, model_list = check_models_info(url)
    
    if not success or not model_list:
        return False, f"Failed to get available models: {message}", {}
    
    # Try to import the task configuration module
    try:
        from MetaConsciousness.models.lmstudio.task_config import get_task_config
        task_config = get_task_config()
    except ImportError:
        return False, "Task configuration module not available", {}
    
    # Get model IDs
    available_models = [model["id"] for model in model_list]
    
    # Get suggestions
    suggestions = task_config.suggest_models(available_models)
    
    # Apply suggestions
    success = task_config.apply_suggestions(suggestions)
    
    # Try to configure model router
    try:
        from MetaConsciousness.models.router import default_router
        router_config_success = task_config.configure_model_router(default_router)
        
        if router_config_success:
            print(f"✅ Successfully configured model router with task-specific models")
        else:
            print(f"⚠️ Partial success configuring model router")
    except ImportError:
        print(f"⚠️ Could not configure model router (module not found)")
    
    # Build result message
    result_message = "Task-specific model configuration:\n"
    for task, model in suggestions.items():
        result_message += f"  - {task}: {model}\n"
    
    return success, result_message, suggestions

def analyze_model_complexity(model_name: str) -> Dict[str, Any]:
    """
    Analyze model complexity based on name.

    Args:
        model_name: Model name

    Returns:
        Dict with complexity analysis
    """
    model_name = model_name.lower()

    # Default values
    result = {
        "size_class": "unknown",
        "param_count": None,
        "recommended_timeout": 60,
        "min_timeout": 30
    }

    # Check for parameter count in model name
    param_indicators = {
        "1b": {"size": "tiny", "count": 1, "timeout": 20},
        "2b": {"size": "tiny", "count": 2, "timeout": 20},
        "3b": {"size": "tiny", "count": 3, "timeout": 30},
        "6b": {"size": "small", "count": 6, "timeout": 30},
        "7b": {"size": "small", "count": 7, "timeout": 45},
        "8b": {"size": "small", "count": 8, "timeout": 45},
        "13b": {"size": "medium", "count": 13, "timeout": 60},
        "14b": {"size": "medium", "count": 14, "timeout": 60},
        "20b": {"size": "medium", "count": 20, "timeout": 75},
        "30b": {"size": "large", "count": 30, "timeout": 90},
        "33b": {"size": "large", "count": 33, "timeout": 90},
        "34b": {"size": "large", "count": 34, "timeout": 90},
        "70b": {"size": "xlarge", "count": 70, "timeout": 120},
        "72b": {"size": "xlarge", "count": 72, "timeout": 120}
    }

    # Check for size indicators
    for indicator, details in param_indicators.items():
        # Check for variations like "7b", "7B", "7-b", etc.
        if (
            indicator in model_name or
            indicator.upper() in model_name or
            indicator.replace("b", "-b") in model_name or
            indicator.replace("b", "b-") in model_name or
            indicator.replace("b", " b") in model_name
        ):
            result["size_class"] = details["size"]
            result["param_count"] = details["count"]
            result["recommended_timeout"] = details["timeout"]
            break

    # Check for specific model families that don't follow the pattern
    model_families = {
        "gpt-3.5": {"size": "medium", "count": 175, "timeout": 60},
        "gpt-4": {"size": "large", "count": None, "timeout": 90},
        "claude": {"size": "large", "count": None, "timeout": 90},
        "gemma": {"size": "small", "count": 7, "timeout": 45},
        "llama-3": {"size": "medium", "count": None, "timeout": 60},
        "phi": {"size": "tiny", "count": 2, "timeout": 30},
        "phi-2": {"size": "tiny", "count": 2, "timeout": 30},
        "mistral": {"size": "medium", "count": 7, "timeout": 45}
    }

    for family, details in model_families.items():
        if family in model_name:
            # Only update if we don't already have a size from parameter count
            if result["size_class"] == "unknown":
                result["size_class"] = details["size"]
                result["param_count"] = details["count"]
                result["recommended_timeout"] = details["timeout"]
            break

    # Adjust timeout for quantized models
    if any(q in model_name for q in ["q4", "q5", "q6", "q8", "int4", "int8"]):
        # Quantized models can be faster
        result["recommended_timeout"] = max(20, int(result["recommended_timeout"] * 0.8))

    # Add timeout margin for safety
    result["recommended_timeout"] += 15

    return result

def test_model_generation(api_url="http://localhost:1234/v1", model="default", timeout=30) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Test generating text with the specified model.

    Args:
        api_url: API URL
        model: Model name to test
        timeout: Request timeout in seconds

    Returns:
        Tuple of (success, message, additional data)
    """
    # Analyze model complexity
    model_analysis = analyze_model_complexity(model)
    recommended_timeout = model_analysis["recommended_timeout"]

    # Adjust timeout if less than recommended
    if timeout < recommended_timeout:
        timeout = recommended_timeout
        print(f"ℹ️ Increasing timeout to {timeout}s based on model complexity")

    print(f"Testing generation with model: {model} (timeout: {timeout}s)...")

    try:
        headers = {"Content-Type": "application/json"}
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Write a very short greeting."}
            ],
            "temperature": 0.7,
            "max_tokens": 20  # Keep this small for quicker testing
        }

        start_time = time.time()
        response = requests.post(
            f"{api_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=timeout
        )
        elapsed_time = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]

            # Check for empty content but available reasoning
            if not content.strip() and "reasoning_content" in result["choices"][0]["message"]:
                content = result["choices"][0]["message"]["reasoning_content"]

            return True, f"✅ Generation successful in {elapsed_time:.1f}s", content
        else:
            return False, f"❌ Generation failed with status code: {response.status_code}", response.text
    except requests.exceptions.Timeout:
        return False, f"❌ Generation timed out after {timeout}s. Try increasing timeout for this model.", None
    except Exception as e:
        return False, f"❌ Error during generation test: {e}", None

def extract_reasoning_from_response(response_data) -> None:
    """
    Extract reasoning from model response if available.

    Args:
        response_data: JSON response from LM Studio

    Returns:
        str: Reasoning content or empty string
    """
    try:
        if isinstance(response_data, dict):
            # Check for reasoning in choices
            if "choices" in response_data and response_data["choices"]:
                choice = response_data["choices"][0]

                # Check for reasoning in message
                if "message" in choice:
                    message = choice["message"]

                    # Check various fields where reasoning might be stored
                    if "reasoning_content" in message:
                        return message["reasoning_content"]
                    elif "reasoning" in message:
                        return message["reasoning"]
                    elif "thought_process" in message:
                        return message["thought_process"]

            # Check for reasoning at top level
            for key in ["reasoning", "reasoning_content", "thought_process", "thoughts"]:
                if key in response_data:
                    return response_data[key]

        return ""
    except Exception as e:
        print(f"Error extracting reasoning: {e}")
        return ""

def test_reasoning_extraction(url="http://localhost:1234/v1", model="default") -> None:
    """
    Test extracting reasoning content from model responses.

    Args:
        url: API URL
        model: Model name

    Returns:
        tuple: (success, reasoning_text)
    """
    print(f"Testing reasoning extraction with model: {model}...")

    try:
        headers = {"Content-Type": "application/json"}
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Explain your thinking step by step."},
                {"role": "user", "content": "What is the square root of 16 and why?"}
            ],
            "temperature": 0.7,
            "max_tokens": 150
        }

        response = requests.post(
            f"{url}/chat/completions",
            headers=headers,
            json=data,
            timeout=20
        )

        if response.status_code == 200:
            result = response.json()
            response_text = result["choices"][0]["message"]["content"]
            reasoning = extract_reasoning_from_response(result)

            if reasoning:
                print("✅ Reasoning extraction successful!")
                return True, reasoning
            else:
                # Try to infer reasoning from the response itself
                lines = response_text.split('\n')
                if len(lines) > 1 and any(kw in response_text.lower() for kw in ["reason", "think", "step", "process"]):
                    print("✓ Inferred reasoning from response")
                    return True, response_text
                else:
                    print("ℹ️ No explicit reasoning found, but got a valid response")
                    return False, ""
        else:
            print(f"❌ API call failed with status code: {response.status_code}")
            return False, ""
    except Exception as e:
        print(f"❌ Error testing reasoning extraction: {e}")
        return False, ""

def main() -> None:
    """Main function to run diagnostics."""
    parser = argparse.ArgumentParser(description="Debug LM Studio connection issues")
    parser.add_argument("--url", default="http://localhost:1234/v1",
                      help="LM Studio API URL")
    parser.add_argument("--model", default=None,
                      help="Model name to test")
    parser.add_argument("--timeout", type=int, default=45,
                      help="Model generation timeout in seconds")
    parser.add_argument("--fix", action="store_true",
                      help="Attempt to fix issues automatically")
    parser.add_argument("--validate", action="store_true",
                      help="Run validation tests after fixes")
    parser.add_argument("--all-models", action="store_true",
                      help="Test all available models")
    parser.add_argument("--reset-fallback", action="store_true",
                      help="Reset fallback mode in adapter")
    parser.add_argument("--setup-tasks", action="store_true",
                      help="Set up task-specific models")

    args = parser.parse_args()

    print("LM Studio Connection Diagnostic Tool")
    print("===================================")

    # 1. Check if LM Studio service is available
    service_ok, message = check_lmstudio_service(args.url)
    print(message)

    # 2. Check available models
    if service_ok:
        models_ok, models_message, models_list = check_models_info(args.url)
        print(models_message)

        # Test specific model or first available
        if args.model:
            model_to_test = args.model
        elif models_list:
            model_to_test = models_list[0]["id"]
        else:
            model_to_test = "default"

        if models_ok:
            test_ok, test_message, _ = test_model_generation(args.url, model_to_test, args.timeout)
            print(test_message)

    # 3. Check if the adapter module is available
    print(f"\nLMStudioAdapter available: {lmstudio_available}")

    # 4. Apply fixes if requested
    if args.fix:
        print("\nApplying fixes...")
        adapter_fixed = fix_adapter_import()

    # 5. Reset fallback mode if requested
    if args.reset_fallback:
        print("\nAttempting to reset fallback mode...")
        try:
            adapter = LMStudioAdapter(fallback_mode=False)
            adapter.force_reconnect()
            print("✅ Created adapter with fallback mode disabled")

            # Save adapter settings to a file that the GUI can read
            settings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "MetaConsciousness", "lmstudio_settings.json")
            with open(settings_path, 'w') as f:
                json.dump({
                    "fallback_mode": False,
                    "model": model_to_test if model_to_test != "default" else None,
                    "last_reset": time.time()
                }, f)
            print(f"✅ Saved settings to {settings_path}")
        except Exception as e:
            print(f"❌ Error resetting fallback mode: {e}")
    
    # 6. Set up task-specific models if requested
    if args.setup_tasks:
        print("\nSetting up task-specific models...")
        setup_success, setup_message, task_model_mapping = setup_task_specific_models(args.url)
        print(setup_message)
        
        if setup_success:
            print("✅ Task-specific model configuration complete")
        else:
            print("⚠️ Task-specific model configuration completed with warnings")

    # 7. Run validation tests if requested
    if args.validate or args.fix:
        print("\nRunning validation tests...")
        adapter_result = test_adapter_details()

    # 8. Print summary and detailed instructions for fallback mode
    print("\nDiagnostic Summary:")
    print(f"- LM Studio Service: {'✅ Available' if service_ok else '❌ Unavailable'}")
    print(f"- Adapter Module: {'✅ Available' if lmstudio_available else '❌ Unavailable'}")

    if not service_ok:
        print("\nTo fix LM Studio connection issues:")
        print("1. Open LM Studio application")
        print("2. Load a model (preferably a smaller one like Phi-2 or Gemma-2B)")
        print("3. Click 'Start Server' in LM Studio's Chat tab")
        print("4. Verify the server is running on http://localhost:1234")
        print("5. Restart the MetaConsciousness GUI")
        print("6. Click 'Reconnect' in the LM Integration panel")
        print("\nIf still in fallback mode:")
        print("1. Toggle the Fallback Mode to Off")
        print("2. Click Reconnect again")
        print("3. Or run this tool with --reset-fallback flag")

    # Enhanced summary and tips section
    print("\nPerformance Tips:")
    print("1. For large models (32B+), use timeouts of 60-90 seconds")
    print("2. For medium models (7B-14B), use timeouts of 30-45 seconds")
    print("3. For small models (<7B), use timeouts of 10-20 seconds")
    print("4. Consider setting model through command line: --model llama-3.1-8b-instruct")
    print("5. Run with --setup-tasks to automatically configure task-specific models")
    print("6. Use debug tool before launching the application to ensure everything is set up")

    return 0

if __name__ == "__main__":
    sys.exit(main())
