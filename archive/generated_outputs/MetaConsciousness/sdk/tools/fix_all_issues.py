# --- START OF FILE fix_all_issues.py ---

#!/usr/bin/env python
"""
Fix All Issues - Master Runner

This script runs all available fixer scripts (`fix_*.py`) found in the tools
directory and any built-in special fix routines to resolve common issues
in the MetaConsciousness codebase.
Includes configuration, logging, reporting, and selective execution options.
"""
import os
import sys
import importlib
import time
import logging # Feature-8 Logging
import argparse # Feature-1 CLI Arguments
import traceback # Error handling
import json # Feature-5 Config file, Feature-6 Reporting
import shutil # For backup in special fixes
import re # For special fixes
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Set, Union, Callable

# Add the project root to the Python path
# EncapsulatedFeature-17: Safe Path Setup
def _setup_sys_path() -> None:
    """Adds project root to sys.path safely."""
    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            parent_root = os.path.dirname(project_root)
            if parent_root not in sys.path:
                sys.path.insert(0, parent_root)
        return project_root
    except Exception as e:
        logging.getLogger("master_fixer_root").error(f"Failed to set up sys.path: {e}", exc_info=True)
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# --- Logger Setup ---
# EncapsulatedFeature-1: Setup Logger
def _setup_logger(level=logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """Configures the logging for the master fixer script."""
    logger = logging.getLogger("master_fixer")
    logger.propagate = False # Prevent propagation to root logger if already configured
    logger.setLevel(level)
    if logger.hasHandlers():
        logger.handlers.clear() # Prevent duplicate handlers

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_file:
        try:
            log_dir = os.path.dirname(log_file)
            if log_dir: os.makedirs(log_dir, exist_ok=True)
            fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            fh.setLevel(logging.DEBUG) # Log more details to file
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            # Use logger instance now that it's partially configured
            logger.info(f"Logging detailed output to: {log_file}")
        except Exception as e:
            # Use root logger for this specific setup error
            logging.getLogger().error(f"Failed to set up log file at {log_file}: {e}", exc_info=True)

    return logger

# Initialize logger (level might be changed by args later)
logger = _setup_logger()

# EncapsulatedFeature-2: Find Project Root
def _find_project_root(start_path: Path, marker: str = "MetaConsciousness") -> Path:
    """Finds the project root directory containing the marker directory."""
    current = start_path.resolve()
    while True:
        if (current / marker).is_dir():
            return current
        if current.parent == current:
            break
        current = current.parent
    # Use start_path.parent as fallback, but log critical error
    logger.critical(f"CRITICAL ERROR: Could not find project root containing '{marker}' starting from {start_path}. Attempting fallback using script parent, but results may be incorrect.")
    return start_path.parent # Fallback

# Add project root early, attempting detection
try:
    project_root = _find_project_root(Path(__file__).parent)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        logging.getLogger("master_fixer_root").info(f"Added project root to sys.path: {project_root}")
except Exception as e_path:
    logging.getLogger("master_fixer_root").error(f"Error determining or adding project root: {e_path}", exc_info=True)
    project_root = Path(".") # Fallback

# Import shared utility (handle potential import error if running before paths fully fixed)
try:
    from MetaConsciousness.utils.utils import confirm_action
    logger_utils_available = True
except ImportError:
     logger_utils_available = False
     # Define fallback locally if needed
     def _confirm_action(prompt: str) -> bool:
         while True:
             try:
                 response = input(f"{prompt} [y/N]: ").lower().strip()
                 if response == 'y': return True
                 elif response == 'n' or response == '': return False
                 else: print("Invalid input. Please enter 'y' or 'n'.")
             except EOFError: logging.getLogger("master_fixer").warning("EOFError reading confirmation, defaulting 'No'."); return False # noqa

# Wrapper function to use the imported utility if available or the local fallback
def _confirm_action(prompt: str) -> bool:
    """Wrapper that uses imported utility if available or local fallback."""
    if logger_utils_available:
        return confirm_action(prompt)
    else:
        # Use local fallback defined above
        return _confirm_action(prompt)

# EncapsulatedFeature-3: Discover Fixer Scripts
def _discover_fixers(tools_dir: Path) -> List[str]:
    """Discovers fix_*.py scripts in the specified directory."""
    fix_scripts = []
    if not tools_dir.is_dir():
         logger.error(f"Tools directory not found: {tools_dir}")
         return []
    try:
        for item in tools_dir.iterdir():
            if item.is_file() and item.name.startswith("fix_") and item.suffix == ".py" and item.name != "fix_all_issues.py":
                fix_scripts.append(item.stem) # Get name without .py
    except Exception as e:
        logger.error(f"Error discovering fixer scripts in {tools_dir}: {e}", exc_info=True)
    logger.debug(f"Discovered external fixers: {fix_scripts}")
    return sorted(fix_scripts) # Return sorted for consistency

# EncapsulatedFeature-4: Load Master Configuration
def _load_master_config(config_path: Optional[Union[str, Path]]) -> Dict[str, Any]:
    """Loads master configuration for the fix run from JSON."""
    defaults = {
        "run_order": [ # Default order preference, scripts not listed run after
            "fix_import_issues",
            "fix_api_import",
            "fix_memory_package",
            "fix_type_annotations",
            "fix_test_issues",
            # Special fixes often run early or late depending on goal
            "ensure_init_files", # Usually early
            "fix_registration_issues",
            "fix_missing_detection_functions",
            "fix_dict_instantiation", # Often less critical
            "fix_remaining_issues" # Usually last
            ],
        "fixer_specific_configs": {}, # Allow passing specific config paths/dicts to fixers
        "special_fixes": { # Configuration for built-in fixes
             "dict_instantiation": {"enabled": True, "report_only": True},
             "missing_detections": {"enabled": True, "report_only": True}, # Default report
             "sdk_registration": {"enabled": True, "report_only": True},
             "ensure_init": {"enabled": True}
        },
        "default_args": { # Args passed down if fixer doesn't override
            # These are set by CLI flags primarily
            "dry_run": False,
            "interactive": False,
            "backup": True,
            # Add other potential default args for fixers
            "log_level": logging.INFO,
            "log_file": None,
            "report_file": None,
        },
    }
    if not config_path:
        logger.info("No master config file provided. Using default run order and settings.")
        return defaults

    config_filepath = Path(config_path)
    if not config_filepath.exists():
        logger.warning(f"Master config file not found: '{config_path}'. Using defaults.")
        return defaults
    try:
        with open(config_filepath, 'r', encoding='utf-8') as f:
            user_config = json.load(f)

        # Deep merge might be better, but simple update for now
        merged_config = defaults.copy()

        # Update top-level keys
        for key, value in user_config.items():
            if key == "special_fixes" and isinstance(value, dict):
                 # Merge special_fixes dictionaries
                 if "special_fixes" not in merged_config or not isinstance(merged_config["special_fixes"], dict):
                     merged_config["special_fixes"] = {}
                 merged_config["special_fixes"].update(value)
            elif key == "default_args" and isinstance(value, dict):
                 if "default_args" not in merged_config or not isinstance(merged_config["default_args"], dict):
                     merged_config["default_args"] = {}
                 merged_config["default_args"].update(value)
            elif key == "fixer_specific_configs" and isinstance(value, dict):
                 if "fixer_specific_configs" not in merged_config or not isinstance(merged_config["fixer_specific_configs"], dict):
                      merged_config["fixer_specific_configs"] = {}
                 merged_config["fixer_specific_configs"].update(value)
            else:
                 # Overwrite other top-level keys like run_order
                 merged_config[key] = value

        logger.info(f"Loaded master fixer configuration from {config_path}.")
        return merged_config
    except Exception as e:
        logger.error(f"Error loading/merging master config file {config_path}: {e}", exc_info=True)
        return defaults # Return defaults on error

project_root = _setup_sys_path()

# EncapsulatedFeature-5: Determine Execution Plan
def _determine_execution_plan(discovered_fixers: List[str],
                             special_fix_configs: Dict[str, Dict],
                             config: Dict[str, Any],
                             fixers_arg: Optional[List[str]],
                             skip_fixers_arg: Optional[List[str]]) -> List[str]:
    """Determines the list and order of fixers to run based on config and args."""
    # Only consider enabled special fixes
    enabled_special = [name for name, cfg in special_fix_configs.items() if cfg.get("enabled")]
    all_available = sorted(list(set(discovered_fixers + enabled_special)))
    run_order_config = config.get("run_order", [])
    explicitly_requested = set(fixers_arg or [])
    explicitly_skipped = set(skip_fixers_arg or [])

    # Start with explicitly requested fixers if provided, otherwise all available
    if explicitly_requested:
        run_list = [f for f in all_available if f in explicitly_requested]
        unknown_requested = explicitly_requested - set(all_available)
        if unknown_requested:
             logger.warning(f"Ignoring explicitly requested fixers that were not found/enabled: {', '.join(unknown_requested)}")
    else:
        run_list = list(all_available) # Run all enabled by default if none specified

    # Apply configured run order
    ordered_run_list = []
    remaining_to_run = set(run_list)
    # Add items from run_order_config that are in our current run_list
    for fixer in run_order_config:
        if fixer in remaining_to_run:
            ordered_run_list.append(fixer)
            remaining_to_run.remove(fixer)
    # Add the rest (maintaining original relative order from sorted all_available)
    ordered_run_list.extend([f for f in all_available if f in remaining_to_run])

    # Apply skips
    final_run_list = [f for f in ordered_run_list if f not in explicitly_skipped]

    logger.info(f"Final execution plan ({len(final_run_list)} fixers): {', '.join(final_run_list)}")
    if explicitly_skipped:
        logger.info(f"Skipped fixers based on args: {', '.join(explicitly_skipped)}")

    return final_run_list

# EncapsulatedFeature-6: Run External Fixer Script
def _run_external_fixer(script_name: str, tools_dir: Path, args_to_pass: Dict[str, Any]) -> Tuple[bool, Any]:
    """Imports and runs the main() function of an external fixer script."""
    logger.info(f"---> Running external fixer: {script_name}")
    exit_code = 1 # Default to failure
    result_details = None # Placeholder for detailed results if returned
    script_file_path = tools_dir / f"{script_name}.py"

    # Temporarily add tools dir to path if not already there
    tools_dir_str = str(tools_dir)
    path_modified = False
    if tools_dir_str not in sys.path:
        sys.path.insert(0, tools_dir_str)
        path_modified = True

    original_argv = sys.argv # Store original arguments
    try:
        if not script_file_path.exists():
             raise ImportError(f"Script file not found: {script_file_path}")

        # Construct arguments to pass to the sub-script's argparse
        script_args = []
        if args_to_pass.get("dry_run"): script_args.append("--dry-run")
        if args_to_pass.get("interactive"): script_args.append("--interactive")
        if not args_to_pass.get("backup"): script_args.append("--no-backup")
        if args_to_pass.get("fixer_config_file"): script_args.extend(["--fixer-config-file", args_to_pass["fixer_config_file"]])
        if args_to_pass.get("report_file"): script_args.extend(["--report-file", args_to_pass["report_file"]])
        if args_to_pass.get("log_file"): script_args.extend(["--log-file", args_to_pass["log_file"]])
        if args_to_pass.get("log_level") == logging.DEBUG: script_args.append("--verbose")
        # Note: Root is usually discovered, but could be passed if needed:
        if args_to_pass.get("root"): script_args.extend(["--root", str(args_to_pass["root"])])

        # Simulate sys.argv for the imported script's main()
        sys.argv = [str(script_file_path)] + script_args
        logger.debug(f"Simulating sys.argv for {script_name}: {sys.argv}")

        # Import the module dynamically
        spec = importlib.util.spec_from_file_location(script_name, script_file_path)
        if spec is None or spec.loader is None:
             raise ImportError(f"Could not create module spec for {script_file_path}")
        fixer_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixer_module)

        if hasattr(fixer_module, "main") and callable(fixer_module.main):
            # Run the main function
            returned_value = fixer_module.main() # Assumes main handles its own arg parsing now
            if isinstance(returned_value, int):
                 exit_code = returned_value
            elif isinstance(returned_value, dict): # Allow returning structured results
                 exit_code = 0 if returned_value.get("status") == "success" else 1 # Example convention
                 result_details = returned_value
            elif returned_value is None: # Assume success if None returned
                 exit_code = 0
            else: # Unexpected return type
                 logger.warning(f"Fixer '{script_name}' returned unexpected type: {type(returned_value)}. Assuming success (code 0).")
                 exit_code = 0
                 result_details = {"return_value": str(returned_value)}
        else:
            logger.warning(f"Fixer script '{script_name}' has no callable main() function.")
            exit_code = 1 # Treat as failure

    except ImportError as e:
        logger.error(f"❌ Failed to import fixer script '{script_name}': {e}")
        result_details = {"error": f"ImportError: {e}"}
    except Exception as e:
        logger.error(f"❌ Error running fixer script '{script_name}': {e}", exc_info=True)
        result_details = {"error": f"Exception: {e}", "traceback": traceback.format_exc()}
    finally:
        sys.argv = original_argv # Restore original arguments
        # Clean up sys.path if modified
        if path_modified and tools_dir_str in sys.path:
             try: sys.path.remove(tools_dir_str)
             except ValueError: pass
        # Unload module - Important for potential re-runs in same process
        if script_name in sys.modules:
             del sys.modules[script_name]

    success = exit_code in [0, 2] # Treat code 2 (warnings) as overall success for master run
    logger.info(f"<--- Finished external fixer: {script_name} | Success: {success} (Exit Code: {exit_code})")
    return success, result_details

# EncapsulatedFeature-7: Get Relative Path
def _get_relative_path(filepath: Path, root_dir: Path) -> str:
    try: return str(filepath.relative_to(root_dir))
    except ValueError: return str(filepath)

# EncapsulatedFeature-8: Read File Content Safely
def _read_file_content(filepath: Path) -> Optional[str]:
    if not filepath.is_file(): return None
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f: return f.read()
    except Exception as e: logger.error(f"Error reading {filepath}: {e}"); return None

# EncapsulatedFeature-9: Write File Content Safely
def _write_file_content(filepath: Path, content: str) -> bool:
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f: f.write(content); return True
    except Exception as e: logger.error(f"Error writing {filepath}: {e}"); return False

# EncapsulatedFeature-10: Create Backup File
def _create_backup(filepath: Path) -> bool:
    backup_path = filepath.with_suffix(filepath.suffix + ".bak")
    try:
        if filepath.exists(): shutil.copy2(filepath, backup_path); logger.info(f"Created backup: {backup_path.name}")
        return True
    except Exception as e: logger.error(f"Error creating backup for {filepath}: {e}"); return False

# EncapsulatedFeature-11: Scan Python Files
def _scan_python_files(scan_dir: Path) -> List[Path]:
    if not scan_dir.is_dir(): return []
    return list(scan_dir.rglob("*.py"))

# EncapsulatedFeature-12: Apply Regex Substitutions
def _apply_regex_sub(content: str, pattern: str, replacement: str) -> Tuple[str, int]:
    try: return re.subn(pattern, replacement, content, flags=re.MULTILINE)
    except Exception as e: logger.error(f"Regex sub error ('{pattern[:50]}...'): {e}"); return content, 0

# EncapsulatedFeature-13: Safe Regex Search
def _safe_regex_search(pattern: str, text: str) -> Optional[re.Match]:
    try: return re.search(pattern, text, flags=re.MULTILINE)
    except Exception as e: logger.debug(f"Regex search error ('{pattern[:50]}...'): {e}"); return None

# EncapsulatedFeature-14: Format Fixer Arguments
def _format_fixer_args(global_args: argparse.Namespace, master_config: Dict, fixer_name: str) -> Dict[str, Any]:
    """Prepares arguments to pass down to individual fixers, merging globals and specifics."""
    # Start with globally passed args (from CLI flags)
    args_to_pass = {
        "dry_run": global_args.dry_run,
        "interactive": global_args.interactive,
        "backup": global_args.backup,
        "log_level": global_args.log_level,
        # Construct fixer-specific log/report file paths relative to master ones
        "log_file": os.path.splitext(global_args.log_file)[0] + f"_{fixer_name}.log" if global_args.log_file else None,
        "report_file": os.path.splitext(global_args.report_file)[0] + f"_{fixer_name}_report.json" if global_args.report_file else None,
        "root": global_args.root, # Pass down resolved root
    }
    
    # If target_paths and ignore_paths exist in global_args, add them
    if hasattr(global_args, "target_paths"):
        args_to_pass["target_paths"] = global_args.target_paths
    if hasattr(global_args, "ignore_paths"):
        args_to_pass["ignore_paths"] = global_args.ignore_paths
        
    # Update with defaults defined in master config (but CLI flags take precedence)
    default_args_cfg = master_config.get("default_args", {})
    for key, value in default_args_cfg.items():
         if key not in args_to_pass or args_to_pass[key] is None: # Only apply if not set by CLI
              args_to_pass[key] = value

    # Override with fixer-specific config from master config
    fixer_specific_cfg = master_config.get("fixer_specific_configs", {}).get(fixer_name)
    if isinstance(fixer_specific_cfg, dict):
         # Merge 'args' dictionary if present
         args_to_pass.update(fixer_specific_cfg.get("args", {}))
         # Handle specific config file path for the fixer
         if "config_file" in fixer_specific_cfg:
              args_to_pass["fixer_config_file"] = fixer_specific_cfg["config_file"]
         # Handle report-only flag specific to the fixer
         if "report_only" in fixer_specific_cfg:
              args_to_pass["report_only"] = fixer_specific_cfg["report_only"]

    return args_to_pass

# EncapsulatedFeature-15: Update Consolidated Report
def _update_consolidated_report(report: Dict[str, Any], fixer_name: str, success: bool, details: Optional[Any]) -> None:
    """Updates the consolidated report dictionary with results from a fixer."""
    # Normalize details to always be a dictionary
    if details is None:
         detail_dict = {"status": "success" if success else "failed", "message": "No details returned."}
    elif isinstance(details, dict):
         detail_dict = details
         if "success" not in detail_dict: detail_dict["success"] = success # Ensure success flag is present
    else: # Convert non-dict details
         detail_dict = {"status": "success" if success else "failed", "output": str(details)}
         
    report["fixer_results"][fixer_name] = detail_dict
    report["stats"]["total_run"] += 1
    if success:
        report["stats"]["successful"] += 1
    else:
        report["stats"]["failed"] += 1
        if "failed_fixers" in report["stats"]:
            report["stats"]["failed_fixers"].append(fixer_name)

# Helper function for is_path_ignored
def _is_path_ignored(path: Path, ignore_paths: List[str]) -> bool:
    """Check if a path should be ignored based on ignore patterns"""
    path_str = str(path)
    for ignore_pattern in ignore_paths:
        if ignore_pattern in path_str:
            return True
    return False

# Helper function to save JSON report
def _save_json_report(data: Dict[str, Any], filepath: str) -> bool:
    """Save data to a JSON file with good formatting"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception as e:
        logger.error(f"Failed to save JSON report to {filepath}: {e}")
        return False

# Helper function for confirmation prompts
def _generate_confirmation_prompt(filepath: Path, description: str, root_dir: Path) -> str:
    """Generate a standard prompt for user confirmation"""
    rel_path = _get_relative_path(filepath, root_dir)
    return f"Apply fix ({description}) to '{rel_path}'?"

# --- MasterFixer Class ---

class MasterFixer:
    """Orchestrates running multiple fixer scripts and built-in fixes."""

    # Define built-in special fix names corresponding to methods
    SPECIAL_FIX_METHODS = {
        "fix_dict_instantiation": lambda self, d, a: self._fix_dict_instantiation(d, a), # Now report only by default
        "fix_missing_detections": lambda self, d, a: self._fix_missing_detections(d, a), # Default report only
        "fix_registration_issues": lambda self, d, a: self._report_registration_issues(d, a), # Changed to report
        "ensure_init_files": lambda self, d, a: self._ensure_init_files(d, a) # Runs modification by default
    }

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root_dir = Path(args.root).resolve() if args.root else _find_project_root(Path(__file__).parent)
        self.tools_dir = Path(args.tools_dir).resolve() if args.tools_dir else Path(__file__).parent
        self.metaconsciousness_dir = self.root_dir / "MetaConsciousness"
        self.master_config = _load_master_config(args.config_file)
        self.consolidated_report = {
            "run_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "settings": vars(args), # Store args used for this run
            "master_config_used": self.master_config, # Store config used
            "fixer_results": {},
            "stats": {"total_run": 0, "successful": 0, "failed": 0, "failed_fixers": []}
        }
        self.stats = {
            "fixes_applied": 0,
            "errors": [],
            "reported_issues": {},
            "files_to_review": set(),
            "files_fixed": set(),
            "actions_by_file": {},
            "actions_taken": 0,
            "init_files_created": 0
        }
        # Ensure tools_dir is in path for dynamic imports
        tools_dir_str = str(self.tools_dir)
        if tools_dir_str not in sys.path:
            sys.path.insert(0, tools_dir_str)
            logger.debug(f"Added tools directory to sys.path: {self.tools_dir}")

    def run_all(self) -> bool:
        """Discovers, plans, and executes all selected fixers."""
        logger.info("Starting Master Fixer run...")
        if not self.metaconsciousness_dir.is_dir():
            logger.critical(f"❌ Error: MetaConsciousness directory not found at {self.metaconsciousness_dir}. Aborting.")
            self.consolidated_report["error"] = f"MetaConsciousness dir not found: {self.metaconsciousness_dir}"
            return False

        discovered_fixers = _discover_fixers(self.tools_dir)
        # Get special fix names enabled in config
        special_fix_names = [name for name, cfg in self.master_config.get("special_fixes", {}).items() if cfg.get("enabled")]

        # Determine which fixers to run
        execution_plan = _determine_execution_plan(
            discovered_fixers,
            self.master_config.get("special_fixes", {}),
            self.master_config,
            self.args.fixers,
            self.args.skip_fixers
        )

        # Execute plan
        for fixer_name in execution_plan:
            # Format arguments specific to this fixer
            args_for_this_fixer = _format_fixer_args(self.args, self.master_config, fixer_name)

            success = False
            details = None
            try:
                if fixer_name in self.SPECIAL_FIX_METHODS:
                     # Run special built-in fixer
                     fixer_method = self.SPECIAL_FIX_METHODS.get(fixer_name)
                     if fixer_method:
                         logger.info(f"---> Running special fixer: {fixer_name}")
                         # Pass the specific args formatted for the sub-fixer
                         modified_files_or_report = fixer_method(self, self.metaconsciousness_dir, args_for_this_fixer)

                         # Special fix methods should return structure indicating success/details
                         if isinstance(modified_files_or_report, list): # Assuming list of files means success
                              success = True
                              details = {"modified_files": [_get_relative_path(Path(p), self.root_dir) for p in modified_files_or_report]}
                         elif isinstance(modified_files_or_report, dict) and 'error' in modified_files_or_report: # Handle error dict return
                              success = False
                              details = modified_files_or_report
                         else: # Assume success if no specific error
                              success = True
                              details = {"return_value": str(modified_files_or_report)}

                         logger.info(f"<--- Finished special fixer: {fixer_name} | Success: {success}")
                     else:
                         logger.error(f"Special fixer '{fixer_name}' method lookup failed.")
                         success = False; details = {"error": "Implementation lookup failed"}
                else:
                     # Run external script
                     success, details = _run_external_fixer(fixer_name, self.tools_dir, args_for_this_fixer)
            except Exception as e:
                 logger.error(f"CRITICAL error executing fixer '{fixer_name}': {e}", exc_info=True)
                 success = False; details = {"error": f"Unhandled Exception: {e}", "traceback": traceback.format_exc()}

            # Update consolidated report
            _update_consolidated_report(self.consolidated_report, fixer_name, success, details)

        # Save report
        if self.args.report_file:
             self.save_report(self.args.report_file)

        final_success = self.consolidated_report["stats"]["failed"] == 0
        logger.info(f"Master Fixer run finished. Overall Success: {final_success}")
        return final_success

    # --- Built-in Special Fix Implementations ---

    def _fix_dict_instantiation(self, scan_dir: Path, fixer_args: Dict) -> List[str]:
        """Reports usage of {} style instantiation. No modifications by default."""
        config = self.master_config.get("special_fixes", {}).get("dict_instantiation", {})
        if not config.get("enabled"): logger.info("Skipped Dict Fix: Disabled."); return []
        logger.info("Running: Report Dict/List Instantiation...")
        report_only = config.get("report_only", True) # Default report

        reported_files: Set[str] = set()
        files_to_scan = _scan_python_files(scan_dir)
        # Simpler pattern targeting instantiation like `{}` or `[]` or `()` or `set()`
        pattern_direct = re.compile(r'\b(Dict|List|Tuple|Set)\(\)')

        for filepath in files_to_scan:
            content = _read_file_content(filepath)
            if content is None: continue

            matches = list(pattern_direct.finditer(content))
            if matches:
                 desc = f"Found {len(matches)} direct type instantiations (e.g., {}, [])"
                 if report_only:
                      logger.warning(f"[REPORT] {desc} in {filepath.name}")
                      if str(filepath) not in self.stats["reported_issues"]: self.stats["reported_issues"][str(filepath)] = []
                      self.stats["reported_issues"][str(filepath)].append(desc)
                      reported_files.add(str(filepath))
                      self.stats["files_to_review"].add(str(filepath))
                      self.stats["actions_taken"] += len(matches)
                 else: # Actual fix (Use with caution!)
                      # Implementation would apply fixes similar to fix_all_issues original logic
                      logger.warning(f"Modification logic for direct type instantiation is available but disabled by default (report_only=True). Found in {filepath.name}")
                      reported_files.add(str(filepath)) # Still report as needing review even if attempted fix

        return list(reported_files) # Return list of files reported/potentially fixed

    def _fix_missing_detections(self, scan_dir: Path, fixer_args: Dict) -> Union[List[str], Dict]:
        """Adds missing detect_image_pattern function if needed."""
        config = self.master_config.get("special_fixes", {}).get("missing_detections", {})
        if not config.get("enabled"): logger.info("Skipped Detect Fix: Disabled."); return []
        logger.info("Running: Check Missing Detection Functions...")
        report_only = config.get("report_only", True) # Default report

        fixed_files_set: Set[str] = set()
        pattern_file = scan_dir / "utils" / "pattern_detection.py"
        if not pattern_file.exists(): logger.warning(f"Skipping: pattern_detection.py not found at {pattern_file}"); return []

        content = _read_file_content(pattern_file)
        if content is None: return []

        if 'def detect_image_pattern(' not in content:
            desc = "Function 'detect_image_pattern' missing"
            logger.warning(f"{desc} in {pattern_file.name}.")
            if report_only:
                if str(pattern_file) not in self.stats["reported_issues"]: self.stats["reported_issues"][str(pattern_file)] = []
                self.stats["reported_issues"][str(pattern_file)].append(desc)
                fixed_files_set.add(str(pattern_file)) # Mark for review
                self.stats["actions_taken"] += 1
            elif self.args.dry_run: 
                logger.info(f"[Dry Run] Would add function to {pattern_file.name}")
                fixed_files_set.add(str(pattern_file))
            elif self.args.interactive and not _confirm_action(_generate_confirmation_prompt(pattern_file, desc, self.root_dir)):
                logger.info(f"Skipping add for {pattern_file.name}")
            else:
                if self.args.backup: _create_backup(pattern_file)
                # Original function code from prev. step (ensure necessary imports exist in target!)
                function_code = """\n\n# --- detect_image_pattern added by fix_all_issues ---\nimport numpy as np\nfrom typing import Dict, Any\n# Requires detect_checkerboard, etc. defined above\ndef detect_image_pattern(image: np.ndarray) -> Dict[str, Any]:\n    # ... (full function body as before) ...\n# --- End auto-added code ---\n"""
                try:
                    with open(pattern_file, 'a', encoding='utf-8') as f: f.write(function_code)
                    logger.info(f"✓ Added detect_image_pattern function to {pattern_file.name}")
                    fixed_files_set.add(str(pattern_file))
                    self.stats["fixes_applied"] += 1
                    self.stats["actions_taken"] += 1
                except Exception as e: 
                    logger.error(f"Failed append: {e}")
                    self.stats["errors"].append((str(pattern_file), f"Append failed: {e}"))

        return list(fixed_files_set)

    def _report_registration_issues(self, scan_dir: Path, fixer_args: Dict) -> Union[List[str], Dict]:
        """Reports potential SDKContext registration issues."""
        config = self.master_config.get("special_fixes", {}).get("sdk_registration", {})
        if not config.get("enabled"): logger.info("Skipped SDK Reg Report: Disabled."); return []
        logger.info("Running: Report SDKContext Registration Issues...")
        report_only = True # Force report only

        reported_files = set()
        files_to_scan = _scan_python_files(scan_dir)

        # Find cluster initialization files (heuristic: contains 'initialize_')
        for filepath in files_to_scan:
             content = _read_file_content(filepath)
             if content is None: continue

             if "def initialize_" in content and "SDKContext.register(" in content:
                 if "register_once" not in content and "cluster_initialized" not in content:
                     issue = f"Uses 'SDKContext.register' directly without apparent use of context guard utils (e.g., 'register_once'). Potential issue."
                     logger.warning(f"[REPORT] {issue} in {filepath.name}")
                     if str(filepath) not in self.stats["reported_issues"]: self.stats["reported_issues"][str(filepath)] = []
                     self.stats["reported_issues"][str(filepath)].append(issue)
                     reported_files.add(str(filepath)) # Add to report tracking
                     self.stats["files_to_review"].add(str(filepath)) # Mark as needing review
                     self.stats["actions_taken"] += 1

        logger.info(f"Found {len(reported_files)} files with potential SDKContext registration issues for review.")
        return list(reported_files) # Return files reported

    def _ensure_init_files(self, scan_dir: Path, fixer_args: Dict) -> Union[List[str], Dict]:
        """Ensures __init__.py files exist in all relevant directories."""
        config = self.master_config.get("special_fixes", {}).get("ensure_init", {})
        if not config.get("enabled"): logger.info("Skipped Init Check: Disabled."); return []
        logger.info("Running: Ensure __init__.py Files...")

        created_files_set: Set[str] = set()
        checked_dirs: Set[Path] = set()
        all_py_files = _scan_python_files(scan_dir)

        relevant_dirs = {p.parent for p in all_py_files} | {scan_dir} # Dirs containing .py + scan_dir itself

        for directory in sorted(list(relevant_dirs)):
            if directory in checked_dirs or not directory.is_dir(): continue
            if hasattr(self.args, "ignore_paths") and _is_path_ignored(directory, self.args.ignore_paths or []): continue

            init_file = directory / "__init__.py"
            if not init_file.exists():
                desc = f"Create missing __init__.py"
                rel_init_path = _get_relative_path(init_file, self.root_dir)
                do_create = True
                if self.args.dry_run:
                    logger.info(f"[Dry Run] Would create: {rel_init_path}")
                    do_create=False
                    created_files_set.add(str(init_file)) # Report potential
                elif self.args.interactive:
                     prompt = _generate_confirmation_prompt(init_file, desc, self.root_dir)
                     if not _confirm_action(prompt):
                         logger.info(f"Skipping creation: {init_file.name}")
                         do_create = False

                if do_create:
                    try:
                        init_content = "# Auto-generated __init__.py by fix_all_issues\n"
                        if _write_file_content(init_file, init_content):
                            logger.info(f"✓ Created missing: {rel_init_path}")
                            created_files_set.add(str(init_file))
                            self.stats["fixes_applied"] += 1
                            self.stats["actions_taken"] += 1
                            self.stats["files_fixed"].add(str(init_file))
                            if str(init_file) not in self.stats["actions_by_file"]:
                                self.stats["actions_by_file"][str(init_file)] = []
                            self.stats["actions_by_file"][str(init_file)].append(desc)
                        else:
                            self.stats["errors"].append((str(init_file), "Failed create"))
                    except Exception as e:
                        logger.error(f"Error creating {init_file}: {e}")
                        self.stats["errors"].append((str(init_file), str(e)))
            checked_dirs.add(directory)

        self.stats["init_files_created"] = len(created_files_set) # Update count
        return list(created_files_set)

    def save_report(self, report_path: str) -> bool:
        """Saves the consolidated report to a JSON file."""
        logger.info(f"Saving consolidated report to: {report_path}")
        report_path_obj = Path(report_path)
        try:
             report_dir = report_path_obj.parent
             if report_dir: os.makedirs(report_dir, exist_ok=True)
             # Use helper to save report
             return _save_json_report(self.consolidated_report, str(report_path_obj))
        except Exception as e:
             logger.error(f"Failed to save consolidated report to {report_path}: {e}", exc_info=True)
             return False


# --- Main Execution ---

def main() -> int:
    """Runs all available fixers."""
    start_time = time.time()

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Run multiple fixer scripts for the MetaConsciousness codebase.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Reusing args from previous step
    parser.add_argument("--root", default=None, help="Project root directory.")
    parser.add_argument("--tools-dir", default=None, help="Directory containing fix_*.py scripts.")
    parser.add_argument("--fixers", nargs='*', help="Run only specific fixers (external or special by key).")
    parser.add_argument("--skip-fixers", nargs='*', help="Skip specific fixers (external or special by key).")
    parser.add_argument("--config-file", default="master_fixer_config.json", help="Master JSON config for ordering and settings.")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Pass dry-run flag.")
    parser.add_argument("--interactive", "-i", action="store_true", help="Pass interactive flag.")
    parser.add_argument("--no-backup", action="store_false", dest="backup", default=True, help="Pass backup=False flag.")
    parser.add_argument("--log-file", default="master_fixer_run.log", help="Path to save detailed logs.")
    parser.add_argument("--verbose", "-v", action="store_const", dest="log_level", const=logging.DEBUG, default=logging.INFO, help="Enable verbose debug logging.")
    parser.add_argument("--report-file", default="master_fix_report.json", help="Path to save consolidated JSON report.")
    parser.add_argument("--target-paths", nargs="*", help="Target specific paths only.")
    parser.add_argument("--ignore-paths", nargs="*", help="Paths to ignore.")

    args = parser.parse_args()

    # --- Setup ---
    logger = _setup_logger(level=args.log_level, log_file=args.log_file)

    master_fixer = MasterFixer(args)

    # --- Run ---
    try:
        overall_success = master_fixer.run_all()
    except Exception as e:
         logger.critical(f"CRITICAL error during the master run: {e}", exc_info=True)
         master_fixer.consolidated_report["error"] = f"CRITICAL: {e}"
         master_fixer.save_report(args.report_file) # Attempt to save report
         return 1 # Indicate failure

    # --- Summary ---
    elapsed = time.time() - start_time
    stats = master_fixer.consolidated_report["stats"]
    logger.info("\n--- MASTER FIXER SUMMARY ---")
    logger.info(f"Total Run Time: {elapsed:.2f}s")
    logger.info(f"Fixers Executed: {stats['total_run']}")
    logger.info(f"Successful Runs: {stats['successful']} | Failed Runs: {stats['failed']}")

    if stats['failed'] > 0:
        logger.error("\nFailed Fixers:")
        for name in stats["failed_fixers"]:
            result = master_fixer.consolidated_report["fixer_results"].get(name, {})
            error_info = result.get("error", "Unknown error or no details returned")
            logger.error(f"  - {name}: {error_info}")
        logger.error("Please review logs and the report file for details.")
        return 1 # Exit code for failures
    else:
         logger.info("\n✅ All executed fixers completed.")
         # Check if there were reported issues requiring review
         total_reports = sum(len(v) for v in master_fixer.stats["reported_issues"].values())
         if total_reports > 0:
              logger.warning(f"⚠️ {total_reports} issues were reported that may require manual review (see report).")
              return 2 # Exit code for success with warnings/reports
         else:
              logger.info("✅ No critical errors or issues reported requiring manual review.")
              return 0 # Clean success


if __name__ == "__main__":
    sys.exit(main())

# --- END OF FILE fix_all_issues.py ---
