"""
Emergency Reset Utility for LM Studio Connection

This standalone script provides a way to reset the LM Studio connection settings
when the GUI gets stuck in fallback mode.
"""
import os
import sys
import json
import time
import logging
import argparse
import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import platform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def reset_lmstudio_config() -> None:
    """
    Reset the LM Studio connection configuration.

    Returns:
        bool: Success status
    """
    try:
        # Ensure config directory exists
        config_dir = os.path.join(os.path.expanduser("~"), ".metaconsciousness", "config")
        os.makedirs(config_dir, exist_ok=True)

        # Create or update the config file
        config_file = os.path.join(config_dir, "lmstudio_state.json")

        # Create reset config
        reset_config = {
            "fallback_mode": False,
            "reset_timestamp": time.time(),
            "model": None,
            "forced_reset": True
        }

        # Write to file
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(reset_config, f, indent=2)

        logger.info(f"Created reset config at {config_file}")
        return True
    except Exception as e:
        logger.error(f"Error resetting LM Studio config: {e}")
        return False

def check_lmstudio_process() -> None:
    """
    Check if LM Studio process is running.

    Returns:
        bool: True if LM Studio is running
    """
    try:
        if platform.system() == "Windows":
            output = subprocess.check_output("tasklist", shell=True).decode("utf-8", errors="ignore")
            return "LMStudio.exe" in output
        elif platform.system() == "Darwin":  # macOS
            output = subprocess.check_output(["ps", "-A"]).decode("utf-8", errors="ignore")
            return "LMStudio" in output
        else:  # Linux
            output = subprocess.check_output(["ps", "-A"]).decode("utf-8", errors="ignore")
            return "LMStudio" in output
    except Exception as e:
        logger.warning(f"Error checking LM Studio process: {e}")
        return False

def launch_lmstudio() -> None:
    """
    Attempt to launch LM Studio application.

    Returns:
        bool: Success status
    """
    try:
        if platform.system() == "Windows":
            # Common installation locations
            possible_paths = [
                os.path.join(os.environ.get('PROGRAMFILES', 'C:\\Program Files'), "LMStudio", "LMStudio.exe"),
                os.path.join(os.environ.get('PROGRAMFILES(X86)', 'C:\\Program Files (x86)'), "LMStudio", "LMStudio.exe"),
                os.path.join(os.environ.get('LOCALAPPDATA', 'C:\\Users\\Default\\AppData\\Local'), "Programs", "LMStudio", "LMStudio.exe")
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    subprocess.Popen([path])
                    logger.info(f"Launched LM Studio from {path}")
                    return True

            logger.warning("Could not find LM Studio executable")
            return False
        elif platform.system() == "Darwin":  # macOS
            subprocess.Popen(["open", "-a", "LMStudio"])
            return True
        else:  # Linux
            subprocess.Popen(["LMStudio"])
            return True
    except Exception as e:
        logger.error(f"Error launching LM Studio: {e}")
        return False

def create_gui() -> None:
    """Create a simple GUI for the reset utility."""
    root = tk.Tk()
    root.title("LM Studio Connection Reset")
    root.geometry("500x400")
    root.resizable(True, True)

    # Set up styles
    style = ttk.Style()
    style.configure("TButton", font=("Arial", 10))
    style.configure("Reset.TButton", font=("Arial", 10, "bold"), foreground="red")
    style.configure("Launch.TButton", font=("Arial", 10, "bold"), foreground="green")

    # Main frame
    main_frame = ttk.Frame(root, padding=20)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Title
    title_label = ttk.Label(
        main_frame,
        text="MetaConsciousness LM Studio Reset Utility",
        font=("Arial", 14, "bold")
    )
    title_label.pack(pady=(0, 20))

    # Explanation text
    explanation = ttk.Label(
        main_frame,
        text=(
            "This utility will reset the LM Studio connection settings when the GUI\n"
            "is stuck in fallback mode and won't connect to LM Studio.\n\n"
            "Steps to resolve the issue:\n"
            "1. Close the MetaConsciousness GUI if it's running\n"
            "2. Make sure LM Studio is running with a model loaded\n"
            "3. Click the Reset button below\n"
            "4. Restart the MetaConsciousness GUI"
        ),
        justify=tk.LEFT,
        wraplength=450
    )
    explanation.pack(pady=10, fill=tk.X)

    # Status frame
    status_frame = ttk.LabelFrame(main_frame, text="Status")
    status_frame.pack(fill=tk.X, pady=10)

    # LM Studio process status
    lmstudio_running = check_lmstudio_process()
    lmstudio_status = ttk.Label(
        status_frame,
        text=f"LM Studio Running: {'Yes' if lmstudio_running else 'No'}",
        foreground="green" if lmstudio_running else "red"
    )
    lmstudio_status.pack(pady=5, padx=10, anchor=tk.W)

    # Config status
    config_path = os.path.join(os.path.expanduser("~"), ".metaconsciousness", "config", "lmstudio_state.json")
    config_exists = os.path.exists(config_path)
    config_status = ttk.Label(
        status_frame,
        text=f"Config File: {'Exists' if config_exists else 'Does not exist'}",
        foreground="green" if config_exists else "orange"
    )
    config_status.pack(pady=5, padx=10, anchor=tk.W)

    # Status message
    status_message = tk.StringVar(value="Ready")
    status_label = ttk.Label(
        main_frame,
        textvariable=status_message,
        font=("Arial", 10, "italic")
    )
    status_label.pack(pady=10, fill=tk.X)

    # Buttons frame
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(pady=10, fill=tk.X)

    # Launch LM Studio button
    launch_button = ttk.Button(
        button_frame,
        text="Launch LM Studio",
        command=lambda: launch_and_update(),
        style="Launch.TButton"
    )
    launch_button.pack(side=tk.LEFT, padx=10)

    # Reset button
    reset_button = ttk.Button(
        button_frame,
        text="Reset LM Studio Connection",
        command=lambda: reset_and_update(),
        style="Reset.TButton"
    )
    reset_button.pack(side=tk.RIGHT, padx=10)

    # Close button
    close_button = ttk.Button(
        main_frame,
        text="Close",
        command=root.destroy
    )
    close_button.pack(side=tk.BOTTOM, pady=10)

    def launch_and_update() -> None:
        """Launch LM Studio and update the UI."""
        status_message.set("Launching LM Studio...")
        root.update_idletasks()

        success = launch_lmstudio()
        if success:
            status_message.set("LM Studio launched successfully")
            # Update status after a delay to give time for LM Studio to start
            root.after(3000, update_status)
        else:
            status_message.set("Failed to launch LM Studio. Please launch it manually.")

    def reset_and_update() -> None:
        """Reset the LM Studio connection and update the UI."""
        status_message.set("Resetting LM Studio connection...")
        root.update_idletasks()

        success = reset_lmstudio_config()
        if success:
            status_message.set("Reset successful. You can now restart the MetaConsciousness GUI.")
            messagebox.showinfo("Reset Complete", "Reset successful. You can now restart the MetaConsciousness GUI.")
        else:
            status_message.set("Reset failed. Check the console for details.")
            messagebox.showerror("Reset Failed", "Failed to reset LM Studio connection settings. Check the console for details.")

        # Update status
        update_status()

    def update_status() -> None:
        """Update the status display."""
        lmstudio_running = check_lmstudio_process()
        lmstudio_status.config(
            text=f"LM Studio Running: {'Yes' if lmstudio_running else 'No'}",
            foreground="green" if lmstudio_running else "red"
        )

        config_exists = os.path.exists(config_path)
        config_status.config(
            text=f"Config File: {'Exists' if config_exists else 'Does not exist'}",
            foreground="green" if config_exists else "orange"
        )

    # Start with status check
    update_status()

    return root

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Reset LM Studio connection settings")
    parser.add_argument("--no-gui", action="store_true", help="Run in command-line mode without GUI")
    parser.add_argument("--force", action="store_true", help="Force reset without confirmation in command-line mode")

    args = parser.parse_args()

    if args.no_gui:
        # Command-line mode
        if not args.force:
            confirmation = input("Are you sure you want to reset the LM Studio connection settings? (y/n): ")
            if confirmation.lower() != 'y':
                print("Reset cancelled.")
                return 1

        success = reset_lmstudio_config()
        if success:
            print("Reset successful. You can now restart the MetaConsciousness GUI.")
            return 0
        else:
            print("Reset failed. Check the logs for details.")
            return 1
    else:
        # GUI mode
        root = create_gui()
        root.mainloop()
        return 0

if __name__ == "__main__":
    sys.exit(main())
