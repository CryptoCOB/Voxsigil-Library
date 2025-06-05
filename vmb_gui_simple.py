#!/usr/bin/env python3
"""
🔥 VMB-GUI Integration Launcher (Simplified)
Connects the Visual Model Bootstrap (VMB) system with the VoxSigil GUI
Following BootSigil directive with sigil ⟠∆∇𓂀
"""

import asyncio
import logging
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from typing import Any, Dict

import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("VMB_GUI_Launcher")

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "GUI" / "components"))

# Import VMB system
try:
    from vmb_activation import CopilotSwarm

    VMB_AVAILABLE = True
except ImportError as e:
    logger.error(f"VMB not available: {e}")
    VMB_AVAILABLE = False

# Import Production Executor
try:
    from vmb_production_executor import ProductionTaskExecutor

    PRODUCTION_AVAILABLE = True
except ImportError as e:
    logger.error(f"Production executor not available: {e}")
    PRODUCTION_AVAILABLE = False

# Import GUI components
try:
    from dynamic_gridformer_gui import DynamicGridFormerGUI

    GUI_AVAILABLE = True
except ImportError as e:
    logger.warning(f"GUI not available: {e}")
    GUI_AVAILABLE = False


class VMBGUILauncher:
    """Simple VMB-GUI Integration Launcher."""

    def __init__(self):
        self.config = self._load_config()
        self.vmb_swarm = None
        self.production_executor = None
        self.root = None

        logger.info("🚀 VMB-GUI Launcher initialized")
        logger.info(f"🔮 Sigil: {self.config.get('sigil', '⟠∆∇𓂀')}")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from sigil_trace.yaml."""
        config_path = PROJECT_ROOT / "sigil_trace.yaml"
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Config not found, using defaults: {e}")
            return {
                "sigil": "⟠∆∇𓂀",
                "agent_class": "CopilotSwarm",
                "swarm_variant": "RPG_Sentinel",
                "role_scope": ["planner", "validator", "executor", "summarizer"],
                "activation_mode": "VMB_FirstRun",
            }

    async def initialize_vmb(self):
        """Initialize VMB system."""
        if not VMB_AVAILABLE:
            logger.error("❌ VMB system not available")
            return False

        try:
            self.vmb_swarm = CopilotSwarm(self.config)
            await self.vmb_swarm.initialize_swarm()
            logger.info("✅ VMB CopilotSwarm initialized")

            if PRODUCTION_AVAILABLE:
                self.production_executor = ProductionTaskExecutor()
                await self.production_executor.initialize()
                logger.info("✅ Production executor initialized")

            return True
        except Exception as e:
            logger.error(f"❌ VMB initialization failed: {e}")
            return False

    def create_gui(self):
        """Create the integrated GUI."""
        self.root = tk.Tk()
        self.root.title(
            f"🧠 VoxSigil VMB Interface - {self.config.get('sigil', '⟠∆∇𓂀')}"
        )
        self.root.geometry("1200x800")
        self.root.configure(bg="#1a1a2e")

        # Create main frame
        main_frame = tk.Frame(self.root, bg="#1a1a2e")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # VMB Status Panel
        self._create_vmb_status_panel(main_frame)

        # Control Panel
        self._create_control_panel(main_frame)

        # Log Panel
        self._create_log_panel(main_frame)

        # Initialize GUI components if available
        if GUI_AVAILABLE:
            try:
                # Embed the main GUI in a frame
                gui_frame = tk.Frame(main_frame, bg="#2a2a4e", relief=tk.RAISED, bd=2)
                gui_frame.pack(fill=tk.BOTH, expand=True, pady=10)

                gui_label = tk.Label(
                    gui_frame,
                    text="🎯 VoxSigil Dynamic GridFormer Interface",
                    bg="#2a2a4e",
                    fg="#00ff88",
                    font=("Consolas", 14, "bold"),
                )
                gui_label.pack(pady=10)

                # Create a button to launch the full GUI
                launch_btn = tk.Button(
                    gui_frame,
                    text="🚀 Launch Full GridFormer GUI",
                    command=self._launch_full_gui,
                    bg="#ff6b35",
                    fg="white",
                    font=("Consolas", 12, "bold"),
                    relief=tk.RAISED,
                    bd=2,
                )
                launch_btn.pack(pady=10)

            except Exception as e:
                logger.error(f"GUI embedding failed: {e}")

        return True

    def _create_vmb_status_panel(self, parent):
        """Create VMB status panel."""
        status_frame = tk.Frame(parent, bg="#2a2a4e", relief=tk.RAISED, bd=2)
        status_frame.pack(fill=tk.X, pady=5)

        title_label = tk.Label(
            status_frame,
            text="🤖 VMB CopilotSwarm Status",
            bg="#2a2a4e",
            fg="#00ff88",
            font=("Consolas", 14, "bold"),
        )
        title_label.pack(pady=10)

        # Status indicators
        status_text = f"Status: {'🟢 Active' if self.vmb_swarm and self.vmb_swarm.active else '🔴 Inactive'}"
        status_label = tk.Label(
            status_frame,
            text=status_text,
            bg="#2a2a4e",
            fg="#ffaa00",
            font=("Consolas", 11),
        )
        status_label.pack(pady=5)

        # Agent count
        if self.vmb_swarm and hasattr(self.vmb_swarm, "agents"):
            agent_count = len(self.vmb_swarm.agents)
        else:
            agent_count = 0

        agent_label = tk.Label(
            status_frame,
            text=f"Active Agents: {agent_count}/4 [planner, validator, executor, summarizer]",
            bg="#2a2a4e",
            fg="#88aaff",
            font=("Consolas", 10),
        )
        agent_label.pack(pady=5)

        # Sigil display
        sigil_label = tk.Label(
            status_frame,
            text=f"Bound Sigil: {self.config.get('sigil', '⟠∆∇𓂀')}",
            bg="#2a2a4e",
            fg="#ff88aa",
            font=("Consolas", 12, "bold"),
        )
        sigil_label.pack(pady=5)

    def _create_control_panel(self, parent):
        """Create control panel."""
        control_frame = tk.Frame(parent, bg="#2a2a4e", relief=tk.RAISED, bd=2)
        control_frame.pack(fill=tk.X, pady=5)

        title_label = tk.Label(
            control_frame,
            text="🎮 VMB Control Panel",
            bg="#2a2a4e",
            fg="#00ff88",
            font=("Consolas", 14, "bold"),
        )
        title_label.pack(pady=10)

        # Button frame
        btn_frame = tk.Frame(control_frame, bg="#2a2a4e")
        btn_frame.pack(pady=10)

        # Execute task button
        execute_btn = tk.Button(
            btn_frame,
            text="🚀 Execute VMB Task",
            command=self._execute_vmb_task,
            bg="#ff6b35",
            fg="white",
            font=("Consolas", 11, "bold"),
            relief=tk.RAISED,
            bd=2,
            width=20,
        )
        execute_btn.pack(side=tk.LEFT, padx=10)

        # System status button
        status_btn = tk.Button(
            btn_frame,
            text="📊 System Status",
            command=self._show_system_status,
            bg="#4ecdc4",
            fg="white",
            font=("Consolas", 11, "bold"),
            relief=tk.RAISED,
            bd=2,
            width=20,
        )
        status_btn.pack(side=tk.LEFT, padx=10)

        # Reinitialize button
        reinit_btn = tk.Button(
            btn_frame,
            text="🔄 Reinitialize VMB",
            command=self._reinitialize_vmb,
            bg="#9b59b6",
            fg="white",
            font=("Consolas", 11, "bold"),
            relief=tk.RAISED,
            bd=2,
            width=20,
        )
        reinit_btn.pack(side=tk.LEFT, padx=10)

    def _create_log_panel(self, parent):
        """Create log display panel."""
        log_frame = tk.Frame(parent, bg="#2a2a4e", relief=tk.RAISED, bd=2)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        title_label = tk.Label(
            log_frame,
            text="📋 System Log",
            bg="#2a2a4e",
            fg="#00ff88",
            font=("Consolas", 14, "bold"),
        )
        title_label.pack(pady=10)

        # Text area for logs
        self.log_text = tk.Text(
            log_frame,
            bg="#1a1a1a",
            fg="#00ff00",
            font=("Consolas", 9),
            wrap=tk.WORD,
            height=10,
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Add scrollbar
        scrollbar = tk.Scrollbar(self.log_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.log_text.yview)

        # Initial log entry
        self._log_message("🚀 VMB-GUI Launcher initialized")
        self._log_message(f"🔮 Sigil bound: {self.config.get('sigil', '⟠∆∇𓂀')}")

    def _log_message(self, message):
        """Add message to log panel."""
        if hasattr(self, "log_text"):
            self.log_text.insert(tk.END, f"{message}\n")
            self.log_text.see(tk.END)

    def _execute_vmb_task(self):
        """Execute VMB task."""
        if not self.production_executor:
            messagebox.showwarning("Warning", "Production executor not initialized")
            return

        self._log_message("🚀 Executing VMB production task...")

        # Create a sample task
        task = {
            "name": "Sample VMB Task",
            "type": "demonstration",
            "description": "Testing VMB system functionality",
        }

        # Execute in background
        asyncio.create_task(self._async_execute_task(task))

    async def _async_execute_task(self, task):
        """Async task execution."""
        try:
            result = await self.production_executor.execute_production_task(task)
            if result.get("success", False):
                self._log_message("✅ VMB task completed successfully")
                messagebox.showinfo("Success", "VMB task executed successfully!")
            else:
                error_msg = result.get("error", "Unknown error")
                self._log_message(f"❌ VMB task failed: {error_msg}")
                messagebox.showerror("Error", f"Task failed: {error_msg}")
        except Exception as e:
            self._log_message(f"❌ Task execution error: {e}")
            messagebox.showerror("Error", f"Execution error: {e}")

    def _show_system_status(self):
        """Show system status."""
        status_info = f"""
VMB System Status:
- Sigil: {self.config.get("sigil", "⟠∆∇𓂀")}
- Agent Class: {self.config.get("agent_class", "Unknown")}
- Variant: {self.config.get("swarm_variant", "Unknown")}
- CopilotSwarm: {"🟢 Active" if self.vmb_swarm and self.vmb_swarm.active else "🔴 Inactive"}
- Production Executor: {"🟢 Ready" if self.production_executor else "🔴 Not Available"}
- GUI Integration: {"🟢 Available" if GUI_AVAILABLE else "🔴 Not Available"}
        """

        messagebox.showinfo("System Status", status_info)
        self._log_message("📊 System status displayed")

    def _reinitialize_vmb(self):
        """Reinitialize VMB system."""
        self._log_message("🔄 Reinitializing VMB system...")
        asyncio.create_task(self._async_reinitialize())

    async def _async_reinitialize(self):
        """Async reinitialize."""
        try:
            success = await self.initialize_vmb()
            if success:
                self._log_message("✅ VMB system reinitialized successfully")
                messagebox.showinfo("Success", "VMB system reinitialized!")
            else:
                self._log_message("❌ VMB reinitialize failed")
                messagebox.showerror("Error", "Reinitialize failed")
        except Exception as e:
            self._log_message(f"❌ Reinitialize error: {e}")
            messagebox.showerror("Error", f"Reinitialize error: {e}")

    def _launch_full_gui(self):
        """Launch the full GridFormer GUI."""
        try:
            import subprocess

            subprocess.Popen([sys.executable, "run_complete_gui.py"], cwd=PROJECT_ROOT)
            self._log_message("🚀 Full GridFormer GUI launched")
        except Exception as e:
            self._log_message(f"❌ Failed to launch full GUI: {e}")
            messagebox.showerror("Error", f"Failed to launch GUI: {e}")

    async def run(self):
        """Main run method."""
        logger.info("🌟 Starting VMB-GUI Launcher...")

        # Initialize VMB system
        vmb_success = await self.initialize_vmb()

        # Create and show GUI
        gui_success = self.create_gui()

        if gui_success:
            logger.info("✅ VMB-GUI Launcher ready!")
            self.root.mainloop()
        else:
            logger.error("❌ Failed to create GUI")


async def main():
    """Main entry point."""
    print("=" * 80)
    print("🔥 VMB-GUI INTEGRATION LAUNCHER")
    print("🔮 Sigil: ⟠∆∇𓂀")
    print("🤖 Following BootSigil Directive v1.6")
    print("=" * 80)

    launcher = VMBGUILauncher()
    await launcher.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Launcher stopped by user")
    except Exception as e:
        logger.error(f"❌ Critical error: {e}")
        sys.exit(1)
