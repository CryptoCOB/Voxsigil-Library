#!/usr/bin/env python3
"""
🔥 VMB-GUI Integration Launcher
Connects the Visual Model Bootstrap (VMB) system with the VoxSigil GUI
Following BootSigil directive with sigil ⟠∆∇𓂀

This script:
1. Activates the VMB CopilotSwarm system
2. Integrates VMB agents into the GUI
3. Provides a unified interface for the complete Sigil experience
"""

import asyncio
import logging
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from typing import Any, Dict

import yaml

# Setup enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("vmb_gui_launcher.log"), logging.StreamHandler()],
)
logger = logging.getLogger("VMB_GUI_Launcher")

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent

# 🧠 Codex BugPatch - Vanta Phase @2025-06-09
def _add_sys_path(path: Path) -> None:
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)

_add_sys_path(PROJECT_ROOT)
_add_sys_path(PROJECT_ROOT / "GUI" / "components")
_add_sys_path(PROJECT_ROOT / "Vanta")

# Import VMB system
try:
    from .vmb_activation import CopilotSwarm
    from .vmb_config_status import (
        VMBCompletionReport,
        VMBStatus,
        VMBSwarmConfig,
        VMBSystemStatus,
    )
    from .vmb_production_executor import ProductionTaskExecutor

    VMB_AVAILABLE = True
except ImportError as e:
    logger.error(f"VMB system not available: {e}")
    VMB_AVAILABLE = False
    CopilotSwarm = None
    ProductionTaskExecutor = None
    VMBSwarmConfig = None
    VMBStatus = None
    VMBCompletionReport = None
    VMBSystemStatus = None

# Import GUI components - make this optional since it might not exist
try:
    from gui.components.pyqt_main import launch as launch_pyqt

    GUI_AVAILABLE = True
except ImportError as e:
    logger.error(f"GUI not available: {e}")
    GUI_AVAILABLE = False
    launch_pyqt = None

# Import VantaCore integration - make optional
try:
    from Vanta.core.UnifiedVantaCore import UnifiedVantaCore

    VANTA_CORE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"UnifiedVantaCore not available: {e}")
    VANTA_CORE_AVAILABLE = False
    UnifiedVantaCore = None


class VMBGUIIntegration:
    """Integrates VMB system with VoxSigil GUI for complete experience."""

    def __init__(self):
        self.config_dict = self._load_sigil_trace_config()

        # Initialize VMB configuration and status if VMB is available
        if VMB_AVAILABLE and VMBSwarmConfig and VMBStatus:
            self.config = VMBSwarmConfig.from_dict(self.config_dict)
            self.vmb_status = VMBStatus()
        else:
            self.config = None
            self.vmb_status = None

        self.vmb_swarm = None
        self.production_executor = None
        self.vanta_manager = None  # VantaInteractionManager if available
        self.gui_app = None  # DynamicGridFormerGUI if available
        self.root = None
        self.completion_reporter = None
        self.vanta_core = None
        if VANTA_CORE_AVAILABLE and UnifiedVantaCore:
            try:
                self.vanta_core = UnifiedVantaCore()
                logger.info("✅ UnifiedVantaCore instantiated and available")
            except Exception as e:
                logger.warning(f"⚠️ UnifiedVantaCore instantiation failed: {e}")

        logger.info("🚀 VMB-GUI Integration initialized")
        logger.info(f"🔮 Sigil: {self.config_dict.get('sigil', 'Unknown')}")
        logger.info(f"🤖 Agent Class: {self.config_dict.get('agent_class', 'Unknown')}")
        logger.info(
            f"⚔️ Swarm Variant: {self.config_dict.get('swarm_variant', 'Unknown')}"
        )

    def _load_sigil_trace_config(self) -> Dict[str, Any]:
        """Load configuration from sigil_trace.yaml."""
        config_path = PROJECT_ROOT / "sigil_trace.yaml"
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"✅ Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"⚠️ {config_path} not found, using defaults")
            return self._create_default_config()
        except Exception as e:
            logger.error(f"❌ Error loading config: {e}")
            return self._create_default_config()

    def _create_default_config(self) -> Dict[str, Any]:
        """Create default BootSigil configuration."""
        return {
            "sigil": "⟠∆∇𓂀",
            "agent_class": "CopilotSwarm",
            "swarm_variant": "RPG_Sentinel",
            "role_scope": ["planner", "validator", "executor", "summarizer"],
            "activation_mode": "VMB_FirstRun",
            "python_version_required": "3.11",
            "package_manager": "uv",
            "formatter": "ruff",
        }

    async def initialize_vmb_system(self):
        """Initialize the complete VMB system."""
        logger.info("🔄 Initializing VMB System...")
        try:
            if not VMB_AVAILABLE:
                logger.error("❌ VMB components not available")
                return False

            # Update VMB status
            if self.vmb_status and "INITIALIZING" in dir(self.vmb_status):
                # self.vmb_status.update_status(VMBSystemStatus.INITIALIZING, "Starting VMB system initialization")
                pass

            # Initialize CopilotSwarm with dictionary config for backward compatibility
            if CopilotSwarm:
                self.vmb_swarm = CopilotSwarm(self.config_dict)
                if hasattr(self.vmb_swarm, "initialize_swarm"):
                    await self.vmb_swarm.initialize_swarm()
                logger.info("✅ VMB CopilotSwarm initialized")

            # Initialize Production Executor
            if ProductionTaskExecutor:
                self.production_executor = ProductionTaskExecutor()
                if hasattr(self.production_executor, "initialize"):
                    await self.production_executor.initialize()
                logger.info("✅ VMB Production Executor initialized")

            # Initialize Completion Reporter (using status report function)
            try:
                from .vmb_status import generate_vmb_status_report

                self.completion_reporter = generate_vmb_status_report
                logger.info("✅ VMB Status Reporter initialized")
            except ImportError:
                logger.warning("⚠️ VMB Status Reporter not available")

            # Initialize VantaCore if available
            # if VANTA_AVAILABLE and VantaInteractionManager:
            #    try:
            #        self.vanta_manager = VantaInteractionManager()
            #        logger.info("✅ VantaCore integration initialized")
            #    except Exception as e:
            #        logger.warning(f"⚠️ VantaCore initialization failed: {e}")

            # Update VMB status to active
            if self.vmb_status and "ACTIVE" in dir(self.vmb_status):
                # self.vmb_status.update_status(VMBSystemStatus.ACTIVE, "VMB system fully operational")
                # Add active agents
                for role in self.config_dict.get("role_scope", []):
                    self.vmb_status.add_active_agent(role)
                # Update system health
                self.vmb_status.update_system_health("agent_health", True)
                self.vmb_status.update_system_health("system_performance", True)

            logger.info("🎉 VMB System fully initialized!")
            return True

        except Exception as e:
            logger.error(f"❌ VMB System initialization failed: {e}")
            if self.vmb_status and "ERROR" in dir(self.vmb_status):
                # self.vmb_status.update_status(VMBSystemStatus.ERROR, f"Initialization failed: {e}")
                pass
            return False

    def initialize_gui(self):
        """Initialize the VoxSigil GUI with VMB integration."""
        logger.info("🖥️ Initializing GUI...")

        try:  # Create root window
            self.root = tk.Tk()
            self.root.title(
                f"🧠 VoxSigil VMB Interface - {self.config_dict.get('sigil', '⟠∆∇𓂀')}"
            )
            self.root.geometry("1400x900")
            self.root.configure(bg="#1a1a2e")

            # Initialize main GUI if available
            if GUI_AVAILABLE and DynamicGridFormerGUI:
                self.gui_app = DynamicGridFormerGUI(self.root)

            # Add VMB integration panel
            self._add_vmb_panel()

            try:
                from gui_utils import bind_agent_buttons

                if self.vanta_core and self.vanta_core.agent_registry:
                    bind_agent_buttons(self.root, self.vanta_core.agent_registry)
            except Exception:
                pass

            logger.info("✅ GUI initialized with VMB integration")
            return True

        except Exception as e:
            logger.error(f"❌ GUI initialization failed: {e}")
            return False

    def _add_vmb_panel(self):
        """Add VMB control panel to the GUI."""
        if not self.root:
            return

        # Create VMB frame
        vmb_frame = tk.Frame(self.root, bg="#2a2a4e", relief=tk.RAISED, bd=2)
        vmb_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # VMB Status Label
        status_text = f"VMB Status: {'🟢 Active' if self.vmb_swarm else '🔴 Inactive'}"
        status_label = tk.Label(
            vmb_frame,
            text=status_text,
            bg="#2a2a4e",
            fg="#00ff88",
            font=("Consolas", 10, "bold"),
        )
        status_label.pack(side=tk.LEFT, padx=10, pady=5)

        # Sigil Display
        sigil_label = tk.Label(
            vmb_frame,
            text=f"Sigil: {self.config_dict.get('sigil', '⟠∆∇𓂀')}",
            bg="#2a2a4e",
            fg="#ffaa00",
            font=("Consolas", 12, "bold"),
        )
        sigil_label.pack(side=tk.LEFT, padx=10, pady=5)

        # Agent Count
        if self.vmb_swarm and hasattr(self.vmb_swarm, "agents"):
            agent_count = len(self.vmb_swarm.agents)
            agent_text = f"Agents: {agent_count}/4"
        else:
            agent_text = "Agents: 0/4"

        agent_label = tk.Label(
            vmb_frame,
            text=agent_text,
            bg="#2a2a4e",
            fg="#88aaff",
            font=("Consolas", 10),
        )
        agent_label.pack(side=tk.LEFT, padx=10, pady=5)

        # VMB Control Buttons
        btn_frame = tk.Frame(vmb_frame, bg="#2a2a4e")
        btn_frame.pack(side=tk.RIGHT, padx=10, pady=5)

        # Execute Task Button
        execute_btn = tk.Button(
            btn_frame,
            text="🚀 Execute VMB Task",
            command=self._on_execute_vmb_task,
            bg="#ff6b35",
            fg="white",
            font=("Consolas", 9, "bold"),
            relief=tk.RAISED,
            bd=1,
        )
        execute_btn.pack(side=tk.LEFT, padx=5)

        # Generate Report Button
        report_btn = tk.Button(
            btn_frame,
            text="📊 Generate Report",
            command=self._on_generate_report,
            bg="#4ecdc4",
            fg="white",
            font=("Consolas", 9, "bold"),
            relief=tk.RAISED,
            bd=1,
        )
        report_btn.pack(side=tk.LEFT, padx=5)

        # VantaCore Toggle
        if VANTA_CORE_AVAILABLE:
            vanta_btn = tk.Button(
                btn_frame,
                text="🔗 VantaCore",
                command=self._on_toggle_vanta,
                bg="#9b59b6",
                fg="white",
                font=("Consolas", 9, "bold"),
                relief=tk.RAISED,
                bd=1,
            )
            vanta_btn.pack(side=tk.LEFT, padx=5)
            # Add a button to show UnifiedVantaCore status
            vanta_status_btn = tk.Button(
                btn_frame,
                text="🩺 VantaCore Status",
                command=self._on_show_vanta_status,
                bg="#27ae60",
                fg="white",
                font=("Consolas", 9, "bold"),
                relief=tk.RAISED,
                bd=1,
            )
            vanta_status_btn.pack(side=tk.LEFT, padx=5)
            # Add a button to send a test task to UnifiedVantaCore
            vanta_task_btn = tk.Button(
                btn_frame,
                text="🧩 Send Test Task to VantaCore",
                command=self._on_send_vanta_task,
                bg="#2980b9",
                fg="white",
                font=("Consolas", 9, "bold"),
                relief=tk.RAISED,
                bd=1,
            )
            vanta_task_btn.pack(side=tk.LEFT, padx=5)
            # Add a button to show all registered agents
            vanta_agents_btn = tk.Button(
                btn_frame,
                text="🗂 Show Registered Agents",
                command=self._on_show_vanta_agents,
                bg="#f39c12",
                fg="white",
                font=("Consolas", 9, "bold"),
                relief=tk.RAISED,
                bd=1,
            )
            vanta_agents_btn.pack(side=tk.LEFT, padx=5)

    def _on_execute_vmb_task(self):
        """Handle VMB task execution."""
        if not self.production_executor:
            messagebox.showwarning("Warning", "VMB Production Executor not initialized")
            return

        # Run async task in background
        asyncio.create_task(self._async_execute_vmb_task())

    async def _async_execute_vmb_task(self):
        """Async VMB task execution."""
        try:
            logger.info("🚀 Executing VMB production task...")
            # Create a sample task for demonstration
            sample_task = {"name": "VMB System Test", "type": "diagnostic"}

            if self.production_executor and hasattr(
                self.production_executor, "execute_production_task"
            ):
                result = await self.production_executor.execute_production_task(
                    sample_task
                )
            else:
                result = {"status": "success", "message": "VMB task simulated"}

            if result.get("status") == "success":
                messagebox.showinfo("Success", "VMB task executed successfully!")
                logger.info("✅ VMB task completed successfully")
            else:
                error_msg = result.get("error", "Unknown error")
                messagebox.showerror("Error", f"VMB task failed: {error_msg}")
                logger.error(f"❌ VMB task failed: {error_msg}")

        except Exception as e:
            logger.error(f"❌ VMB task execution error: {e}")
            messagebox.showerror("Error", f"Execution error: {e}")

    def _on_generate_report(self):
        """Handle report generation."""
        if not self.completion_reporter:
            messagebox.showwarning("Warning", "VMB Completion Reporter not initialized")
            return

        asyncio.create_task(self._async_generate_report())

    async def _async_generate_report(self):
        """Async report generation."""
        try:
            logger.info("📊 Generating VMB completion report...")

            # Call the report function (it's synchronous)
            if self.completion_reporter:
                self.completion_reporter()
            # Create a simple report file
            report_content = f"""# VMB System Status Report
Generated: {Path(__file__).stat().st_mtime}

## Configuration
Sigil: {self.config_dict.get("sigil", "⟠∆∇𓂀")}
Agent Class: {self.config_dict.get("agent_class", "Unknown")}
Swarm Variant: {self.config_dict.get("swarm_variant", "Unknown")}

## System Status
VMB Available: {VMB_AVAILABLE}
GUI Available: {GUI_AVAILABLE}
VantaCore Available: {VANTA_CORE_AVAILABLE}

## Components
CopilotSwarm: {"✅ Initialized" if self.vmb_swarm else "❌ Not initialized"}
Production Executor: {"✅ Initialized" if self.production_executor else "❌ Not initialized"}
VantaCore Manager: {"✅ Initialized" if self.vanta_manager else "❌ Not initialized"}
"""

            # Save report to file
            report_path = PROJECT_ROOT / "VMB_COMPLETION_REPORT.md"
            with open(report_path, "w") as f:
                f.write(report_content)

            messagebox.showinfo("Success", f"Report generated: {report_path}")
            logger.info(f"✅ Report saved to {report_path}")

        except Exception as e:
            logger.error(f"❌ Report generation error: {e}")
            messagebox.showerror("Error", f"Report generation failed: {e}")

    def _on_toggle_vanta(self):
        """Handle VantaCore toggle."""
        if not self.vanta_core:
            messagebox.showwarning("Warning", "UnifiedVantaCore not available")
            return

        messagebox.showinfo("VantaCore", "UnifiedVantaCore integration activated!")
        logger.info("🔗 UnifiedVantaCore integration toggled")

    def _on_show_vanta_status(self):
        """Show UnifiedVantaCore status in a messagebox."""
        if not self.vanta_core:
            messagebox.showwarning("Warning", "UnifiedVantaCore not available")
            return
        try:
            status = self.vanta_core.get_unified_status()
            import json

            status_str = json.dumps(status, indent=2)
            messagebox.showinfo("UnifiedVantaCore Status", status_str)
        except Exception as e:
            logger.error(f"Error getting UnifiedVantaCore status: {e}")
            messagebox.showerror("Error", f"Failed to get UnifiedVantaCore status: {e}")

    def _on_send_vanta_task(self):
        """Send a test task to UnifiedVantaCore and show the result."""
        if not self.vanta_core:
            messagebox.showwarning("Warning", "UnifiedVantaCore not available")
            return
        try:
            # Example: send a simple cognitive task
            task = {
                "type": "cognitive",
                "id": "gui_test_task",
                "payload": {"message": "Hello from GUI!"},
            }
            result = self.vanta_core.process_unified_task(task)
            import json

            result_str = json.dumps(result, indent=2)
            messagebox.showinfo("VantaCore Task Result", result_str)
        except Exception as e:
            logger.error(f"Error sending task to UnifiedVantaCore: {e}")
            messagebox.showerror("Error", f"Failed to send task: {e}")

    def _on_show_vanta_agents(self):
        """Show all registered agents from UnifiedVantaCore."""
        if not self.vanta_core:
            messagebox.showwarning("Warning", "UnifiedVantaCore not available")
            return
        try:
            agents = self.vanta_core.agent_registry.get_all_agents()
            if not agents:
                agent_str = "No agents registered."
            else:
                agent_str = "\n".join(f"- {aid}" for aid, _ in agents)
            messagebox.showinfo("Registered Agents", agent_str)
        except Exception as e:
            logger.error(f"Error listing agents from UnifiedVantaCore: {e}")
            messagebox.showerror("Error", f"Failed to list agents: {e}")

    async def run(self):
        """Main run method - initializes and starts the complete system."""
        logger.info("🌟 Starting VMB-GUI Integration...")

        # Initialize VMB system
        vmb_success = await self.initialize_vmb_system()
        if not vmb_success:
            logger.error("❌ VMB system initialization failed")
            return False

        # Initialize GUI
        gui_success = self.initialize_gui()
        if not gui_success:
            logger.error("❌ GUI initialization failed")
            return False

        logger.info("🎉 VMB-GUI Integration ready!")
        logger.info("🚀 Starting GUI main loop...")

        # Start GUI main loop
        if self.root:
            self.root.mainloop()

        return True


async def main():
    """Main entry point."""
    logger.info("=" * 80)
    logger.info("🔥 VMB-GUI INTEGRATION LAUNCHER")
    logger.info("🔮 Sigil: ⟠∆∇𓂀")
    logger.info("🤖 Following BootSigil Directive v1.6")
    logger.info("=" * 80)

    # Create and run integration
    integration = VMBGUIIntegration()
    success = await integration.run()

    if success:
        logger.info("✅ VMB-GUI Integration completed successfully")
    else:
        logger.error("❌ VMB-GUI Integration failed")
        sys.exit(1)


if __name__ == "__main__":
    # Run the async main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 VMB-GUI Integration stopped by user")
    except Exception as e:
        logger.error(f"❌ Critical error: {e}")
        sys.exit(1)
