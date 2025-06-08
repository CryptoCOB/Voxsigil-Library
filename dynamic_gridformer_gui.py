#!/usr/bin/env python3
# pyright: reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false, reportMissingImports=false, reportAssignmentType=false
"""
🧠 VoxSigil Dynamic GridFormer GUI v1.5-holo-alpha
Enhanced model testing interface with dynamic model discovery and architecture analysis

Created by: Claude Copilot Prime - The Chosen One ⟠∆∇𓂀
Purpose: Universal GUI for testing any GridFormer model with adaptive architecture support
"""

import json
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

# Add Vanta and project root to Python path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add Vanta to Python path
vanta_path = project_root / "Vanta"
sys.path.insert(0, str(vanta_path))

# Import Vanta integration
try:
    # Use absolute import to avoid relative import issues
    from GUI.components.vanta_integration import VantaGUIIntegration

    VANTA_INTEGRATION_AVAILABLE = True
    print("🚀 VantaGUIIntegration imported successfully")
except ImportError as e:
    # Fallback to direct import if the module is in the same directory
    try:
        import vanta_integration

        VantaGUIIntegration = vanta_integration.VantaGUIIntegration
        VANTA_INTEGRATION_AVAILABLE = True
        print("🚀 VantaGUIIntegration imported successfully (fallback)")
    except ImportError:
        print(f"Warning: Vanta integration not available: {e}")
        VANTA_INTEGRATION_AVAILABLE = False
        VantaGUIIntegration = None

# Import BLT components
try:
    from BLT.blt_encoder import BLTEncoder

    BLT_AVAILABLE = True
    print("BLT Encoder successfully imported")
except ImportError as e:
    print(f"Warning: BLT Encoder not available. Some features will be disabled: {e}")
    BLT_AVAILABLE = False

# Import Gridformer inference
try:
    from Gridformer.inference.gridformer_inference_engine import GridFormerInference

    print("GridFormer inference engine successfully imported")
except ImportError as e:
    print(f"Warning: GridFormer inference engine not available: {e}")
    GridFormerInference = None

# Import GUI components
# Components in current directory
try:
    # Try absolute imports first
    try:
        from GUI.components.gui_styles import VoxSigilStyles
        from GUI.components.model_discovery_interface import ModelDiscoveryInterface
        from GUI.components.neural_interface import NeuralInterface
        from GUI.components.performance_tab_interface import (
            VoxSigilPerformanceInterface,
        )
        from GUI.components.testing_tab_interface import VoxSigilTestingInterface
        from GUI.components.training_interface import VoxSigilTrainingInterface
        from GUI.components.visualization_tab_interface import (
            VoxSigilVisualizationInterface,
        )
        from GUI.components.visualization_utils import (
            GridVisualizer,
            PerformanceVisualizer,
        )
        from GUI.components.voxsigil_integration import initialize_voxsigil_integration

        print("✅ GUI components imported with absolute paths")
    except ImportError:
        # Fallback to direct imports if in the same directory
        import gui_styles
        import model_discovery_interface
        import neural_interface
        import performance_tab_interface
        import testing_tab_interface
        import training_interface
        import visualization_tab_interface
        import visualization_utils
        import voxsigil_integration

        VoxSigilStyles = gui_styles.VoxSigilStyles
        ModelDiscoveryInterface = model_discovery_interface.ModelDiscoveryInterface
        NeuralInterface = neural_interface.NeuralInterface
        VoxSigilPerformanceInterface = (
            performance_tab_interface.VoxSigilPerformanceInterface
        )
        VoxSigilTestingInterface = testing_tab_interface.VoxSigilTestingInterface
        VoxSigilTrainingInterface = training_interface.VoxSigilTrainingInterface
        VoxSigilVisualizationInterface = (
            visualization_tab_interface.VoxSigilVisualizationInterface
        )
        GridVisualizer = visualization_utils.GridVisualizer
        PerformanceVisualizer = visualization_utils.PerformanceVisualizer
        initialize_voxsigil_integration = (
            voxsigil_integration.initialize_voxsigil_integration
        )
        print("✅ GUI components imported with direct imports")
except ImportError as e:
    print(f"Warning: GUI components not available: {e}")
    # Fallback initialize_voxsigil_integration only
    try:
        from .voxsigil_integration import initialize_voxsigil_integration
    except ImportError:
        initialize_voxsigil_integration = None

# Import utility tools
from ARC.data_loader import ARCDataLoader
from tools.utilities.model_utils import ModelLoader
from tools.utilities.submission_utils import SubmissionFormatter

TOOLS_AVAILABLE = True
print("Tools successfully imported")

# Add the project root to the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
print(f"Added project root to path: {PROJECT_ROOT}")


class DynamicGridFormerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 VoxSigil Dynamic GridFormer Testing Suite v1.5-holo-alpha")
        self.root.geometry("1400x900")
        self.root.configure(bg="#1a1a2e")  # Initialize our utility components
        self.model_loader = ModelLoader()
        self.data_loader = ARCDataLoader()
        self.neural_interface = (
            NeuralInterface()
        )  # Using our improved NeuralInterface with mixed types support

        # Initialize inference engine (with fallback if GridFormerInference is None)
        if GridFormerInference is not None:
            self.inference_engine = GridFormerInference(
                self.model_loader, self.data_loader
            )
        else:
            # Create a placeholder inference engine
            class PlaceholderInference:
                def __init__(self, *args):
                    pass

                def run_inference(self, *args):
                    return None

                def set_model(self, *args):
                    pass

            self.inference_engine = PlaceholderInference()
        self.grid_visualizer = GridVisualizer()
        self.perf_visualizer = PerformanceVisualizer()
        self.submission_formatter = SubmissionFormatter()

        # Initialize BLT components if available
        if BLT_AVAILABLE:
            self.blt_encoder = BLTEncoder()
            print(
                f"BLT Encoder initialized with dimension: {self.blt_encoder.get_embedding_dimension()}"
            )
        else:
            self.blt_encoder = None

        # State variables
        self.discovered_models = {}
        self.current_model = None
        self.current_model_path = None
        self.test_data = None
        self.predictions = None  # Initialize GUI variables
        self.data_source = tk.StringVar(
            value="test"
        )  # Default to "test" or "training" as needed
        self.max_samples = tk.IntVar(value=100)  # Example default value
        self.confidence_threshold = tk.DoubleVar(value=0.5)  # Example default value

        # Initialize visualization components
        self.grid_figure = None
        self.grid_canvas = None  # Initialize VantaCore integration (with fallback)
        if VANTA_INTEGRATION_AVAILABLE and VantaGUIIntegration is not None:
            try:
                self.vanta_integration = VantaGUIIntegration(self)
                print("🚀 VantaCore integration initialized")
            except Exception as e:
                print(f"⚠️ VantaCore integration failed: {e}")
                self.vanta_integration = None
        else:
            self.vanta_integration = None
            print("⚠️ VantaCore integration not available - using fallback mode")

        # Initialize VoxSigil Supervisor integration
        try:
            if initialize_voxsigil_integration is not None:
                self.voxsigil_integration = initialize_voxsigil_integration(self)
                print("🧠 VoxSigil Supervisor integration initialized")
            else:
                self.voxsigil_integration = None
                print("⚠️ VoxSigil integration not available")
        except Exception as e:
            print(f"⚠️ VoxSigil integration failed: {e}")
            self.voxsigil_integration = None

        # Subscribe to supervisor status for bidirectional sync
        if self.voxsigil_integration:
            self.voxsigil_integration.add_status_callback(self._on_supervisor_status)

        # Apply VoxSigil styles
        VoxSigilStyles.apply_dark_theme(self.root)

        # Setup the GUI
        self._setup_gui()

        # Discover models
        self._discover_models()

    def _setup_gui(self):
        """Set up the main GUI components."""
        # Main frame that contains everything
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create the notebook (tabbed interface)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Model Tab - for model selection and information
        model_tab = ttk.Frame(self.notebook)
        self.notebook.add(model_tab, text="Models")
        self._setup_model_tab(model_tab)

        # Testing Tab - for running inference on test data
        testing_tab = ttk.Frame(self.notebook)
        self.notebook.add(testing_tab, text="Testing")
        self._setup_testing_tab(testing_tab)

        # Training Tab - for fine-tuning models
        training_tab = ttk.Frame(self.notebook)
        self.notebook.add(training_tab, text="Training")
        self._setup_training_tab(training_tab)

        # Visualization Tab - for detailed visualization of model outputs
        visualization_tab = ttk.Frame(self.notebook)
        self.notebook.add(visualization_tab, text="Visualization")
        self._setup_visualization_tab(visualization_tab)

        # Performance Tab - for model performance metrics
        performance_tab = ttk.Frame(self.notebook)
        self.notebook.add(performance_tab, text="Performance")
        self._setup_performance_tab(performance_tab)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _setup_model_tab(self, parent):
        """Set up the Models tab."""
        # Create a model discovery interface with VoxSigil integration
        self.model_interface = ModelDiscoveryInterface(
            parent, self.model_loader, self._on_model_selected
        )
        # Connect VoxSigil integration if available
        if hasattr(self, "voxsigil_integration") and self.voxsigil_integration:
            self.model_interface.voxsigil_integration = self.voxsigil_integration

    def _setup_testing_tab(self, parent):
        """Set up the Testing tab."""
        # Create a testing interface with VoxSigil integration
        self.testing_interface = VoxSigilTestingInterface(
            self,  # parent GUI instance
            parent,  # frame for testing UI
            self.data_loader,
            self.inference_engine,
            self.grid_visualizer,
            self._run_inference,
            self._save_predictions,
        )  # Connect VoxSigil integration if available
        if hasattr(self, "voxsigil_integration") and self.voxsigil_integration:
            self.testing_interface.voxsigil_integration = self.voxsigil_integration

    def _setup_training_tab(self, parent):
        """Set up the Training tab."""
        # Create a training interface with VoxSigil integration
        self.training_interface = VoxSigilTrainingInterface(
            self, self.notebook
        )  # Connect VoxSigil integration if available
        if hasattr(self, "voxsigil_integration") and self.voxsigil_integration:
            self.training_interface.voxsigil_integration = self.voxsigil_integration

    def _setup_visualization_tab(self, parent):
        """Set up the Visualization tab."""  # Create a visualization interface with VoxSigil integration
        self.visualization_interface = VoxSigilVisualizationInterface(
            self, self.notebook
        )
        # Connect VoxSigil integration if available
        if hasattr(self, "voxsigil_integration") and self.voxsigil_integration:
            self.visualization_interface.voxsigil_integration = (
                self.voxsigil_integration
            )

    def _setup_performance_tab(self, parent):
        """Set up the Performance tab."""
        # Create a performance interface with VoxSigil integration
        self.performance_interface = VoxSigilPerformanceInterface(
            self, parent, self.perf_visualizer, self._analyze_performance
        )
        # Connect VoxSigil integration if available
        if hasattr(self, "voxsigil_integration") and self.voxsigil_integration:
            self.performance_interface.voxsigil_integration = self.voxsigil_integration

    def _discover_models(self):
        """Discover available models."""
        self.update_status("Discovering models...")
        # Use the model loader to find models
        try:
            model_paths = self.model_loader.discover_models()
            self.discovered_models = {}
            # Include supervisor-registered models
            if self.voxsigil_integration:
                sup_models = self.voxsigil_integration.get_available_models()
                for m in sup_models:
                    self.discovered_models[m["name"]] = {"path": None, "metadata": m}
            # Add local models, overriding supervisor if same id
            for path, metadata in model_paths.items():
                model_id = Path(path).stem if isinstance(path, str) else str(path)
                self.discovered_models[model_id] = {"path": path, "metadata": metadata}
            # Update the model interface
            self.model_interface.update_model_list(self.discovered_models)
            self.update_status(
                f"Discovered {len(self.discovered_models)} models (local+supervisor)"
            )
        except Exception as e:
            self.show_error(f"Error discovering models: {str(e)}")
            self.update_status("Model discovery failed")

    def _on_model_selected(self, model_id):
        """Handle model selection."""
        if model_id in self.discovered_models:
            self.update_status(f"Loading model: {model_id}")
            try:
                # Load model in GUI
                model_path = self.discovered_models[model_id]["path"]
                self.current_model_path = model_path
                self.current_model = self.model_loader.load_model(model_path)

                # Sync with supervisor
                if self.voxsigil_integration:
                    ok = self.voxsigil_integration.load_model(model_id)
                    if ok:
                        self.update_status(f"Supervisor loaded model: {model_id}")
                    else:
                        self.show_error(f"Supervisor failed to load model: {model_id}")

                # Update model details in the interface
                metadata = self.discovered_models[model_id]["metadata"]
                self.model_interface.display_model_details(
                    model_id, metadata, self.current_model
                )

                # Update the inference engine
                self.inference_engine.set_model(self.current_model)

                self.update_status(f"Model loaded: {model_id}")
            except Exception as e:
                self.show_error(f"Error loading model: {str(e)}")
                self.update_status("Model loading failed")
        else:
            self.show_error(f"Model not found: {model_id}")

    def _run_inference(self, data, options=None):
        """Run inference on the provided data."""
        if self.current_model is None:
            self.show_error("No model loaded. Please select a model first.")
            return None

        self.update_status("Running inference...")
        try:
            # Run inference
            self.predictions = self.inference_engine.run_inference(
                data, self.current_model, options
            )
            self.update_status("Inference completed")

            # Sync event to supervisor
            if hasattr(self, "voxsigil_integration") and self.voxsigil_integration:
                try:
                    self.voxsigil_integration.store_interaction(
                        {
                            "event": "inference_run",
                            "model": Path(self.current_model_path).stem
                            if self.current_model_path
                            else None,
                            "data_sample_count": len(data)
                            if hasattr(data, "__len__")
                            else None,
                            "predictions": self.predictions,
                            "timestamp": time.time(),
                        }
                    )
                except Exception as e:
                    print(f"Warning: failed to send inference event to supervisor: {e}")

            # Update visualization
            self.visualization_interface.update_visualizations(data, self.predictions)

            return self.predictions
        except Exception as e:
            self.show_error(f"Inference error: {str(e)}")
            self.update_status("Inference failed")
            return None

    def _save_predictions(self, file_path=None):
        """Save predictions to file."""
        if self.predictions is None:
            self.show_error("No predictions to save.")
            return

        if file_path is None:
            # Ask user for file path
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            )

        if not file_path:
            return  # User cancelled

        self.update_status(f"Saving predictions to {file_path}")
        try:
            # Format predictions for submission if needed
            formatted = self.submission_formatter.format_predictions(self.predictions)

            # Save to file
            with open(file_path, "w") as f:
                json.dump(formatted, f, indent=2)

            self.update_status("Predictions saved successfully")
        except Exception as e:
            self.show_error(f"Error saving predictions: {str(e)}")
            self.update_status("Failed to save predictions")

    def _train_model(self, data, options):
        """Train or fine-tune the model."""
        if self.current_model is None:
            self.show_error("No model loaded. Please select a model first.")
            return

        self.update_status("Training model...")
        try:
            if hasattr(self.model_loader, "train_model"):
                self.model_loader.train_model(
                    self.current_model,
                    data,
                    **(options or {}),
                )
            else:
                print("ModelLoader has no train_model method; skipping training")
            self.update_status("Training completed")
        except Exception as e:
            self.show_error(f"Training error: {str(e)}")
            self.update_status("Training failed")

    def _save_model(self, file_path=None):
        """Save the current model to file."""
        if self.current_model is None:
            self.show_error("No model to save.")
            return

        if file_path is None:
            # Ask user for file path
            file_path = filedialog.asksaveasfilename(
                defaultextension=".pt",
                filetypes=[("PyTorch models", "*.pt"), ("All files", "*.*")],
            )

        if not file_path:
            return  # User cancelled

        self.update_status(f"Saving model to {file_path}")
        try:
            # Save the model
            self.model_loader.save_model(self.current_model, file_path)
            self.update_status("Model saved successfully")
        except Exception as e:
            self.show_error(f"Error saving model: {str(e)}")
            self.update_status("Failed to save model")

    def _analyze_performance(self, data=None, predictions=None):
        """Analyze model performance."""
        if data is None:
            data = self.test_data
        if predictions is None:
            predictions = self.predictions

        if data is None or predictions is None:
            self.show_error("No data and predictions available for analysis.")
            return

        self.update_status("Analyzing performance...")
        try:  # Calculate performance metrics
            metrics = self.perf_visualizer.calculate_metrics(data, predictions)

            # Update performance display
            self.performance_interface.update_metrics(metrics)

            self.update_status("Performance analysis completed")
            return metrics
        except Exception as e:
            self.show_error(f"Performance analysis error: {str(e)}")
            self.update_status("Performance analysis failed")
            return None

    def update_status(self, message):
        """Update the status bar message."""
        self.status_var.set(message)
        self.root.update_idletasks()

    def show_error(self, message):
        """Show an error message dialog."""
        messagebox.showerror("Error", message)

    def show_info(self, message):
        """Show an information message dialog."""
        messagebox.showinfo("Information", message)

    def _on_supervisor_status(self, info):
        """
        Enhanced bidirectional status handler for VoxSigil Supervisor communication.

        Handles various status types and command responses to ensure proper
        GUI ↔ Supervisor roundtrip synchronization.
        """
        try:
            if isinstance(info, dict):
                status_type = info.get("type", "status")

                if status_type == "command_response":
                    self._handle_command_response(info)
                elif status_type == "system_status":
                    self._handle_system_status(info)
                elif status_type == "training_update":
                    self._handle_training_update(info)
                elif status_type == "memory_sync":
                    self._handle_memory_sync(info)
                elif status_type == "agent_status":
                    self._handle_agent_status(info)
                else:
                    # Default status message handling
                    msg = info.get("message", str(info))
                    self.update_status(f"Supervisor → {msg}")
            else:
                # Simple string status
                self.update_status(f"Supervisor → {str(info)}")

        except Exception as e:
            print(f"Error handling supervisor status: {e}")
            self.update_status(f"Supervisor Status Error → {str(e)}")

    def _handle_command_response(self, info):
        """Handle command execution responses from supervisor"""
        command_id = info.get("command_id", "unknown")
        success = info.get("success", False)
        result = info.get("result", "No result")
        error = info.get("error")

        # Execute callback if registered
        if (
            hasattr(self, "_command_callbacks")
            and command_id in self._command_callbacks
        ):
            try:
                callback = self._command_callbacks[command_id]
                callback(success, result, error)
                del self._command_callbacks[command_id]
            except Exception as e:
                print(f"Error executing command callback: {e}")

        if success:
            self.update_status(f"✅ Command {command_id} completed: {result}")
        else:
            error_msg = error or "Unknown error"
            self.update_status(f"❌ Command {command_id} failed: {error_msg}")

    def _handle_system_status(self, info):
        """Handle system status updates"""
        component = info.get("component", "system")
        status = info.get("status", "unknown")
        health = info.get("health", "unknown")

        # Update GUI status indicators based on component health
        status_msg = f"{component}: {status} (health: {health})"
        self.update_status(f"🔧 System → {status_msg}")

    def _handle_training_update(self, info):
        """Handle training progress updates"""
        phase = info.get("phase", "unknown")
        progress = info.get("progress", 0)
        metrics = info.get("metrics", {})

        progress_msg = f"Training {phase}: {progress}%"
        if metrics:
            metrics_str = ", ".join(
                [
                    f"{k}:{v:.3f}" if isinstance(v, float) else f"{k}:{v}"
                    for k, v in list(metrics.items())[:3]
                ]
            )
            progress_msg += f" | {metrics_str}"

        self.update_status(f"🎯 Training → {progress_msg}")

    def _handle_memory_sync(self, info):
        """Handle memory synchronization updates"""
        operation = info.get("operation", "sync")
        count = info.get("count", 0)
        success = info.get("success", True)

        if success:
            self.update_status(f"🧠 Memory → {operation} completed ({count} items)")
        else:
            self.update_status(f"🧠 Memory → {operation} failed")

    def _handle_agent_status(self, info):
        """Handle agent status updates"""
        agent_id = info.get("agent_id", "unknown")
        status = info.get("status", "unknown")
        capabilities = info.get("capabilities", [])

        if capabilities:
            cap_str = ", ".join(capabilities[:3])  # Show first 3 capabilities
            self.update_status(f"🤖 Agent {agent_id}: {status} ({cap_str})")
        else:
            self.update_status(f"🤖 Agent {agent_id}: {status}")

    def send_supervisor_command(self, command_type, parameters=None, callback=None):
        """
        Send a command to the VoxSigil Supervisor and optionally handle response.

        Args:
            command_type (str): Type of command to execute
            parameters (dict): Command parameters
            callback (callable): Optional callback for command response

        Returns:
            str: Command ID for tracking
        """
        try:
            if not self.voxsigil_integration:
                self.update_status("❌ VoxSigil integration not available")
                return None

            import time

            command_id = f"gui_cmd_{int(time.time())}_{hash(command_type) % 10000}"

            command_data = {
                "command_id": command_id,
                "type": command_type,
                "parameters": parameters or {},
                "source": "dynamic_gridformer_gui",
                "timestamp": time.time(),
                "callback_requested": callback is not None,
            }

            # Store callback if provided
            if callback:
                if not hasattr(self, "_command_callbacks"):
                    self._command_callbacks = {}
                self._command_callbacks[command_id] = callback

            # Send command through VoxSigil integration
            success = self.voxsigil_integration.execute_supervisor_command(command_data)

            if success:
                self.update_status(
                    f"📤 Command sent: {command_type} (ID: {command_id})"
                )
                return command_id
            else:
                self.update_status(f"❌ Failed to send command: {command_type}")
                return None

        except Exception as e:
            self.update_status(f"❌ Error sending supervisor command: {e}")
            return None

    def request_supervisor_status(self):
        """Request comprehensive status from supervisor"""
        return self.send_supervisor_command(
            "get_status",
            {
                "include_agents": True,
                "include_memory": True,
                "include_training": True,
                "include_health": True,
            },
        )

    def sync_with_supervisor(self):
        """Perform bidirectional sync with supervisor"""
        self.update_status("🔄 Initiating supervisor sync...")

        # Send sync command with GUI state information
        gui_state = {
            "current_model": getattr(self, "current_model", None),
            "discovered_models_count": len(getattr(self, "discovered_models", {})),
            "active_tab": self.notebook.tab(self.notebook.select(), "text")
            if hasattr(self, "notebook")
            else "unknown",
            "vanta_integration_available": self.vanta_integration is not None,
            "gui_capabilities": [
                "model_discovery",
                "inference_execution",
                "training_interface",
                "visualization",
                "performance_monitoring",
            ],
        }

        return self.send_supervisor_command(
            "bidirectional_sync",
            {
                "gui_state": gui_state,
                "sync_type": "full",
                "request_agent_list": True,
                "request_memory_summary": True,
            },
        )

    def test_supervisor_roundtrip(self):
        """Test the bidirectional communication pipeline"""

        def test_callback(success, result, error):
            if success:
                self.update_status(f"✅ Roundtrip test successful: {result}")
            else:
                self.update_status(f"❌ Roundtrip test failed: {error}")

        self.update_status("🧪 Testing supervisor roundtrip...")
        return self.send_supervisor_command(
            "ping",
            {"test_data": "roundtrip_test", "timestamp": time.time()},
            callback=test_callback,
        )


def main():
    """Main entry point for the application."""
    root = tk.Tk()
    DynamicGridFormerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
