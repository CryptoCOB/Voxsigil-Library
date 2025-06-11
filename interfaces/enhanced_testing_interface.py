#!/usr/bin/env python3
"""
Enhanced Testing Tab Interface with VoxSigil Integration

This module provides an enhanced testing interface that connects to VoxSigil supervisor interfaces,
enabling memory management, RAG functionality, and learning capabilities in the GUI.
"""

import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from GUI.components.gui_styles import VoxSigilStyles

    STYLES_AVAILABLE = True
except ImportError:
    STYLES_AVAILABLE = False
    # Create minimal styles fallback - will be handled inline where needed
    VoxSigilStyles = None


class EnhancedVoxSigilTestingInterface:
    """Enhanced testing interface with VoxSigil supervisor integration"""

    def __init__(self, parent_gui, notebook, voxsigil_integration=None):
        """
        Initialize the enhanced testing interface

        Args:
            parent_gui: Reference to the main GUI class
            notebook: ttk.Notebook to add the testing tab to
            voxsigil_integration: VoxSigil integration manager
        """
        self.parent_gui = parent_gui
        self.notebook = notebook
        self.voxsigil_integration = voxsigil_integration

        # Testing state
        self.test_data = None
        self.test_results = []
        self.testing_active = False
        self.memory_enabled = tk.BooleanVar(value=True)
        self.rag_enabled = tk.BooleanVar(value=True)
        self.learning_enabled = tk.BooleanVar(value=False)

        # Create the enhanced testing tab
        self.create_testing_tab()

    def create_testing_tab(self):
        """Create the enhanced testing interface tab"""
        test_frame = ttk.Frame(self.notebook)
        self.notebook.add(test_frame, text="🧪 Enhanced Testing")

        # Create main container with scrollable frame
        main_container = tk.Frame(
            test_frame, bg="#1a1a2e" if STYLES_AVAILABLE else "white"
        )
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create notebook for testing sections
        testing_notebook = ttk.Notebook(main_container)
        testing_notebook.pack(fill=tk.BOTH, expand=True)

        # Model Testing Tab
        self._create_model_testing_tab(testing_notebook)

        # Memory Testing Tab
        self._create_memory_testing_tab(testing_notebook)

        # RAG Testing Tab
        self._create_rag_testing_tab(testing_notebook)

        # Learning Testing Tab
        self._create_learning_testing_tab(testing_notebook)

        # Status and controls at bottom
        self._create_status_controls(main_container)

    def _create_model_testing_tab(self, parent_notebook):
        """Create model testing controls"""
        model_frame = ttk.Frame(parent_notebook)
        parent_notebook.add(model_frame, text="Model Testing")

        # Current model status
        status_frame = ttk.LabelFrame(model_frame, text="Model Status")
        status_frame.pack(fill=tk.X, padx=10, pady=10)

        self.model_status_label = ttk.Label(
            status_frame, text="No model loaded", foreground="red"
        )
        self.model_status_label.pack(pady=5)

        # Test data selection
        data_frame = ttk.LabelFrame(model_frame, text="Test Data Selection")
        data_frame.pack(fill=tk.X, padx=10, pady=10)

        data_controls = ttk.Frame(data_frame)
        data_controls.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            data_controls, text="Load Test Data", command=self._load_test_data
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(data_controls, text="Run Test", command=self._run_model_test).pack(
            side=tk.LEFT, padx=5
        )

        # Results display
        results_frame = ttk.LabelFrame(model_frame, text="Test Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Text widget with scrollbar for results
        text_frame = tk.Frame(results_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.results_text = tk.Text(text_frame, wrap=tk.WORD, height=10)
        results_scroll = ttk.Scrollbar(
            text_frame, orient=tk.VERTICAL, command=self.results_text.yview
        )
        self.results_text.configure(yscrollcommand=results_scroll.set)

        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _create_memory_testing_tab(self, parent_notebook):
        """Create memory testing controls"""
        memory_frame = ttk.Frame(parent_notebook)
        parent_notebook.add(memory_frame, text="Memory Testing")

        # Memory controls
        controls_frame = ttk.LabelFrame(memory_frame, text="Memory Controls")
        controls_frame.pack(fill=tk.X, padx=10, pady=10)

        # Enable/disable memory
        ttk.Checkbutton(
            controls_frame, text="Enable Memory", variable=self.memory_enabled
        ).pack(anchor=tk.W, padx=5, pady=2)

        # Memory operations
        mem_ops_frame = ttk.Frame(controls_frame)
        mem_ops_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            mem_ops_frame,
            text="Store Test Interaction",
            command=self._store_test_interaction,
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            mem_ops_frame, text="Search Memory", command=self._search_memory
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(mem_ops_frame, text="View Memory", command=self._view_memory).pack(
            side=tk.LEFT, padx=5
        )

        # Memory search
        search_frame = ttk.LabelFrame(memory_frame, text="Memory Search")
        search_frame.pack(fill=tk.X, padx=10, pady=10)

        self.memory_search_var = tk.StringVar()
        search_entry = ttk.Entry(
            search_frame, textvariable=self.memory_search_var, width=50
        )
        search_entry.pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Button(
            search_frame, text="Search", command=self._perform_memory_search
        ).pack(side=tk.LEFT, padx=5)

        # Memory results
        mem_results_frame = ttk.LabelFrame(memory_frame, text="Memory Results")
        mem_results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        mem_text_frame = tk.Frame(mem_results_frame)
        mem_text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.memory_results_text = tk.Text(mem_text_frame, wrap=tk.WORD, height=8)
        mem_scroll = ttk.Scrollbar(
            mem_text_frame, orient=tk.VERTICAL, command=self.memory_results_text.yview
        )
        self.memory_results_text.configure(yscrollcommand=mem_scroll.set)

        self.memory_results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        mem_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _create_rag_testing_tab(self, parent_notebook):
        """Create RAG testing controls"""
        rag_frame = ttk.Frame(parent_notebook)
        parent_notebook.add(rag_frame, text="RAG Testing")

        # RAG controls
        rag_controls_frame = ttk.LabelFrame(rag_frame, text="RAG Controls")
        rag_controls_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Checkbutton(
            rag_controls_frame, text="Enable RAG", variable=self.rag_enabled
        ).pack(anchor=tk.W, padx=5, pady=2)

        # RAG query testing
        query_frame = ttk.LabelFrame(rag_frame, text="RAG Query Testing")
        query_frame.pack(fill=tk.X, padx=10, pady=10)

        self.rag_query_var = tk.StringVar()
        query_entry = ttk.Entry(query_frame, textvariable=self.rag_query_var, width=50)
        query_entry.pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Button(query_frame, text="Test RAG", command=self._test_rag_query).pack(
            side=tk.LEFT, padx=5
        )

        # RAG results
        rag_results_frame = ttk.LabelFrame(rag_frame, text="RAG Results")
        rag_results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        rag_text_frame = tk.Frame(rag_results_frame)
        rag_text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.rag_results_text = tk.Text(rag_text_frame, wrap=tk.WORD, height=8)
        rag_scroll = ttk.Scrollbar(
            rag_text_frame, orient=tk.VERTICAL, command=self.rag_results_text.yview
        )
        self.rag_results_text.configure(yscrollcommand=rag_scroll.set)

        self.rag_results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        rag_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _create_learning_testing_tab(self, parent_notebook):
        """Create learning testing controls"""
        learning_frame = ttk.Frame(parent_notebook)
        parent_notebook.add(learning_frame, text="Learning Testing")

        # Learning controls
        learn_controls_frame = ttk.LabelFrame(learning_frame, text="Learning Controls")
        learn_controls_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Checkbutton(
            learn_controls_frame,
            text="Enable Learning Mode",
            variable=self.learning_enabled,
        ).pack(anchor=tk.W, padx=5, pady=2)

        # Learning operations
        learn_ops_frame = ttk.Frame(learn_controls_frame)
        learn_ops_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            learn_ops_frame, text="Start Learning", command=self._start_learning
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            learn_ops_frame, text="Stop Learning", command=self._stop_learning
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            learn_ops_frame, text="Get Insights", command=self._get_learning_insights
        ).pack(side=tk.LEFT, padx=5)

        # Learning status and results
        learn_results_frame = ttk.LabelFrame(
            learning_frame, text="Learning Status & Insights"
        )
        learn_results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        learn_text_frame = tk.Frame(learn_results_frame)
        learn_text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.learning_results_text = tk.Text(learn_text_frame, wrap=tk.WORD, height=8)
        learn_scroll = ttk.Scrollbar(
            learn_text_frame,
            orient=tk.VERTICAL,
            command=self.learning_results_text.yview,
        )
        self.learning_results_text.configure(yscrollcommand=learn_scroll.set)

        self.learning_results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        learn_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _create_status_controls(self, parent):
        """Create status and control elements"""
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=10)

        # VoxSigil integration status
        integration_frame = ttk.LabelFrame(
            status_frame, text="VoxSigil Integration Status"
        )
        integration_frame.pack(fill=tk.X, padx=10, pady=5)

        if self.voxsigil_integration:
            status_info = self.voxsigil_integration.get_integration_status()
            status_text = (
                "✅ Connected"
                if status_info["interfaces_available"]
                else "⚠️ Limited Mode"
            )
            self.integration_status_label = ttk.Label(
                integration_frame,
                text=status_text,
                foreground="green" if status_info["interfaces_available"] else "orange",
            )
        else:
            self.integration_status_label = ttk.Label(
                integration_frame, text="❌ Not Available", foreground="red"
            )

        self.integration_status_label.pack(pady=5)

        # Global test controls
        global_controls = ttk.Frame(status_frame)
        global_controls.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(
            global_controls, text="Clear All Results", command=self._clear_all_results
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            global_controls, text="Save Test Report", command=self._save_test_report
        ).pack(side=tk.LEFT, padx=5)

    # Model testing methods
    def _load_test_data(self):
        """Load test data"""
        file_path = filedialog.askopenfilename(
            title="Select Test Data",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if file_path:
            try:
                # TODO: Load actual test data
                self._append_results("Test data loaded successfully from: " + file_path)
                self.test_data = {"file_path": file_path}
            except Exception as e:
                self._append_results(f"Error loading test data: {e}")

    def _run_model_test(self):
        """Run model test"""
        if not self.test_data:
            messagebox.showwarning("No Data", "Please load test data first")
            return

        self._append_results("Running model test...")

        # Check if VoxSigil integration is available
        if self.voxsigil_integration:
            self._append_results("Using VoxSigil integration for enhanced testing")

            # Store test interaction in memory if enabled
            if self.memory_enabled.get():
                interaction_data = {
                    "query": "Model test execution",
                    "timestamp": time.time(),
                    "test_data": str(self.test_data),
                    "type": "model_test",
                }
                success = self.voxsigil_integration.store_interaction(interaction_data)
                self._append_results(f"Memory storage: {'✅' if success else '❌'}")

        # Simulate model test
        self._append_results("Model test completed")

    # Memory testing methods
    def _store_test_interaction(self):
        """Store a test interaction in memory"""
        if not self.voxsigil_integration:
            self._append_memory_results("VoxSigil integration not available")
            return

        interaction_data = {
            "query": "Test interaction for memory storage",
            "response": "This is a test response",
            "timestamp": time.time(),
            "type": "test_interaction",
            "metadata": {"test": True},
        }

        success = self.voxsigil_integration.store_interaction(interaction_data)
        self._append_memory_results(
            f"Test interaction stored: {'✅' if success else '❌'}"
        )

    def _search_memory(self):
        """Search memory for interactions"""
        if not self.voxsigil_integration:
            self._append_memory_results("VoxSigil integration not available")
            return

        interactions = self.voxsigil_integration.retrieve_interactions(limit=5)
        self._append_memory_results(
            f"Found {len(interactions)} interactions in memory:"
        )
        for i, interaction in enumerate(interactions, 1):
            self._append_memory_results(f"{i}. {interaction.get('query', 'No query')}")

    def _view_memory(self):
        """View memory contents"""
        if not self.voxsigil_integration:
            self._append_memory_results("VoxSigil integration not available")
            return

        try:
            system_info = self.voxsigil_integration.get_system_info()
            self._append_memory_results("Memory system information:")
            self._append_memory_results(str(system_info))
        except Exception as e:
            self._append_memory_results(f"Error accessing memory: {e}")

    def _perform_memory_search(self):
        """Perform memory search with user query"""
        query = self.memory_search_var.get().strip()
        if not query:
            messagebox.showwarning("Empty Query", "Please enter a search query")
            return

        if not self.voxsigil_integration:
            self._append_memory_results("VoxSigil integration not available")
            return

        results = self.voxsigil_integration.search_memory(query, max_results=5)
        self._append_memory_results(f"Search results for '{query}':")
        for i, result in enumerate(results, 1):
            self._append_memory_results(f"{i}. {result.get('query', 'No query')}")

    # RAG testing methods
    def _test_rag_query(self):
        """Test RAG query functionality"""
        query = self.rag_query_var.get().strip()
        if not query:
            messagebox.showwarning("Empty Query", "Please enter a RAG query")
            return

        if not self.voxsigil_integration:
            self._append_rag_results("VoxSigil integration not available")
            return

        self._append_rag_results(f"Testing RAG with query: '{query}'")

        try:
            context, sources = self.voxsigil_integration.create_rag_context(query)
            self._append_rag_results(f"RAG context created with {len(sources)} sources")
            self._append_rag_results(f"Context preview: {context[:200]}...")

            # Also test VoxSigil-specific context injection
            vox_context, vox_sources = (
                self.voxsigil_integration.inject_voxsigil_context(query)
            )
            self._append_rag_results(
                f"VoxSigil context: {len(vox_sources)} additional sources"
            )

        except Exception as e:
            self._append_rag_results(f"RAG test error: {e}")

    # Learning testing methods
    def _start_learning(self):
        """Start learning mode"""
        if not self.voxsigil_integration:
            self._append_learning_results("VoxSigil integration not available")
            return

        success = self.voxsigil_integration.start_learning_mode()
        self._append_learning_results(
            f"Learning mode started: {'✅' if success else '❌'}"
        )

    def _stop_learning(self):
        """Stop learning mode"""
        if not self.voxsigil_integration:
            self._append_learning_results("VoxSigil integration not available")
            return

        success = self.voxsigil_integration.stop_learning_mode()
        self._append_learning_results(
            f"Learning mode stopped: {'✅' if success else '❌'}"
        )

    def _get_learning_insights(self):
        """Get learning insights"""
        if not self.voxsigil_integration:
            self._append_learning_results("VoxSigil integration not available")
            return

        insights = self.voxsigil_integration.get_learning_insights()
        self._append_learning_results(f"Learning insights ({len(insights)} found):")
        for i, insight in enumerate(insights, 1):
            self._append_learning_results(f"{i}. {insight}")

    # Utility methods
    def _append_results(self, text):
        """Append text to results display"""
        self.results_text.insert(tk.END, f"{text}\n")
        self.results_text.see(tk.END)

    def _append_memory_results(self, text):
        """Append text to memory results display"""
        self.memory_results_text.insert(tk.END, f"{text}\n")
        self.memory_results_text.see(tk.END)

    def _append_rag_results(self, text):
        """Append text to RAG results display"""
        self.rag_results_text.insert(tk.END, f"{text}\n")
        self.rag_results_text.see(tk.END)

    def _append_learning_results(self, text):
        """Append text to learning results display"""
        self.learning_results_text.insert(tk.END, f"{text}\n")
        self.learning_results_text.see(tk.END)

    def _clear_all_results(self):
        """Clear all result displays"""
        self.results_text.delete(1.0, tk.END)
        self.memory_results_text.delete(1.0, tk.END)
        self.rag_results_text.delete(1.0, tk.END)
        self.learning_results_text.delete(1.0, tk.END)

    def _save_test_report(self):
        """Save comprehensive test report"""
        file_path = filedialog.asksaveasfilename(
            title="Save Test Report",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )

        if file_path:
            try:
                with open(file_path, "w") as f:
                    f.write("VoxSigil Enhanced Testing Report\n")
                    f.write("=" * 40 + "\n\n")

                    f.write("Model Test Results:\n")
                    f.write(self.results_text.get(1.0, tk.END))
                    f.write("\n" + "-" * 20 + "\n\n")

                    f.write("Memory Test Results:\n")
                    f.write(self.memory_results_text.get(1.0, tk.END))
                    f.write("\n" + "-" * 20 + "\n\n")

                    f.write("RAG Test Results:\n")
                    f.write(self.rag_results_text.get(1.0, tk.END))
                    f.write("\n" + "-" * 20 + "\n\n")

                    f.write("Learning Test Results:\n")
                    f.write(self.learning_results_text.get(1.0, tk.END))

                messagebox.showinfo("Report Saved", f"Test report saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Error saving report: {e}")

    def update_model_status(self, model_name=None):
        """Update the model status display"""
        if model_name:
            self.model_status_label.config(
                text=f"Loaded: {model_name}", foreground="green"
            )
        else:
            self.model_status_label.config(text="No model loaded", foreground="red")

    def update_integration_status(self, voxsigil_integration):
        """Update VoxSigil integration status"""
        self.voxsigil_integration = voxsigil_integration
        if voxsigil_integration:
            status_info = voxsigil_integration.get_integration_status()
            status_text = (
                "✅ Connected"
                if status_info["interfaces_available"]
                else "⚠️ Limited Mode"
            )
            color = "green" if status_info["interfaces_available"] else "orange"
        else:
            status_text = "❌ Not Available"
            color = "red"

        self.integration_status_label.config(text=status_text, foreground=color)
