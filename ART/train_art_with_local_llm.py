#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train ART Module with Local LLM

This script demonstrates how to train the ART (Adaptive Resonance Theory) module
with a local LLM using the VoxSigil system. It shows the integration between
ART, BLT middleware, RAG systems, and a local LLM.

Usage:
    python train_art_with_local_llm.py

Requirements:
    - VoxSigil library with ART module
    - A local LLM (this example uses TinyLlama)
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Add paths to Python's module search path
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
# script_parent_dir = current_dir.parent # This would be Desktop
# voxsigil_library_actual_dir = script_parent_dir / "Voxsigil_Library" # This assumes Voxsigil_Library is a sibling of Voxsigil folder
# Corrected: Assume train_art_with_local_llm.py is in C:\\Users\\16479\\Desktop\\Voxsigil
# and Voxsigil_Library is in C:\\Users\\16479\\Desktop\\Voxsigil\\Voxsigil_Library
voxsigil_library_actual_dir = current_dir / "Voxsigil_Library"
# If Voxsigil_Library is a sibling to the folder containing this script (i.e. sibling to "Voxsigil" folder)
# then it should be:
# voxsigil_library_actual_dir = current_dir.parent / "Voxsigil_Library"

# Based on previous logs, Voxsigil_Library is at C:\\Users\\16479\\Desktop\\Voxsigil\\Voxsigil_Library
# and the script is at C:\\Users\\16479\\Desktop\\Voxsigil\\train_art_with_local_llm.py
# So, the script's directory *is* "C:\\Users\\16479\\Desktop\\Voxsigil"
# and "Voxsigil_Library" is a direct subdirectory of the script's parent, *if* the script is one level down.
# Let's re-evaluate based on the full path of the script.
# Script path: c:\\Users\\16479\\Desktop\\Voxsigil\\train_art_with_local_llm.py
# current_dir = Path("c:\\Users\\16479\\Desktop\\Voxsigil")
# Voxsigil_Library path: C:\\Users\\16479\\Desktop\\Voxsigil\\Voxsigil_Library
# So, voxsigil_library_actual_dir should be:
# voxsigil_library_actual_dir = current_dir / "Voxsigil_Library" # This seems incorrect.
# It should be:
voxsigil_library_root_dir = Path(
    os.path.abspath(__file__)
).parent.parent  # This should be C:\\Users\\16479\\Desktop
voxsigil_library_actual_dir = (
    voxsigil_library_root_dir / "Voxsigil" / "Voxsigil_Library"
)  # This is C:\\Users\\16479\\Desktop\\Voxsigil\\Voxsigil_Library

# The most robust way if the script is in '.../Voxsigil/' and the library is in '.../Voxsigil/Voxsigil_Library/':
# Assuming the script is in C:\\Users\\16479\\Desktop\\Voxsigil
# And the library is C:\\Users\\16479\\Desktop\\Voxsigil\\Voxsigil_Library
# The path to Voxsigil_Library needs to be added so that `from voxsigil_supervisor...` works.
# The `voxsigil_supervisor` package is inside `Voxsigil_Library`.
# So, `Voxsigil_Library` itself needs to be in `sys.path`.

# Path setup based on the file structure:
# C:/Users/16479/Desktop/Voxsigil/train_art_with_local_llm.py
# C:/Users/16479/Desktop/Voxsigil/Voxsigil_Library/
# The parent of Voxsigil_Library is C:/Users/16479/Desktop/Voxsigil

# script_dir = Path(__file__).resolve().parent # c:\\Users\\16479\\Desktop\\Voxsigil
# voxsigil_lib_dir = script_dir / "Voxsigil_Library" # c:\\Users\\16479\\Desktop\\Voxsigil\\Voxsigil_Library

# Let's use the user-provided path from the conversation summary:
# `c:\\Users\\16479\\Desktop\\Voxsigil\\Voxsigil_Library\\voxsigil_supervisor\\art\\art_trainer.py`
# This means `c:\\Users\\16479\\Desktop\\Voxsigil\\Voxsigil_Library` should be in sys.path

# Determine the absolute path to the 'Voxsigil_Library'
# The script is in 'C:\\Users\\16479\\Desktop\\Voxsigil'
# The library is in 'C:\\Users\\16479\\Desktop\\Voxsigil\\Voxsigil_Library'
path_to_voxsigil_folder = (
    Path(__file__).resolve().parent
)  # This is C:\\Users\\16479\\Desktop\\Voxsigil
voxsigil_library_path = path_to_voxsigil_folder / "Voxsigil_Library"

if str(voxsigil_library_path) not in sys.path:
    sys.path.insert(0, str(voxsigil_library_path))  # Add to the beginning of the path

# Import VoxSigil components
try:
    # Core ART components
    from ART.art_hybrid_blt_bridge import ARTHybridBLTBridge
    from ART.art_manager import ARTManager
    from ART.art_rag_bridge import ARTRAGBridge
    from ART.art_trainer import get_art_logger

    # Assuming tinyllama_assistant.py is at the root of Voxsigil_Library or a top-level package
    # If it's inside a sub-package of Voxsigil_Library, adjust the import accordingly.
    # e.g., from some_sub_package.tinyllama_assistant import TinyLlamaAssistant    from tinyllama_assistant import TinyLlamaAssistant
    # Assuming VoxSigilRag is a package at the root of Voxsigil_Library or a top-level package
    # If it's inside a sub-package of Voxsigil_Library, adjust the import accordingly.
    # e.g., from some_sub_package.VoxSigilRag.voxsigil_blt_rag import BLTEnhancedRAG
    from VoxSigilRag.voxsigil_blt_rag import BLTEnhancedRAG

    # Import TinyLlamaAssistant for the local LLM; provide a stub if it is absent.
    try:
        from tinyllama_assistant import TinyLlamaAssistant  # noqa: F401
    except ImportError:

        class TinyLlamaAssistant:
            """
            Fallback TinyLlamaAssistant used when the real implementation
            is unavailable. Generates a dummy response and logs a warning.
            """

            def __init__(self, *args, **kwargs):
                logging.getLogger("ART_LLM_Training").warning(
                    "tinyllama_assistant module not found; using stub TinyLlamaAssistant."
                )

            def generate_response(self, prompt: str) -> str:
                return "[TinyLlamaAssistant unavailable]"

    IMPORTS_SUCCESS = True
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure that VoxSigil, ART module, and local LLM are properly installed.")
    IMPORTS_SUCCESS = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("art_llm_training.log"), logging.StreamHandler()],
)
logger = get_art_logger("ART_LLM_Training")


class ARTLLMTrainer:
    """
    Trainer class that connects ART with a local LLM for training and pattern recognition.
    """

    def __init__(
        self,
        llm_model_path: Optional[str] = None,
        voxsigil_library_path: Optional[str] = None,
        batch_size: int = 8,
        entropy_threshold: float = 0.4,
        hybrid_weight: float = 0.7,
        use_rag: bool = True,
        use_hybrid_blt: bool = True,
        cache_dir: Optional[str] = None,
        logger_instance: Optional[logging.Logger] = None,
    ):
        """
        Initialize the ART LLM Trainer.

        Args:
            llm_model_path: Path to the local LLM model
            voxsigil_library_path: Path to the VoxSigil library
            batch_size: Batch size for training
            entropy_threshold: Entropy threshold for BLT
            hybrid_weight: Hybrid weight for BLT
            use_rag: Whether to use RAG for enhanced context
            use_hybrid_blt: Whether to use the hybrid BLT bridge
            cache_dir: Directory to store cache
            logger_instance: Logger instance
        """
        self.logger = logger_instance or logger
        self.batch_size = batch_size
        self.use_rag = use_rag
        self.use_hybrid_blt = use_hybrid_blt

        # Set default paths if not specified
        self.llm_model_path = llm_model_path
        self.voxsigil_library_path = voxsigil_library_path or str(voxsigil_library_actual_dir)

        # Initialize ART Manager
        self.logger.info("Initializing ART Manager...")
        self.art_manager = ARTManager(logger_instance=self.logger)

        # Initialize BLT bridge based on configuration
        if use_hybrid_blt:
            self.logger.info("Initializing ARTHybridBLTBridge...")
            self.blt_bridge = ARTHybridBLTBridge(
                art_manager=self.art_manager,
                entropy_threshold=entropy_threshold,
                blt_hybrid_weight=hybrid_weight,
                logger_instance=self.logger,
            )
        else:
            self.logger.info("Using standard ARTBLTBridge...")
            from blt.art_blt_bridge import ARTBLTBridge

            self.blt_bridge = ARTBLTBridge(
                art_manager=self.art_manager,
                entropy_threshold=entropy_threshold,
                logger_instance=self.logger,
            )

        # Initialize RAG if enabled
        self.rag_bridge = None
        if use_rag:
            try:
                self.logger.info("Initializing BLTEnhancedRAG...")
                self.rag = BLTEnhancedRAG(
                    voxsigil_library_path=Path(self.voxsigil_library_path),
                    entropy_threshold=entropy_threshold,
                    blt_hybrid_weight=hybrid_weight,
                )

                self.logger.info("Initializing ARTRAGBridge...")
                self.rag_bridge = ARTRAGBridge(
                    art_manager=self.art_manager,
                    rag_system=self.rag,
                    logger_instance=self.logger,
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize RAG components: {e}")
                self.use_rag = False

        # Initialize LLM
        self.llm = None
        try:
            self.logger.info("Initializing local LLM...")
            self.llm = TinyLlamaAssistant(model_path=self.llm_model_path, cache_dir=cache_dir)
            self.logger.info("Local LLM initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize local LLM: {e}")
            raise

        # Training stats
        self.stats = {
            "total_batches_processed": 0,
            "total_examples_processed": 0,
            "high_entropy_examples": 0,
            "novel_patterns_detected": 0,
            "start_time": time.time(),
        }

        self.logger.info("ARTLLMTrainer initialized successfully")

    def generate_llm_responses(self, prompts: List[str]) -> List[str]:
        """
        Generate responses from the LLM for a batch of prompts.

        Args:
            prompts: List of prompt strings

        Returns:
            List of response strings from the LLM
        """
        if not self.llm:
            self.logger.error("LLM is not initialized")
            return [""] * len(prompts)

        responses = []
        for prompt in prompts:
            try:
                response = self.llm.generate_response(prompt)
                responses.append(response)
            except Exception as e:
                self.logger.error(f"Error generating LLM response: {e}")
                responses.append("")

        return responses

    def enhance_with_rag(self, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Enhance a query with RAG-retrieved context.

        Args:
            query: The query string

        Returns:
            Tuple of (enhanced_context, retrieved_documents)
        """
        if not self.use_rag or not self.rag_bridge:
            return query, []

        try:
            # Use the ARTRAGBridge to get enhanced context
            context, docs = self.rag_bridge.create_enhanced_context(
                query, num_results=5, rerank_with_art=True
            )
            return context, docs
        except Exception as e:
            self.logger.error(f"Error enhancing query with RAG: {e}")
            return query, []

    def process_with_blt(self, input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process input data with the BLT bridge.

        Args:
            input_data: The input data to process

        Returns:
            Analysis results
        """
        try:
            # Process through BLT bridge to determine if ART analysis is needed
            result = self.blt_bridge.process_input(input_data)
            return result
        except Exception as e:
            self.logger.error(f"Error processing input with BLT: {e}")
            return {"input_processed": False, "error": str(e)}

    def extract_training_pairs(self, raw_data: List[str]) -> List[Tuple[str, str]]:
        """
        Extract query-response pairs for training.
        This could be from a corpus, conversation log, etc.

        Args:
            raw_data: Raw data strings to process

        Returns:
            List of (query, response) tuples
        """
        pairs = []
        for data in raw_data:
            # Simple heuristic to split data into query-response pairs
            # In a real system, this would be more sophisticated
            if "\n" in data:
                parts = data.split("\n", 1)
                query = parts[0].strip()
                response = parts[1].strip() if len(parts) > 1 else ""
                pairs.append((query, response))
            else:
                # If no clear separation, use the data as a query and generate a response
                query = data.strip()
                response = self.generate_llm_responses([query])[0]
                pairs.append((query, response))

        return pairs

    def train_batch(self, batch: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Train ART on a batch of query-response pairs.

        Args:
            batch: List of (query, response) tuples

        Returns:
            Training statistics
        """
        if not batch:
            return {"error": "Empty batch provided"}

        self.logger.info(f"Training on batch of {len(batch)} examples")
        self.stats["total_batches_processed"] += 1
        self.stats["total_examples_processed"] += len(batch)

        # Use BLT to determine which examples to train on
        filtered_batch = []
        for query, response in batch:
            # Process the query with BLT to check entropy
            blt_result = self.process_with_blt(query)

            if blt_result.get("analysis_performed", False):
                # If BLT determined this should be analyzed, add to training batch
                filtered_batch.append((query, response))
                self.stats["high_entropy_examples"] += 1

        # If selective processing filtered out all examples, use the full batch
        training_batch = filtered_batch if filtered_batch else batch

        # Train ART on the selected batch
        try:
            # First check if we can enhance with RAG
            enhanced_batch = []
            for query, response in training_batch:
                enhanced_context, _ = self.enhance_with_rag(query)
                enhanced_batch.append((enhanced_context, response))

            # Train ART using the enhanced batch
            result = self.art_manager.train_on_batch(enhanced_batch)

            # Update stats
            novel_categories = result.get("novel_categories", 0)
            self.stats["novel_patterns_detected"] += novel_categories

            return {
                "batch_size": len(batch),
                "filtered_batch_size": len(training_batch),
                "novel_categories": novel_categories,
                "training_time": result.get("training_time", 0),
            }

        except Exception as e:
            self.logger.error(f"Error training ART on batch: {e}")
            return {"error": str(e)}

    def generate_responses(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Generate enhanced responses using ART, BLT, RAG, and LLM.

        Args:
            queries: List of query strings

        Returns:
            List of response dictionaries with metadata
        """
        results = []

        for query in queries:
            result = {
                "query": query,
                "art_analysis": None,
                "rag_enhanced": False,
                "llm_response": None,
                "art_novel_pattern": False,
            }

            # Step 1: Process with BLT to determine if ART analysis is needed
            blt_result = self.process_with_blt(query)

            # Step 2: If BLT indicated high entropy, perform ART analysis
            if blt_result.get("analysis_performed", False):
                art_result = blt_result.get("art_result", {})
                result["art_analysis"] = {
                    "category_id": art_result.get("category", {}).get("id", "unknown"),
                    "is_novel": art_result.get("is_novel_category", False),
                    "confidence": art_result.get("confidence", 0.0),
                }
                result["art_novel_pattern"] = art_result.get("is_novel_category", False)

            # Step 3: Enhance with RAG if available
            enhanced_query = query
            if self.use_rag and self.rag_bridge:
                enhanced_query, docs = self.enhance_with_rag(query)
                result["rag_enhanced"] = bool(docs)
                result["rag_docs"] = [
                    doc.get("id", "unknown") for doc in docs[:3]
                ]  # Include top 3 doc IDs

            # Step 4: Generate LLM response
            llm_response = self.generate_llm_responses([enhanced_query])[0]
            result["llm_response"] = llm_response

            results.append(result)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get training statistics.

        Returns:
            Dictionary of training statistics
        """
        stats = self.stats.copy()

        # Add derived statistics
        elapsed_time = time.time() - stats["start_time"]
        stats["elapsed_time"] = elapsed_time

        if stats["total_examples_processed"] > 0:
            stats["high_entropy_ratio"] = (
                stats["high_entropy_examples"] / stats["total_examples_processed"]
            )
            stats["examples_per_second"] = stats["total_examples_processed"] / max(elapsed_time, 1)
        else:
            stats["high_entropy_ratio"] = 0
            stats["examples_per_second"] = 0

        # Add ART stats
        if hasattr(self.art_manager, "status"):
            art_stats = self.art_manager.status()
            stats["art_stats"] = art_stats

        return stats

    def save_state(self, path: str) -> bool:
        """
        Save the current state of ART and training statistics.

        Args:
            path: Path to save the state

        Returns:
            Success flag
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Save ART state
            art_path = f"{path}_art_state.json"
            self.art_manager.save_state(art_path)

            # Save training stats
            stats_path = f"{path}_stats.json"
            with open(stats_path, "w") as f:
                json.dump(self.get_stats(), f, indent=2)

            self.logger.info(f"Saved ART state to {art_path} and stats to {stats_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving state: {e}")
            return False

    def load_state(self, path: str) -> bool:
        """
        Load a previously saved state.

        Args:
            path: Path to load the state from

        Returns:
            Success flag
        """
        try:
            # Load ART state
            art_path = f"{path}_art_state.json"
            self.art_manager.load_state(art_path)

            # Load training stats
            stats_path = f"{path}_stats.json"
            if os.path.exists(stats_path):
                with open(stats_path, "r") as f:
                    loaded_stats = json.load(f)
                    # Update only the fields that exist in self.stats
                    for key in self.stats:
                        if key in loaded_stats:
                            self.stats[key] = loaded_stats[key]

            self.logger.info(f"Loaded ART state from {art_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading state: {e}")
            return False


def load_sample_data(
    file_path: Optional[str] = None, generate_if_missing: bool = True
) -> List[str]:
    """
    Load sample data for training.

    Args:
        file_path: Path to sample data file
        generate_if_missing: Whether to generate sample data if file is missing

    Returns:
        List of data strings
    """
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = [line.strip() for line in f.readlines() if line.strip()]
            return data
        except Exception as e:
            print(f"Error loading sample data: {e}")

    if generate_if_missing:
        print("Generating sample training data...")

        # Example prompts covering different types of content
        sample_prompts = [
            "What is the capital of France?",
            "Explain how neural networks work",
            "Write a short poem about the moon",
            "What are the main differences between Python and JavaScript?",
            "Create a recipe for chocolate chip cookies",
            "What is artificial general intelligence?",
            "Tell me about the history of Rome",
            "How do I solve a quadratic equation?",
            "What are the ethical implications of AI?",
            "Explain the water cycle in simple terms",
            "What were the causes of World War II?",
            "How does a nuclear reactor work?",
            "Write a short story about a robot discovering emotions",
            "What is climate change and how does it affect ecosystems?",
            "Compare and contrast classical and operant conditioning",
            "How do vaccines work?",
            "What are the key principles of object-oriented programming?",
            "Explain the theory of relativity in simple terms",
            "What is machine learning and how does it differ from traditional programming?",
            "How do humans dream and why do we need sleep?",
        ]

        # Generate more variations to increase the dataset
        more_samples = []

        for prompt in sample_prompts:
            # Add "Can you..." variations
            more_samples.append(f"Can you {prompt.lower()}")

            # Add "I need..." variations
            if prompt.startswith("What"):
                more_samples.append(prompt.replace("What", "I need to know what"))
            elif prompt.startswith("How"):
                more_samples.append(prompt.replace("How", "I need to understand how"))
            elif prompt.startswith("Explain"):
                more_samples.append(prompt.replace("Explain", "I need an explanation of"))

        sample_prompts.extend(more_samples)
        return sample_prompts

    return []


def main():
    """Main function to run the ART LLM training script."""
    parser = argparse.ArgumentParser(description="Train ART with a local LLM")
    parser.add_argument("--model-path", type=str, help="Path to the local LLM model")
    parser.add_argument("--voxsigil-path", type=str, help="Path to the VoxSigil library")
    parser.add_argument("--data-path", type=str, help="Path to training data file")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--entropy-threshold", type=float, default=0.4, help="Entropy threshold")
    parser.add_argument("--hybrid-weight", type=float, default=0.7, help="Hybrid weight")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG")
    parser.add_argument(
        "--no-hybrid-blt",
        action="store_true",
        help="Use standard BLT instead of hybrid",
    )
    parser.add_argument("--save-path", type=str, default="art_llm_model", help="Path to save model")
    parser.add_argument("--load-path", type=str, help="Path to load model from")
    parser.add_argument("--num-epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    args = parser.parse_args()

    if not IMPORTS_SUCCESS:
        print("Failed to import required modules. Exiting.")
        return

    try:
        # Initialize the trainer
        trainer = ARTLLMTrainer(
            llm_model_path=args.model_path,
            voxsigil_library_path=args.voxsigil_path,
            batch_size=args.batch_size,
            entropy_threshold=args.entropy_threshold,
            hybrid_weight=args.hybrid_weight,
            use_rag=not args.no_rag,
            use_hybrid_blt=not args.no_hybrid_blt,
        )

        # Load state if specified
        if args.load_path:
            trainer.load_state(args.load_path)

        # Load training data
        data = load_sample_data(args.data_path)
        print(f"Loaded {len(data)} training examples")

        if not data:
            print("No training data available. Exiting.")
            return

        # Extract training pairs
        pairs = trainer.extract_training_pairs(data)
        print(f"Extracted {len(pairs)} query-response pairs")

        # Training loop
        if not args.interactive:
            print(f"Starting training for {args.num_epochs} epochs...")
            for epoch in range(args.num_epochs):
                print(f"\nEpoch {epoch + 1}/{args.num_epochs}")

                # Shuffle data for each epoch
                random.shuffle(pairs)

                # Process in batches
                for i in range(0, len(pairs), args.batch_size):
                    batch = pairs[i : i + args.batch_size]
                    print(
                        f"Training batch {i // args.batch_size + 1}/{(len(pairs) - 1) // args.batch_size + 1}",
                        end="",
                    )

                    # Train on batch
                    result = trainer.train_batch(batch)

                    if "error" in result:
                        print(f" - Error: {result['error']}")
                    else:
                        print(
                            f" - {result['filtered_batch_size']}/{result['batch_size']} examples, "
                            f"{result['novel_categories']} novel patterns"
                        )

                # Print stats after each epoch
                stats = trainer.get_stats()
                print(f"\nStats after epoch {epoch + 1}:")
                print(f"  Processed {stats['total_examples_processed']} examples")
                print(f"  {stats['novel_patterns_detected']} novel patterns detected")
                print(
                    f"  {stats['high_entropy_examples']} high entropy examples ({stats['high_entropy_ratio']:.2%})"
                )
                print(f"  Processing speed: {stats['examples_per_second']:.2f} examples/sec")

                # Save state after each epoch
                if args.save_path:
                    save_path = f"{args.save_path}_epoch{epoch + 1}"
                    trainer.save_state(save_path)

            # Final save
            if args.save_path:
                trainer.save_state(args.save_path)

            print("\nTraining complete!")

        # Interactive mode
        else:
            print("\nEntering interactive mode. Type 'exit' to quit, 'stats' for statistics.")
            while True:
                query = input("\nEnter a query: ")

                if query.lower() == "exit":
                    break

                if query.lower() == "stats":
                    stats = trainer.get_stats()
                    print("\nCurrent Statistics:")
                    for key, value in stats.items():
                        if key != "art_stats" and not isinstance(value, dict):
                            print(f"  {key}: {value}")
                    continue

                if query.lower().startswith("save:"):
                    save_path = query.split(":", 1)[1].strip()
                    if save_path:
                        if trainer.save_state(save_path):
                            print(f"State saved to {save_path}")
                        else:
                            print("Failed to save state")
                    else:
                        print("Please provide a save path")
                    continue

                # Process the query
                results = trainer.generate_responses([query])
                if not results:
                    print("No response generated")
                    continue

                result = results[0]

                # Print the response with metadata
                print("\nART Analysis:")
                if result.get("art_analysis"):
                    art = result["art_analysis"]
                    print(f"  Category: {art.get('category_id', 'unknown')}")
                    print(f"  Novel Pattern: {art.get('is_novel', False)}")
                    print(f"  Confidence: {art.get('confidence', 0.0):.2f}")
                else:
                    print("  No ART analysis performed (low entropy input)")

                if result.get("rag_enhanced"):
                    print("\nRAG Enhancement:")
                    print(f"  Used documents: {', '.join(result.get('rag_docs', ['none']))}")

                print("\nResponse:")
                print(result.get("llm_response", "No response generated"))

                # Ask if we should train on this example
                train = input("\nTrain on this example? (y/n): ").lower()
                if train == "y":
                    response = result.get("llm_response", "")
                    trainer.train_batch([(query, response)])
                    print("Trained on example")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error in training process: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
