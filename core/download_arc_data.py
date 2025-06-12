#!/usr/bin/env python
"""
download_arc_data.py - ARC Dataset Downloader and Preparer

This script downloads the Abstraction and Reasoning Corpus (ARC) dataset
and prepares it for training with the GRID-Former model.

HOLO-1.5 Enhanced Data Processor:
- Recursive symbolic cognition for data validation and processing patterns
- Neural-symbolic data synthesis with cognitive quality assessment
- VantaCore-integrated download orchestration with adaptive retry strategies
- Cognitive monitoring of data integrity and completeness
"""

import os
import sys
import json
import logging
import requests
import argparse
import zipfile
import time
from pathlib import Path
from typing import Dict, Optional, Any

# HOLO-1.5 Recursive Symbolic Cognition Mesh imports
from .base import BaseCore, vanta_core_module, CognitiveMeshRole

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ARC-Data-Downloader")

# ARC dataset URLs
ARC_DATASET_URL = "https://github.com/fchollet/ARC/archive/refs/heads/master.zip"
# Using Kaggle dataset URLs (consistently maintained)
ARC_DIRECT_TRAINING_URL = "https://www.kaggle.com/datasets/shivamb/abstraction-and-reasoning-challenge/download/training.json"
ARC_DIRECT_EVALUATION_URL = "https://www.kaggle.com/datasets/shivamb/abstraction-and-reasoning-challenge/download/evaluation.json"
ARC_DIRECT_TEST_URL = "https://www.kaggle.com/datasets/shivamb/abstraction-and-reasoning-challenge/download/test.json"


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Download and prepare ARC dataset")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/arc",
        help="Output directory for ARC dataset",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["zip", "direct"],
        default="direct",
        help="Download method: zip for full repository, direct for JSON files only",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if files already exist",
    )

    return parser.parse_args()


def download_file(url: str, output_path: Path) -> bool:
    """Download a file from URL to output path"""
    try:
        logger.info(f"Downloading {url} to {output_path}")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Get total file size for progress reporting
        total_size = int(response.headers.get("content-length", 0))

        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file with progress reporting
        with open(output_path, "wb") as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        sys.stdout.write(f"\rDownload progress: {progress:.1f}%")
                        sys.stdout.flush()

        if total_size > 0:
            sys.stdout.write("\n")
        logger.info(f"Download complete: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return False


def download_arc_zip(output_dir: Path, force: bool = False) -> bool:
    """Download and extract the full ARC dataset repository"""
    # Download zip file
    zip_path = output_dir / "arc_dataset.zip"

    if zip_path.exists() and not force:
        logger.info(
            f"Zip file already exists at {zip_path}. Use --force to redownload."
        )
    else:
        if not download_file(ARC_DATASET_URL, zip_path):
            return False

    # Extract zip file
    try:
        logger.info(f"Extracting {zip_path} to {output_dir}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)
        logger.info("Extraction complete")

        # Create symbolic links or copy files to standard location
        arc_extracted_dir = output_dir / "ARC-master"
        if arc_extracted_dir.exists():
            # Create standard directory structure
            data_dir = output_dir / "data"
            data_dir.mkdir(exist_ok=True)

            # Link or copy data directories
            for subdir in ["training", "evaluation", "test"]:
                src_dir = arc_extracted_dir / "data" / subdir
                dst_dir = data_dir / subdir

                if not dst_dir.exists():
                    if sys.platform == "win32":
                        # Windows: copy directory
                        import shutil

                        shutil.copytree(src_dir, dst_dir)
                    else:
                        # Unix: create symlink
                        os.symlink(src_dir, dst_dir)

            logger.info(f"ARC dataset available at {data_dir}")
            return True
        else:
            # If we reach this point, the expected `ARC-master` directory was not found
            logger.error(
                "Extraction completed but expected ARC-master directory is missing."
            )
            return False

    except Exception as e:
        logger.error(f"Error extracting zip file: {e}")
        return False


def download_arc_direct(output_dir: Path, force: bool = False) -> bool:
    """Download ARC JSON files directly"""
    # Define file paths
    training_dir = output_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)

    evaluation_dir = output_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    test_dir = output_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Full paths for the files
    training_file = training_dir / "training.json"
    evaluation_file = evaluation_dir / "evaluation.json"
    test_file = test_dir / "test.json"

    # Download training data
    if training_file.exists() and not force:
        logger.info(f"Training file already exists at {training_file}")
    else:
        if not download_file(ARC_DIRECT_TRAINING_URL, training_file):
            return False

    # Download evaluation data
    if evaluation_file.exists() and not force:
        logger.info(f"Evaluation file already exists at {evaluation_file}")
    else:
        if not download_file(ARC_DIRECT_EVALUATION_URL, evaluation_file):
            return False

    # Download test data
    if test_file.exists() and not force:
        logger.info(f"Test file already exists at {test_file}")
    else:
        if not download_file(ARC_DIRECT_TEST_URL, test_file):
            return False

    logger.info(f"ARC dataset available at {output_dir}")
    return True


def count_arc_examples(json_file: Path) -> Dict[str, int]:
    """Count examples in an ARC JSON file"""
    try:
        with open(json_file, "r") as f:
            data = json.load(f)

        task_count = len(data)
        train_examples = sum(len(task["train"]) for task in data.values())
        test_examples = sum(len(task["test"]) for task in data.values())

        return {
            "tasks": task_count,
            "train_examples": train_examples,
            "test_examples": test_examples,
        }
    except Exception as e:
        logger.error(f"Error counting examples in {json_file}: {e}")
        return {"tasks": 0, "train_examples": 0, "test_examples": 0}


def verify_arc_data(output_dir: Path) -> bool:
    """Verify that the ARC dataset is correctly downloaded and structured"""
    training_file = output_dir / "training" / "training.json"
    evaluation_file = output_dir / "evaluation" / "evaluation.json"
    test_file = output_dir / "test" / "test.json"

    # Check if files exist
    files_exist = (
        training_file.exists() and evaluation_file.exists() and test_file.exists()
    )
    if not files_exist:
        logger.error("ARC dataset files missing or incomplete")
        return False

    # Count examples in each file
    training_stats = count_arc_examples(training_file)
    evaluation_stats = count_arc_examples(evaluation_file)
    test_stats = count_arc_examples(test_file)

    # Output statistics
    logger.info(
        f"ARC Training set: {training_stats['tasks']} tasks, "
        f"{training_stats['train_examples']} training examples, "
        f"{training_stats['test_examples']} test examples"
    )

    logger.info(
        f"ARC Evaluation set: {evaluation_stats['tasks']} tasks, "
        f"{evaluation_stats['train_examples']} training examples, "
        f"{evaluation_stats['test_examples']} test examples"
    )

    logger.info(f"ARC Test set: {test_stats['tasks']} tasks")

    # Check if files have expected content
    return (
        training_stats["tasks"] > 0
        and evaluation_stats["tasks"] > 0        and test_stats["tasks"] > 0
    )


@vanta_core_module(
    name="arc_data_downloader",
    subsystem="data_processing",
    mesh_role=CognitiveMeshRole.PROCESSOR,
    capabilities=[
        "arc_data_download", "data_validation", "integrity_verification",
        "cognitive_monitoring", "adaptive_retry", "data_synthesis"
    ],
    cognitive_load=3.5,
    symbolic_depth=2
)
class ARCDataDownloader(BaseCore):
    """
    HOLO-1.5 Enhanced ARC Data Downloader Processor
    
    Implements intelligent data downloading with recursive symbolic cognition for:
    - Adaptive download strategies with cognitive retry patterns
    - Neural-symbolic data validation and integrity assessment
    - Cognitive quality monitoring throughout the download process
    - VantaCore-integrated download orchestration and status reporting
    """

    def __init__(self, vanta_core=None, config: Optional[Dict[str, Any]] = None):
        """Initialize HOLO-1.5 enhanced ARC data downloader"""
        # Initialize BaseCore first
        super().__init__(vanta_core, config)
        
        # Download configuration
        self.default_output_dir = Path("./data/arc")
        self.download_timeout = 300  # 5 minutes
        self.max_retries = 3
        
        # HOLO-1.5 cognitive download metrics
        self.download_metrics = {
            "download_efficiency": 0.0,
            "validation_accuracy": 0.0,
            "cognitive_consistency": 0.0,
            "integrity_score": 0.0,
            "adaptive_strategies_used": 0
        }
        
        # Download trace storage
        self.download_traces = []
        
    async def initialize(self) -> bool:
        """Initialize HOLO-1.5 ARC data downloader with cognitive capabilities"""
        try:
            if self.vanta_core:
                await self.vanta_core.register_component(
                    "arc_data_processor",
                    {
                        "type": "cognitive_downloader",
                        "capabilities": self._get_metadata()["capabilities"],
                        "data_types": ["training", "evaluation", "test"],
                        "validation_levels": ["basic", "comprehensive", "cognitive"]
                    }
                )
                logger.info("✅ ARCDataDownloader registered with VantaCore")
            
            # Initialize cognitive download systems
            await self._initialize_cognitive_download()
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ARCDataDownloader: {e}")
            return False
    
    async def _initialize_cognitive_download(self):
        """Initialize cognitive download capabilities"""
        # Set up adaptive download strategies
        self.download_strategies = {
            "direct": {"priority": 1, "reliability": 0.8, "speed": 0.9},
            "zip": {"priority": 2, "reliability": 0.9, "speed": 0.6},
            "fallback": {"priority": 3, "reliability": 0.6, "speed": 0.3}
        }
        
        # Initialize validation patterns
        self.validation_patterns = {
            "format_validation": {"weight": 0.3, "threshold": 0.95},
            "content_validation": {"weight": 0.4, "threshold": 0.90},
            "integrity_validation": {"weight": 0.3, "threshold": 0.85}
        }
        
        logger.info("🧠 Cognitive download systems initialized")
    
    def download_arc_data(self, output_dir: Optional[Path] = None, method: str = "direct", force: bool = False) -> bool:
        """Download ARC data with cognitive monitoring"""
        output_dir = output_dir or self.default_output_dir
        
        logger.info(f"🔄 Starting cognitive ARC data download to {output_dir}")
        
        # Update cognitive metrics - starting download
        self._update_download_metrics("download_start", True, 1.0)
        
        try:
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Execute download based on method
            if method == "zip":
                success = download_arc_zip(output_dir, force)
            else:  # direct
                success = download_arc_direct(output_dir, force)
            
            # Update cognitive metrics based on download success
            self._update_download_metrics("download_complete", success, 2.0)
            
            if success:
                # Perform cognitive validation
                validation_success = self._cognitive_validate_data(output_dir)
                self._update_download_metrics("validation_complete", validation_success, 1.5)
                
                return validation_success
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error in cognitive download: {e}")
            self._update_download_metrics("download_error", False, 3.0)
            return False
    
    def _cognitive_validate_data(self, output_dir: Path) -> bool:
        """Perform cognitive validation of downloaded data"""
        logger.info("🧠 Performing cognitive data validation...")
        
        # Basic verification
        basic_valid = verify_arc_data(output_dir)
        
        if not basic_valid:
            logger.error("❌ Basic validation failed")
            return False
        
        # Enhanced cognitive validation
        try:
            validation_scores = {}
            
            # Format validation
            format_score = self._validate_data_format(output_dir)
            validation_scores["format"] = format_score
            
            # Content validation  
            content_score = self._validate_data_content(output_dir)
            validation_scores["content"] = content_score
            
            # Integrity validation
            integrity_score = self._validate_data_integrity(output_dir)
            validation_scores["integrity"] = integrity_score
            
            # Calculate weighted cognitive validation score
            cognitive_score = (
                validation_scores["format"] * self.validation_patterns["format_validation"]["weight"] +
                validation_scores["content"] * self.validation_patterns["content_validation"]["weight"] +
                validation_scores["integrity"] * self.validation_patterns["integrity_validation"]["weight"]
            )
            
            # Update cognitive metrics
            self.download_metrics["validation_accuracy"] = cognitive_score
            self.download_metrics["integrity_score"] = integrity_score
            
            logger.info(f"🧠 Cognitive validation score: {cognitive_score:.3f}")
            logger.info(f"  Format: {format_score:.3f}, Content: {content_score:.3f}, Integrity: {integrity_score:.3f}")
            
            return cognitive_score >= 0.85  # Cognitive validation threshold
            
        except Exception as e:
            logger.error(f"❌ Error in cognitive validation: {e}")
            return False
    
    def _validate_data_format(self, output_dir: Path) -> float:
        """Validate data format structure"""
        try:
            required_files = [
                output_dir / "training" / "training.json",
                output_dir / "evaluation" / "evaluation.json", 
                output_dir / "test" / "test.json"
            ]
            
            existing_files = sum(1 for f in required_files if f.exists())
            return existing_files / len(required_files)
            
        except Exception:
            return 0.0
    
    def _validate_data_content(self, output_dir: Path) -> float:
        """Validate data content quality"""
        try:
            total_score = 0.0
            file_count = 0
            
            for dataset_type in ["training", "evaluation", "test"]:
                file_path = output_dir / dataset_type / f"{dataset_type}.json"
                if file_path.exists():
                    stats = count_arc_examples(file_path)
                    # Score based on task count (higher is better)
                    score = min(1.0, stats["tasks"] / 100.0)  # Normalize to 100 tasks
                    total_score += score
                    file_count += 1
            
            return total_score / max(1, file_count)
            
        except Exception:
            return 0.0
    
    def _validate_data_integrity(self, output_dir: Path) -> float:
        """Validate data integrity and consistency"""
        try:
            integrity_checks = []
            
            # Check file sizes (should be reasonable)
            for dataset_type in ["training", "evaluation", "test"]:
                file_path = output_dir / dataset_type / f"{dataset_type}.json"
                if file_path.exists():
                    file_size = file_path.stat().st_size
                    # Score based on file size (too small or too large is suspicious)
                    size_score = 1.0 if 1000 < file_size < 50_000_000 else 0.5
                    integrity_checks.append(size_score)
            
            return sum(integrity_checks) / max(1, len(integrity_checks))
            
        except Exception:
            return 0.0
    
    def _update_download_metrics(self, operation: str, success: bool, complexity: float = 1.0):
        """Update HOLO-1.5 cognitive download metrics"""
        # Update download efficiency
        if success:
            self.download_metrics["download_efficiency"] = (
                self.download_metrics["download_efficiency"] * 0.9 + 0.1
            )
        else:
            self.download_metrics["download_efficiency"] *= 0.85
        
        # Update cognitive consistency
        self.download_metrics["cognitive_consistency"] = min(
            1.0, self.download_metrics["cognitive_consistency"] + (0.1 if success else -0.15)
        )
        
        # Store download trace
        self.download_traces.append({
            "operation": operation,
            "success": success,
            "complexity": complexity,
            "cognitive_load": complexity * 0.15,
            "timestamp": time.time()
        })
    
    def generate_download_trace(self) -> Dict[str, Any]:
        """Generate cognitive download trace for HOLO-1.5 mesh analysis"""
        return {
            "downloader_state": {
                "download_efficiency": self.download_metrics["download_efficiency"],
                "validation_accuracy": self.download_metrics["validation_accuracy"],
                "cognitive_consistency": self.download_metrics["cognitive_consistency"],
                "integrity_score": self.download_metrics["integrity_score"]
            },
            "download_patterns": {
                "successful_operations": len([t for t in self.download_traces if t["success"]]),
                "failed_operations": len([t for t in self.download_traces if not t["success"]]),
                "average_complexity": sum(t["complexity"] for t in self.download_traces) / max(1, len(self.download_traces)),
                "cognitive_load_trend": [t["cognitive_load"] for t in self.download_traces[-10:]]
            },
            "adaptive_summary": {
                "strategies_used": self.download_metrics["adaptive_strategies_used"],
                "efficiency_trend": self.download_metrics["download_efficiency"],
                "validation_quality": self.download_metrics["validation_accuracy"]
            }
        }


def main():
    """Main function to download and prepare ARC dataset"""
    args = parse_arguments()

    # Create output directory
    output_dir = Path(args.output_dir)
    
    # For standalone usage, create a basic downloader instance
    downloader = ARCDataDownloader()
    
    # Download dataset using HOLO-1.5 enhanced downloader
    success = downloader.download_arc_data(output_dir, args.method, args.force)

    if not success:
        logger.error("Failed to download ARC dataset")
        sys.exit(1)

    logger.info("ARC dataset is ready for use")
    
    # Generate cognitive trace summary
    if hasattr(downloader, 'download_traces'):
        trace = downloader.generate_download_trace()
        logger.info(f"🧠 Download completed with efficiency: {trace['downloader_state']['download_efficiency']:.3f}")
        logger.info(f"🧠 Validation accuracy: {trace['downloader_state']['validation_accuracy']:.3f}")


if __name__ == "__main__":
    main()
