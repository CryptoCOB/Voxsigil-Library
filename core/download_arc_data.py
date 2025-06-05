#!/usr/bin/env python
"""
download_arc_data.py - ARC Dataset Downloader and Preparer

This script downloads the Abstraction and Reasoning Corpus (ARC) dataset
and prepares it for training with the GRID-Former model.
"""

import os
import sys
import json
import logging
import requests
import argparse
import zipfile
from pathlib import Path
from typing import Dict

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
        and evaluation_stats["tasks"] > 0
        and test_stats["tasks"] > 0
    )


def main():
    """Main function to download and prepare ARC dataset"""
    args = parse_arguments()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download dataset using specified method
    if args.method == "zip":
        success = download_arc_zip(output_dir, args.force)
    else:  # direct
        success = download_arc_direct(output_dir, args.force)

    if not success:
        logger.error("Failed to download ARC dataset")
        sys.exit(1)

    # Verify dataset
    if verify_arc_data(output_dir):
        logger.info("ARC dataset verified successfully")

        # Create a simple README file to help users understand the dataset
        readme_path = output_dir / "README.md"
        with open(readme_path, "w") as f:
            f.write("""# Abstraction and Reasoning Corpus (ARC)

This directory contains the ARC dataset for training and evaluating the GRID-Former model.

## Structure:
- `training/`: Training data tasks
- `evaluation/`: Evaluation data tasks
- `test/`: Test data tasks (without solutions)

## Usage:
To train the GRID-Former model on this dataset, run:
```
python train_grid_former_arc.py --arc_data_dir=./data/arc
```

For more information on the ARC dataset, visit: https://github.com/fchollet/ARC
""")

        logger.info(f"README file created at {readme_path}")
        logger.info(f"ARC dataset is ready for use at {output_dir}")

    else:
        logger.error("ARC dataset verification failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
