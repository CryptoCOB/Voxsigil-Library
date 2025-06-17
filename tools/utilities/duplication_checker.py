#!/usr/bin/env python3
"""
Duplication Checker Utilities for VoxSigil System
Provides duplication detection, validation, and cleanup functionality.
"""

import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Add project paths for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class DuplicationChecker:
    """
    Advanced duplication checker for VoxSigil system.
    Handles detection, analysis, and management of duplicate content across various data types.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or str(project_root / "cache" / "duplication")
        self.hash_database = {}
        self.similarity_threshold = 0.9
        self.content_signatures = {}
        self._ensure_cache_dir()

    def _ensure_cache_dir(self):
        """Ensure cache directory exists."""
        os.makedirs(self.cache_dir, exist_ok=True)

    def check_exact_duplicates(
        self, items: List[Any], item_type: str = "generic"
    ) -> Dict[str, List[int]]:
        """
        Check for exact duplicates in a list of items.

        Args:
            items: List of items to check
            item_type: Type of items ('grid', 'text', 'file', 'generic')

        Returns:
            Dictionary mapping hash to list of indices with that content
        """
        duplicates = {}
        hash_to_indices = {}

        for i, item in enumerate(items):
            item_hash = self._calculate_hash(item, item_type)

            if item_hash not in hash_to_indices:
                hash_to_indices[item_hash] = []
            hash_to_indices[item_hash].append(i)

        # Find duplicates (hashes with more than one index)
        for item_hash, indices in hash_to_indices.items():
            if len(indices) > 1:
                duplicates[item_hash] = indices

        return duplicates

    def check_similar_items(
        self, items: List[Any], threshold: float = None
    ) -> List[Tuple[int, int, float]]:
        """
        Check for similar (but not identical) items.

        Args:
            items: List of items to check
            threshold: Similarity threshold (0-1)

        Returns:
            List of (index1, index2, similarity_score) tuples
        """
        if threshold is None:
            threshold = self.similarity_threshold

        similar_pairs = []

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                similarity = self._calculate_similarity(items[i], items[j])
                if similarity >= threshold:
                    similar_pairs.append((i, j, similarity))

        return similar_pairs

    def _calculate_hash(self, item: Any, item_type: str) -> str:
        """Calculate hash for an item based on its type."""
        if item_type == "grid":
            return self._hash_grid(item)
        elif item_type == "text":
            return self._hash_text(item)
        elif item_type == "file":
            return self._hash_file(item)
        else:
            return self._hash_generic(item)

    def _hash_grid(self, grid: List[List[int]]) -> str:
        """Calculate hash for a grid."""
        if not grid:
            return hashlib.md5(b"empty_grid").hexdigest()

        # Convert grid to string representation
        grid_str = json.dumps(grid, sort_keys=True)
        return hashlib.md5(grid_str.encode()).hexdigest()

    def _hash_text(self, text: str) -> str:
        """Calculate hash for text content."""
        # Normalize text (remove extra whitespace, convert to lowercase)
        normalized = " ".join(text.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()

    def _hash_file(self, filepath: Union[str, Path]) -> str:
        """Calculate hash for file content."""
        filepath = Path(filepath)
        if not filepath.exists():
            return hashlib.md5(b"missing_file").hexdigest()

        hash_md5 = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        except Exception as e:
            logger.warning(f"Failed to hash file {filepath}: {e}")
            return hashlib.md5(str(filepath).encode()).hexdigest()

        return hash_md5.hexdigest()

    def _hash_generic(self, item: Any) -> str:
        """Calculate hash for generic item."""
        try:
            # Try to serialize as JSON
            if isinstance(item, (dict, list)):
                item_str = json.dumps(item, sort_keys=True)
            else:
                item_str = str(item)

            return hashlib.md5(item_str.encode()).hexdigest()
        except Exception:
            # Fallback to string representation
            return hashlib.md5(str(item).encode()).hexdigest()

    def _calculate_similarity(self, item1: Any, item2: Any) -> float:
        """Calculate similarity between two items."""
        # Type-specific similarity calculation
        if isinstance(item1, list) and isinstance(item2, list):
            if item1 and isinstance(item1[0], list) and item2 and isinstance(item2[0], list):
                # Grid similarity
                return self._grid_similarity(item1, item2)
            else:
                # List similarity
                return self._list_similarity(item1, item2)
        elif isinstance(item1, str) and isinstance(item2, str):
            return self._text_similarity(item1, item2)
        elif isinstance(item1, dict) and isinstance(item2, dict):
            return self._dict_similarity(item1, item2)
        else:
            # Generic similarity based on string representation
            return self._string_similarity(str(item1), str(item2))

    def _grid_similarity(self, grid1: List[List[int]], grid2: List[List[int]]) -> float:
        """Calculate similarity between two grids."""
        if not grid1 or not grid2:
            return 1.0 if (not grid1 and not grid2) else 0.0

        # Size similarity
        h1, w1 = len(grid1), len(grid1[0]) if grid1 else 0
        h2, w2 = len(grid2), len(grid2[0]) if grid2 else 0

        if h1 != h2 or w1 != w2:
            # Different sizes - calculate based on overlap
            return self._grid_overlap_similarity(grid1, grid2)

        # Same size - calculate cell-by-cell similarity
        total_cells = h1 * w1
        matching_cells = 0

        for i in range(h1):
            for j in range(w1):
                if grid1[i][j] == grid2[i][j]:
                    matching_cells += 1

        return matching_cells / total_cells if total_cells > 0 else 1.0

    def _grid_overlap_similarity(self, grid1: List[List[int]], grid2: List[List[int]]) -> float:
        """Calculate similarity based on grid overlap."""
        h1, w1 = len(grid1), len(grid1[0]) if grid1 else 0
        h2, w2 = len(grid2), len(grid2[0]) if grid2 else 0

        # Calculate overlap region
        min_h, min_w = min(h1, h2), min(w1, w2)

        if min_h == 0 or min_w == 0:
            return 0.0

        matching_cells = 0
        for i in range(min_h):
            for j in range(min_w):
                if grid1[i][j] == grid2[i][j]:
                    matching_cells += 1

        overlap_cells = min_h * min_w
        total_cells = max(h1 * w1, h2 * w2)

        # Similarity based on overlap and size difference penalty
        overlap_similarity = matching_cells / overlap_cells if overlap_cells > 0 else 0
        size_penalty = overlap_cells / total_cells if total_cells > 0 else 0

        return overlap_similarity * size_penalty

    def _list_similarity(self, list1: List[Any], list2: List[Any]) -> float:
        """Calculate similarity between two lists."""
        if not list1 and not list2:
            return 1.0

        if not list1 or not list2:
            return 0.0

        # Jaccard similarity for lists
        set1, set2 = set(map(str, list1)), set(map(str, list2))
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union if union > 0 else 0.0

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        if text1 == text2:
            return 1.0

        # Normalize texts
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 and not words2:
            return 1.0

        if not words1 or not words2:
            return 0.0

        # Jaccard similarity for word sets
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _dict_similarity(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> float:
        """Calculate similarity between two dictionaries."""
        if dict1 == dict2:
            return 1.0

        all_keys = set(dict1.keys()).union(set(dict2.keys()))

        if not all_keys:
            return 1.0

        matching_keys = 0
        for key in all_keys:
            if key in dict1 and key in dict2:
                if dict1[key] == dict2[key]:
                    matching_keys += 1
                elif isinstance(dict1[key], (int, float)) and isinstance(dict2[key], (int, float)):
                    # Numerical similarity
                    max_val = max(abs(dict1[key]), abs(dict2[key]), 1)
                    diff = abs(dict1[key] - dict2[key])
                    matching_keys += max(0, 1 - diff / max_val)

        return matching_keys / len(all_keys)

    def _string_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings using Levenshtein distance."""
        if str1 == str2:
            return 1.0

        if not str1 or not str2:
            return 0.0

        # Simple Levenshtein distance calculation
        len1, len2 = len(str1), len(str2)

        # Create matrix
        matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]

        # Initialize first row and column
        for i in range(len1 + 1):
            matrix[i][0] = i
        for j in range(len2 + 1):
            matrix[0][j] = j

        # Fill matrix
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if str1[i - 1] == str2[j - 1]:
                    cost = 0
                else:
                    cost = 1

                matrix[i][j] = min(
                    matrix[i - 1][j] + 1,  # deletion
                    matrix[i][j - 1] + 1,  # insertion
                    matrix[i - 1][j - 1] + cost,  # substitution
                )

        # Calculate similarity
        max_len = max(len1, len2)
        distance = matrix[len1][len2]

        return 1.0 - (distance / max_len) if max_len > 0 else 1.0

    def analyze_dataset_duplicates(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze duplicates in a dataset of tasks/examples.

        Args:
            dataset: List of task/example dictionaries

        Returns:
            Analysis report with duplicate information
        """
        analysis = {
            "total_items": len(dataset),
            "exact_duplicates": {},
            "similar_items": [],
            "duplicate_statistics": {},
            "recommendations": [],
        }

        # Check for exact duplicates
        exact_dups = self.check_exact_duplicates(dataset, "generic")
        analysis["exact_duplicates"] = exact_dups

        # Check for similar items
        similar_items = self.check_similar_items(dataset)
        analysis["similar_items"] = similar_items

        # Calculate statistics
        total_duplicates = sum(len(indices) for indices in exact_dups.values())
        unique_items = len(dataset) - total_duplicates + len(exact_dups)

        analysis["duplicate_statistics"] = {
            "exact_duplicate_groups": len(exact_dups),
            "total_duplicate_items": total_duplicates,
            "unique_items": unique_items,
            "duplicate_percentage": (total_duplicates / len(dataset)) * 100 if dataset else 0,
            "similar_pairs": len(similar_items),
        }

        # Generate recommendations
        recommendations = []
        if exact_dups:
            recommendations.append(
                f"Remove {total_duplicates - len(exact_dups)} exact duplicate items"
            )

        if similar_items:
            recommendations.append(
                f"Review {len(similar_items)} similar item pairs for potential consolidation"
            )

        if analysis["duplicate_statistics"]["duplicate_percentage"] > 10:
            recommendations.append("High duplication rate detected - consider dataset cleanup")

        analysis["recommendations"] = recommendations

        return analysis

    def find_file_duplicates(
        self, directory: Union[str, Path], file_patterns: List[str] = None
    ) -> Dict[str, List[str]]:
        """
        Find duplicate files in a directory.

        Args:
            directory: Directory to search
            file_patterns: List of file patterns to include (e.g., ['*.py', '*.json'])

        Returns:
            Dictionary mapping hash to list of duplicate file paths
        """
        directory = Path(directory)

        if file_patterns is None:
            file_patterns = ["*"]

        file_hashes = {}

        # Collect all matching files
        for pattern in file_patterns:
            for filepath in directory.rglob(pattern):
                if filepath.is_file():
                    file_hash = self._hash_file(filepath)

                    if file_hash not in file_hashes:
                        file_hashes[file_hash] = []
                    file_hashes[file_hash].append(str(filepath))

        # Filter to only duplicates
        duplicates = {h: paths for h, paths in file_hashes.items() if len(paths) > 1}

        return duplicates

    def remove_duplicates(
        self, items: List[Any], keep_first: bool = True
    ) -> Tuple[List[Any], List[int]]:
        """
        Remove duplicates from a list while preserving order.

        Args:
            items: List of items to deduplicate
            keep_first: Whether to keep first occurrence of duplicates

        Returns:
            Tuple of (deduplicated_list, removed_indices)
        """
        seen_hashes = set()
        unique_items = []
        removed_indices = []

        for i, item in enumerate(items):
            item_hash = self._calculate_hash(item, "generic")

            if item_hash not in seen_hashes:
                seen_hashes.add(item_hash)
                unique_items.append(item)
            else:
                if keep_first:
                    removed_indices.append(i)
                else:
                    # Remove the first occurrence instead
                    for j, unique_item in enumerate(unique_items):
                        if self._calculate_hash(unique_item, "generic") == item_hash:
                            removed_indices.append(len(unique_items) - len(unique_items[j:]))
                            unique_items[j] = item
                            break

        return unique_items, removed_indices

    def create_signature(self, item: Any) -> str:
        """
        Create a content signature for an item.

        Args:
            item: Item to create signature for

        Returns:
            Content signature string
        """
        # Create a more detailed signature than simple hash
        signature_parts = []

        if isinstance(item, list) and item and isinstance(item[0], list):
            # Grid signature
            height, width = len(item), len(item[0]) if item else 0
            unique_values = len(set(sum(item, [])))
            signature_parts.extend([f"grid_{height}x{width}", f"values_{unique_values}"])

            # Add pattern information
            if self._has_symmetry(item):
                signature_parts.append("symmetric")

        elif isinstance(item, dict):
            # Dictionary signature
            signature_parts.append(f"dict_{len(item)}_keys")
            for key in sorted(item.keys())[:5]:  # First 5 keys
                signature_parts.append(f"key_{key}")

        elif isinstance(item, str):
            # String signature
            word_count = len(item.split())
            signature_parts.extend([f"text_{len(item)}_chars", f"words_{word_count}"])

        return "_".join(signature_parts)

    def _has_symmetry(self, grid: List[List[int]]) -> bool:
        """Check if grid has any symmetry."""
        if not grid:
            return False

        # Check horizontal symmetry
        if grid == grid[::-1]:
            return True

        # Check vertical symmetry
        if all(row == row[::-1] for row in grid):
            return True

        return False

    def batch_check_duplicates(self, datasets: Dict[str, List[Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Check duplicates across multiple datasets.

        Args:
            datasets: Dictionary mapping dataset names to lists of items

        Returns:
            Analysis results for each dataset
        """
        results = {}

        for dataset_name, dataset in datasets.items():
            logger.info(f"Checking duplicates in dataset: {dataset_name}")
            results[dataset_name] = self.analyze_dataset_duplicates(dataset)

        return results

    def export_duplicate_report(
        self, analysis: Dict[str, Any], output_path: Union[str, Path]
    ) -> str:
        """
        Export duplicate analysis report to file.

        Args:
            analysis: Analysis results from analyze_dataset_duplicates
            output_path: Path to save report

        Returns:
            Path to saved report
        """
        output_path = Path(output_path)

        # Create detailed report
        report = {
            "analysis_timestamp": str(pd.Timestamp.now())
            if "pd" in globals()
            else str(datetime.now()),
            "summary": analysis.get("duplicate_statistics", {}),
            "exact_duplicates": analysis.get("exact_duplicates", {}),
            "similar_items": analysis.get("similar_items", []),
            "recommendations": analysis.get("recommendations", []),
        }

        # Save as JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Duplicate report saved to: {output_path}")
        return str(output_path)

    def clear_cache(self):
        """Clear internal caches."""
        self.hash_database.clear()
        self.content_signatures.clear()
        logger.info("Duplication checker cache cleared")


# Utility functions for quick access
def find_exact_duplicates(items: List[Any]) -> Dict[str, List[int]]:
    """Quick function to find exact duplicates."""
    checker = DuplicationChecker()
    return checker.check_exact_duplicates(items)


def find_similar_items(items: List[Any], threshold: float = 0.9) -> List[Tuple[int, int, float]]:
    """Quick function to find similar items."""
    checker = DuplicationChecker()
    return checker.check_similar_items(items, threshold)


def remove_duplicates_simple(items: List[Any]) -> List[Any]:
    """Simple duplicate removal function."""
    checker = DuplicationChecker()
    unique_items, _ = checker.remove_duplicates(items)
    return unique_items


# Import datetime for timestamps
from datetime import datetime
