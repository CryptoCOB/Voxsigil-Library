"""
VoxSigil Schema Validator

This script validates VoxSigil entries against the official 1.4-alpha schema
and performs dependency validation to ensure all modules can be imported correctly.
"""

import os
import sys

import importlib
import traceback


# Add the parent directory to the path to enable imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Check for jsonschema
try:
    import importlib.util

    HAVE_JSONSCHEMA = importlib.util.find_spec("jsonschema") is not None
except ImportError:
    HAVE_JSONSCHEMA = False

# Default schema paths
SCHEMAS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schemas")
DEFAULT_SCHEMA_PATH = os.path.join(
    SCHEMAS_DIR, "voxsigil-schema-1.5-holo-alpha.json"
)  # Updated to 1.5-holo-alpha schema


def check_module(module_name):
    """Check if a module can be imported correctly."""
    try:
        importlib.import_module(module_name)
        print(f"✓ Successfully imported {module_name}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import {module_name}: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"✗ Error when importing {module_name}: {e}")
        traceback.print_exc()
        return False


def run_validation_test():
    """Run a simple validation test to ensure the generator works."""
    try:
        from VoxSigilDatasetTools import generator

        # Create a test directory
        test_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test_output"
        )
        os.makedirs(test_dir, exist_ok=True)

        # Generate a single sigil
        print("\nGenerating a test sigil...")
        sigil = generator.generate_sigil()

        # Write to file
        from VoxSigilDatasetTools.field_generators.utils import export_utils

        test_file = os.path.join(test_dir, "test_sigil.json")
        export_utils.write_json(sigil, test_file)

        # Validate it
        is_valid, missing = generator.validate_sigil_schema(sigil)

        if is_valid:
            print("✓ Successfully generated and validated a test sigil")
            print(f"  Saved to: {test_file}")
            return True
        else:
            print("✗ Generated sigil failed validation")
            print(f"  Missing fields: {', '.join(missing)}")
            return False

    except Exception as e:
        print(f"✗ Validation test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    print("Validating VoxSigil Dataset Tools imports...\n")

    # Define modules to check
    modules = [
        "VoxSigilDatasetTools",
        "VoxSigilDatasetTools.schema_fields",
        "VoxSigilDatasetTools.generator",
        "VoxSigilDatasetTools.field_generators",
        "VoxSigilDatasetTools.field_generators.core_fields",
        "VoxSigilDatasetTools.field_generators.structure_fields",
        "VoxSigilDatasetTools.field_generators.prompt_fields",
        "VoxSigilDatasetTools.field_generators.metadata_fields",
        "VoxSigilDatasetTools.field_generators.advanced_fields",
        "VoxSigilDatasetTools.field_generators.test_fields",
        "VoxSigilDatasetTools.field_generators.utils.json_helpers",
        "VoxSigilDatasetTools.field_generators.utils.random_utils",
        "VoxSigilDatasetTools.field_generators.utils.export_utils",
    ]

    # Check each module
    all_imports_ok = all(check_module(module) for module in modules)

    if all_imports_ok:
        print("\n✓ All imports are valid")

        # Run validation test
        test_ok = run_validation_test()

        if test_ok:
            print("\n✓ Validation test passed")
            print("\nVoxSigil Dataset Tools is ready to use!")
            return 0
        else:
            print("\n✗ Validation test failed")
            return 1
    else:
        print("\n✗ Some imports failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
