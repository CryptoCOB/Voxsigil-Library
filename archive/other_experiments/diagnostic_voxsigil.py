"""
Quick diagnostic - list all voxsigil files and try to identify problem ones.
"""

from pathlib import Path
import yaml

LIBRARY_BASE = Path("c:/nebula-social-crypto-core/voxsigil_library/library_sigil")


categories = ["pglyph", "scaffolds", "sigils", "flows", "tags"]
errors_by_cat = {}

for cat in categories:
    cat_path = LIBRARY_BASE / cat
    if not cat_path.exists():
        continue
    
    files = list(cat_path.glob("*.voxsigil"))
    print(f"\n{cat.upper()}: {len(files)} files")
    
    error_count = 0
    for f in files[:5]:  # Test first 5 of each
        try:
            text = f.read_text(encoding="utf-8")
            data = yaml.safe_load(text)
            if not data:
                print(f"  [WARN] {f.name}: empty YAML")
                error_count += 1
        except (OSError, UnicodeDecodeError, yaml.YAMLError):
            print(f"  [ERROR] {f.name}: parse failed")
            error_count += 1
    
    if error_count == 0:
        print("  ✓ Sample of 5 files OK")
    else:
        print(f"  ✗ {error_count}/5 had issues")




print("\n[*] Diagnostic complete")
