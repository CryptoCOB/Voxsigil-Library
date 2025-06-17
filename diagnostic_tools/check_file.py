#!/usr/bin/env python3
"""Check file integrity."""

import codecs

try:
    with codecs.open(
        "gui/components/mesh_map_panel.py", "r", encoding="utf-8", errors="replace"
    ) as f:
        content = f.read()
        print(f"File length: {len(content)} characters")
        print("Last 500 characters:")
        print(repr(content[-500:]))

        # Check if file ends properly
        lines = content.split("\n")
        print(f"Total lines: {len(lines)}")
        print("Last 10 lines:")
        for i, line in enumerate(lines[-10:], len(lines) - 9):
            print(f"{i:3}: {repr(line)}")

except Exception as e:
    print(f"Error reading file: {e}")
