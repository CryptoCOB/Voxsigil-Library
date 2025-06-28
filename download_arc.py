#!/usr/bin/env python
"""
Download ARC Dataset
"""

import json
import os

import requests

print("Downloading ARC dataset...")

# Create directory
os.makedirs("arc_data", exist_ok=True)

# Direct download from original ARC repository
urls = {
    "training.json": "https://raw.githubusercontent.com/fchollet/ARC/master/data/training.json",
    "evaluation.json": "https://raw.githubusercontent.com/fchollet/ARC/master/data/evaluation.json",
}

# Try alternative Kaggle URLs if GitHub fails
kaggle_urls = {
    "training.json": "https://github.com/fchollet/ARC/raw/master/data/training.json",
    "evaluation.json": "https://github.com/fchollet/ARC/raw/master/data/evaluation.json",
}

for filename, url in urls.items():
    try:
        print(f"Downloading {filename}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Save to file
        with open(f"arc_data/{filename}", "w") as f:
            f.write(response.text)

        # Verify it's valid JSON
        data = json.loads(response.text)
        print(f"✅ {filename}: {len(data)} tasks")

    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")
        # Try Kaggle URL
        if filename in kaggle_urls:
            try:
                print(f"Trying alternative URL for {filename}...")
                response = requests.get(kaggle_urls[filename], timeout=30)
                response.raise_for_status()

                with open(f"arc_data/{filename}", "w") as f:
                    f.write(response.text)

                data = json.loads(response.text)
                print(f"✅ {filename} (alt): {len(data)} tasks")

            except Exception as e2:
                print(f"❌ Alternative URL also failed: {e2}")

print("Download complete!")
