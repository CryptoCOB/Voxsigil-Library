"""
Generate synthetic VoxSigil entries using a local Llama API.

- Reads existing VoxSigil files as style anchors
- Calls an OpenAI-compatible /chat/completions endpoint
- Validates schema fields and deduplicates by content hash
"""

import os
import re
import json
import glob
import time
import hashlib
import random
import urllib.request
import urllib.error
from typing import Dict, List, Tuple

import yaml


REQUIRED_META_FIELDS = ["sigil", "alias", "tag"]
REQUIRED_COGNITIVE_FIELDS = ["principle", "structure"]
REQUIRED_IMPL_FIELDS = ["usage"]
REQUIRED_TOP_LEVEL = ["meta", "holo_mesh", "cognitive", "implementation", "connectivity"]


def load_env(env_path: str) -> Dict[str, str]:
    """Load simple KEY=VALUE pairs from a .env file."""
    env = {}
    if not os.path.exists(env_path):
        return env
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip()
    return env


def canonicalize_sigil_data(sigil_data: Dict) -> str:
    """Canonical JSON representation for stable hashing."""
    return json.dumps(sigil_data, sort_keys=True, ensure_ascii=False)


def hash_sigil_data(sigil_data: Dict) -> str:
    """SHA-256 content hash for a sigil object."""
    canonical = canonicalize_sigil_data(sigil_data)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_existing_sigils(base_dir: str) -> Tuple[List[Dict], Dict[str, str]]:
    """Load existing .voxsigil entries and return list and content hash map."""
    sigils = []
    hashes = {}
    for path in glob.glob(os.path.join(base_dir, "**", "*.voxsigil"), recursive=True):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data:
                    sigils.append(data)
                    hashes[hash_sigil_data(data)] = path
        except (OSError, yaml.YAMLError):
            continue
    return sigils, hashes


def pick_examples(sigils: List[Dict], count: int = 3) -> List[Dict]:
    """Pick random examples for prompt conditioning."""
    if len(sigils) <= count:
        return sigils
    return random.sample(sigils, count)


def build_prompt(schema_guide: str, examples: List[Dict]) -> List[Dict]:
    """Build a chat prompt with schema guide and examples."""
    example_text = "\n\n".join(
        [yaml.safe_dump(ex, sort_keys=False, allow_unicode=True) for ex in examples]
    )
    system_prompt = (
        "You are a VoxSigil generator. Output exactly one YAML object that "
        "conforms to VoxSigil schema. Use real Unicode sigils, rich principle, "
        "and realistic structure. Always include meta, holo_mesh, cognitive, "
        "implementation, and connectivity."
    )
    user_prompt = (
        "Schema guide:\n"
        f"{schema_guide}\n\n"
        "Here are valid examples:\n"
        f"{example_text}\n\n"
        "Create ONE new VoxSigil YAML object with unique sigil, alias, and tag. "
        "Use the same structural format as examples. Ensure cognitive.structure "
        "includes composite_type and temporal_structure. Ensure implementation.usage "
        "has description, example, and explanation."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def call_llama(base_url: str, api_key: str, model: str, messages: List[Dict]) -> str:
    """Call a local OpenAI-compatible chat completion endpoint."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.8,
        "max_tokens": 1200,
    }
    req = urllib.request.Request(
        url=f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}" if api_key else "",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def extract_yaml_block(text: str) -> str:
    """Extract YAML content from fenced code block if present."""
    match = re.search(r"```yaml\n([\s\S]*?)\n```", text)
    if match:
        return match.group(1).strip()
    return text.strip()


def validate_sigil(data: Dict) -> Tuple[bool, List[str]]:
    """Validate minimal VoxSigil schema structure."""
    errors = []
    for field in REQUIRED_TOP_LEVEL:
        if field not in data:
            errors.append(f"{field} missing")
    meta = data.get("meta", {})
    cognitive = data.get("cognitive", {})
    impl = data.get("implementation", {})
    holo = data.get("holo_mesh", {})

    for field in REQUIRED_META_FIELDS:
        if not meta.get(field):
            errors.append(f"meta.{field} missing")
    for field in REQUIRED_COGNITIVE_FIELDS:
        if not cognitive.get(field):
            errors.append(f"cognitive.{field} missing")
    for field in REQUIRED_IMPL_FIELDS:
        if not impl.get(field):
            errors.append(f"implementation.{field} missing")
    if not holo:
        errors.append("holo_mesh missing")

    structure = cognitive.get("structure", {})
    if not structure.get("composite_type"):
        errors.append("cognitive.structure.composite_type missing")
    if not structure.get("temporal_structure"):
        errors.append("cognitive.structure.temporal_structure missing")

    usage = impl.get("usage", {})
    if isinstance(usage, dict):
        if not usage.get("description"):
            errors.append("implementation.usage.description missing")
        if not usage.get("example"):
            errors.append("implementation.usage.example missing")
        if not usage.get("explanation"):
            errors.append("implementation.usage.explanation missing")
    else:
        errors.append("implementation.usage must be a mapping")

    return len(errors) == 0, errors


def safe_filename(name: str) -> str:
    """Normalize alias into a filesystem-friendly base name."""
    name = re.sub(r"[^a-zA-Z0-9_\-]+", "_", name).strip("_")
    return name.lower() or "voxsigil"


def main():
    """Generate VoxSigil entries via local Llama."""
    env = load_env(os.path.join(os.path.dirname(__file__), ".env"))
    base_url = os.getenv("LLAMA_BASE_URL", env.get("LLAMA_BASE_URL", ""))
    api_key = os.getenv("LLAMA_API_KEY", env.get("LLAMA_API_KEY", ""))
    model = os.getenv("LLAMA_MODEL", env.get("LLAMA_MODEL", ""))
    output_dir = os.getenv(
        "VOXSIGIL_OUTPUT_DIR",
        env.get(
            "VOXSIGIL_OUTPUT_DIR",
            r"c:\nebula-social-crypto-core\voxsigil_library\library_sigil\sigils",
        ),
    )
    target_count = int(
        os.getenv("VOXSIGIL_TARGET_COUNT", env.get("VOXSIGIL_TARGET_COUNT", "50"))
    )
    batch_size = int(
        os.getenv("VOXSIGIL_BATCH_SIZE", env.get("VOXSIGIL_BATCH_SIZE", "5"))
    )
    seed = int(os.getenv("VOXSIGIL_SEED", env.get("VOXSIGIL_SEED", "1337")))

    if not base_url or not model:
        raise SystemExit("LLAMA_BASE_URL and LLAMA_MODEL must be set.")

    random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    base_dir = r"c:\nebula-social-crypto-core\voxsigil_library"
    sigils, existing_hashes = load_existing_sigils(base_dir)

    schema_guide_path = r"c:\UBLT\VOXSIGIL COMPLETE SIGIL SCHEMA.md"
    if os.path.exists(schema_guide_path):
        with open(schema_guide_path, "r", encoding="utf-8") as f:
            schema_guide = f.read()
    else:
        schema_guide = (
            "Use VoxSigil schema fields: meta, holo_mesh, cognitive, "
            "implementation, connectivity."
        )

    created = 0
    attempts = 0
    output_hashes = {}

    while created < target_count and attempts < target_count * 10:
        examples = pick_examples(sigils, count=3)
        messages = build_prompt(schema_guide, examples)
        attempts += 1

        try:
            response = call_llama(base_url, api_key, model, messages)
            yaml_text = extract_yaml_block(response)
            data = yaml.safe_load(yaml_text)
        except (OSError, ValueError, yaml.YAMLError, json.JSONDecodeError, urllib.error.URLError):
            continue

        if not isinstance(data, dict):
            continue

        ok, errors = validate_sigil(data)
        if not ok:
            if errors:
                print(f"[SKIP] Validation errors: {errors[:3]}")
            continue

        sigil_hash = hash_sigil_data(data)
        if sigil_hash in existing_hashes or sigil_hash in output_hashes:
            continue

        alias = data.get("meta", {}).get("alias", "voxsigil")
        file_name = safe_filename(alias)
        output_path = os.path.join(output_dir, f"{file_name}.voxsigil")

        if os.path.exists(output_path):
            output_path = os.path.join(output_dir, f"{file_name}_{created}.voxsigil")

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

        output_hashes[sigil_hash] = output_path
        created += 1
        print(f"[OK] {created}/{target_count}: {output_path}")
        if batch_size and created % batch_size == 0:
            time.sleep(0.5)
        else:
            time.sleep(0.2)

    print(f"Created {created} synthetic sigils (attempts={attempts})")


if __name__ == "__main__":
    main()
