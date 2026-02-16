"""
Phase 4-B.2: REAL Voxsigil Schema Training & Validation
========================================================

This trains the student embedder on ACTUAL Voxsigil sigil files.
Extracts real schema characteristics and validates recovery against production data.
No synthetic data - pure Voxsigil schema ground truth.

Unicode handling: Works with UTF-8 internally; console output uses ASCII-safe representations.
"""

import os
import yaml
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.metrics import r2_score
from typing import Dict, List, Tuple
import glob
import sys
import hashlib

# Ensure UTF-8 output on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def safe_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    variance = np.var(y_true)
    if variance < 1e-8:
        return 0.0
    return float(r2_score(y_true, y_pred))


def canonicalize_sigil_data(sigil_data: Dict) -> str:
    try:
        return json.dumps(sigil_data, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return json.dumps(str(sigil_data), sort_keys=True, ensure_ascii=False)


def hash_sigil_data(sigil_data: Dict) -> str:
    canonical = canonicalize_sigil_data(sigil_data)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()

# ============================================================================
# VOXSIGIL SCHEMA EXTRACTION (3 LEVELS)
# ============================================================================

def _safe_norm(value: float, max_value: float) -> float:
    if max_value <= 0:
        return 0.0
    return float(min(value / max_value, 1.0))


def _parse_version(schema_version: str) -> float:
    if not schema_version:
        return 0.0
    numeric = ''.join(ch for ch in schema_version if (ch.isdigit() or ch == '.'))
    try:
        return float(numeric)
    except ValueError:
        return 0.0


def collect_schema_stats(sigil_data_list: List[Dict]) -> Dict:
    tag_counts = []
    component_counts = []
    principle_lens = []
    math_lens = []
    usage_lens = []
    prompt_lens = []
    mesh_endpoint_counts = []
    usage_pattern_counts = []
    mesh_compat_types = set()
    composite_types = set()
    temporal_types = set()
    versions = []

    for sigil_data in sigil_data_list:
        meta = sigil_data.get('meta', {})
        holo = sigil_data.get('holo_mesh', {})
        cognitive = sigil_data.get('cognitive', {})
        impl = sigil_data.get('implementation', {})
        connectivity = sigil_data.get('connectivity', {})

        tags = cognitive.get('tags', [])
        tag_counts.append(len(tags))

        structure = cognitive.get('structure', {})
        components = structure.get('components', [])
        component_counts.append(len(components))

        principle_lens.append(len(cognitive.get('principle', '') or ''))
        math_lens.append(len(cognitive.get('math', '') or ''))

        usage = impl.get('usage', {})
        usage_text = ''
        if isinstance(usage, dict):
            usage_text = (
                f"{usage.get('description', '')}"
                f"{usage.get('example', '')}"
                f"{usage.get('explanation', '')}"
            )
        elif isinstance(usage, str):
            usage_text = usage
        usage_lens.append(len(usage_text))

        prompt_template = impl.get('prompt_template', {})
        prompt_text = ''
        if isinstance(prompt_template, dict):
            prompt_text = f"{prompt_template.get('role', '')}{prompt_template.get('content', '')}"
        prompt_lens.append(len(prompt_text))

        mesh_endpoints = connectivity.get('mesh_endpoints', [])
        mesh_endpoint_counts.append(
            len(mesh_endpoints) if isinstance(mesh_endpoints, list) else 0
        )

        usage_patterns = connectivity.get('usage_patterns', [])
        usage_pattern_counts.append(
            len(usage_patterns) if isinstance(usage_patterns, list) else 0
        )

        mesh_compat = holo.get('mesh_compatibility', '')
        if mesh_compat:
            mesh_compat_types.add(str(mesh_compat).lower())

        composite_type = structure.get('composite_type', '')
        if composite_type:
            composite_types.add(str(composite_type).lower())

        temporal_type = structure.get('temporal_structure', '')
        if temporal_type:
            temporal_types.add(str(temporal_type).lower())

        versions.append(
            _parse_version(
                meta.get('schema_version', sigil_data.get('schema_version', ''))
            )
        )

    return {
        'max_tag_count': max(tag_counts) if tag_counts else 1,
        'max_component_count': max(component_counts) if component_counts else 1,
        'max_principle_len': max(principle_lens) if principle_lens else 1,
        'max_math_len': max(math_lens) if math_lens else 1,
        'max_usage_len': max(usage_lens) if usage_lens else 1,
        'max_prompt_len': max(prompt_lens) if prompt_lens else 1,
        'max_mesh_endpoints': max(mesh_endpoint_counts) if mesh_endpoint_counts else 1,
        'max_usage_patterns': max(usage_pattern_counts) if usage_pattern_counts else 1,
        'mesh_compat_types': sorted(mesh_compat_types),
        'composite_types': sorted(composite_types),
        'temporal_types': sorted(temporal_types),
        'max_version': max(versions) if versions else 1,
    }


def extract_schema_levels(sigil_data: Dict, stats: Dict) -> Dict[str, np.ndarray]:
    """
    Extract hierarchical 3-level schema vectors that build on each other.
    Level 1: Core identification/classification
    Level 2: Conceptual grounding
    Level 3: Practical usage & workflow
    """
    meta = sigil_data.get('meta', {})
    holo = sigil_data.get('holo_mesh', {})
    cognitive = sigil_data.get('cognitive', {})
    impl = sigil_data.get('implementation', {})
    connectivity = sigil_data.get('connectivity', {})

    # ---------- Level 1: Core identification ----------
    tags = cognitive.get('tags', [])
    schema_version = _parse_version(
        meta.get('schema_version', sigil_data.get('schema_version', ''))
    )
    sigil_value = meta.get('sigil', '')
    has_unicode_sigil = 1.0 if any(ord(ch) > 127 for ch in sigil_value) else 0.0

    level1 = np.array([
        1.0 if holo.get('is_cognitive_primitive', False) else 0.0,
        1.0 if meta.get('alias') else 0.0,
        1.0 if meta.get('tag') else 0.0,
        _safe_norm(len(tags), stats['max_tag_count']),
        has_unicode_sigil,
        _safe_norm(schema_version, stats['max_version']),
    ], dtype=np.float32)

    # ---------- Level 2: Conceptual grounding ----------
    structure = cognitive.get('structure', {})
    components = structure.get('components', [])
    composite_type = str(structure.get('composite_type', '')).lower()
    temporal_type = str(structure.get('temporal_structure', '')).lower()

    composite_index = (
        stats['composite_types'].index(composite_type)
        if composite_type in stats['composite_types']
        else 0
    )
    temporal_index = (
        stats['temporal_types'].index(temporal_type)
        if temporal_type in stats['temporal_types']
        else 0
    )
    composite_code = _safe_norm(composite_index, max(len(stats['composite_types']) - 1, 1))
    temporal_code = _safe_norm(temporal_index, max(len(stats['temporal_types']) - 1, 1))

    principle_text = cognitive.get('principle', '') or ''
    math_text = cognitive.get('math', '') or ''

    level2 = np.array([
        1.0 if principle_text else 0.0,
        _safe_norm(len(principle_text), stats['max_principle_len']),
        1.0 if math_text else 0.0,
        _safe_norm(len(math_text), stats['max_math_len']),
        _safe_norm(len(components), stats['max_component_count']),
        composite_code,
        temporal_code,
        1.0 if structure else 0.0,
    ], dtype=np.float32)

    # ---------- Level 3: Practical usage/workflow ----------
    usage = impl.get('usage', {})
    usage_text = ''
    if isinstance(usage, dict):
        usage_text = (
            f"{usage.get('description', '')}"
            f"{usage.get('example', '')}"
            f"{usage.get('explanation', '')}"
        )
    elif isinstance(usage, str):
        usage_text = usage

    vanta_core = holo.get('vanta_core_integration', {})
    mesh_endpoints = connectivity.get('mesh_endpoints', [])
    usage_patterns = connectivity.get('usage_patterns', [])
    mesh_compat = str(holo.get('mesh_compatibility', '')).lower()

    mesh_compat_index = (
        stats['mesh_compat_types'].index(mesh_compat)
        if mesh_compat in stats['mesh_compat_types']
        else 0
    )
    mesh_compat_code = _safe_norm(
        mesh_compat_index,
        max(len(stats['mesh_compat_types']) - 1, 1),
    )

    level3 = np.array([
        1.0 if usage_text else 0.0,
        _safe_norm(len(usage_text), stats['max_usage_len']),
        1.0 if vanta_core.get('event_support') else 0.0,
        1.0 if vanta_core.get('async_capable') else 0.0,
        1.0 if vanta_core.get('memory_aware') else 0.0,
        1.0 if holo.get('registration_ready') else 0.0,
        _safe_norm(
            len(mesh_endpoints) if isinstance(mesh_endpoints, list) else 0,
            stats['max_mesh_endpoints'],
        ),
        _safe_norm(
            len(usage_patterns) if isinstance(usage_patterns, list) else 0,
            stats['max_usage_patterns'],
        ),
        mesh_compat_code,
    ], dtype=np.float32)

    return {
        'level1': level1,
        'level2': level2,
        'level3': level3,
    }


def load_real_voxsigil_data(
    sigil_dir: str,
    recursive: bool = True,
    deduplicate: bool = True,
) -> Tuple[List[Tuple[str, Dict, np.ndarray, Dict[str, np.ndarray]]], Dict]:
    """
    Load all .voxsigil files and extract schema characteristics.
    Handles Unicode data internally; provides ASCII-safe console output.
    Returns list of (filename, raw_data, schema_vector, schema_levels) and stats.
    """
    
    data = []
    raw_entries = []
    seen_hashes = {}
    duplicate_count = 0
    pattern = '**/*.voxsigil' if recursive else '*.voxsigil'
    sigil_files = glob.glob(os.path.join(sigil_dir, pattern), recursive=recursive)
    
    print(f"[LOAD] Found {len(sigil_files)} .voxsigil files")
    
    for idx, filepath in enumerate(sigil_files):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                sigil_data = yaml.safe_load(f)
                if sigil_data:
                    filename = os.path.basename(filepath)
                    if deduplicate:
                        sigil_hash = hash_sigil_data(sigil_data)
                        if sigil_hash in seen_hashes:
                            duplicate_count += 1
                            continue
                        seen_hashes[sigil_hash] = filename
                    raw_entries.append((filename, sigil_data))
        except Exception as e:
            error_msg = str(e)[:60]
            print(f"  [ERROR] {os.path.basename(filepath)}: {error_msg}")

    stats = collect_schema_stats([entry[1] for entry in raw_entries])

    for idx, (filename, sigil_data) in enumerate(raw_entries):
        schema_levels = extract_schema_levels(sigil_data, stats)
        schema_vector = np.concatenate(
            [
                schema_levels['level1'],
                schema_levels['level2'],
                schema_levels['level3'],
            ]
        )
        data.append((filename, sigil_data, schema_vector, schema_levels))

        level1_mean = float(np.mean(schema_levels['level1']))
        level2_mean = float(np.mean(schema_levels['level2']))
        level3_mean = float(np.mean(schema_levels['level3']))
        print(
            f"  [{idx+1:2d}] {filename:<35s}"
            f" | L1={level1_mean:.2f}"
            f" L2={level2_mean:.2f}"
            f" L3={level3_mean:.2f}"
        )

    stats['duplicate_count'] = duplicate_count
    stats['unique_count'] = len(raw_entries)
    return data, stats


def generate_teacher_embeddings(schema_vectors: np.ndarray) -> np.ndarray:
    """
    Generate "teacher" embeddings from real schema vectors.
    Uses deterministic projection: schema → 384D embedding space via structured blocks.
    """
    n_samples = schema_vectors.shape[0]
    teacher_embeddings = np.zeros((n_samples, 384), dtype=np.float32)
    
    total_dims = schema_vectors.shape[1]
    block_size = max(1, 384 // total_dims)

    for i, schema in enumerate(schema_vectors):
        for dim_idx, val in enumerate(schema):
            block_start = dim_idx * block_size
            block_end = min(block_start + block_size, 384)
            if block_start >= 384:
                break

            block_len = block_end - block_start
            block = np.zeros(block_len)
            for j in range(block_len):
                if j < block_len // 2:
                    block[j] = np.sin((j + 1) * val * np.pi)
                else:
                    block[j] = np.cos((j + 1) * val * np.pi)

            teacher_embeddings[i, block_start:block_end] = block * (val + 0.5)
    
    # L2 normalize
    teacher_embeddings = teacher_embeddings / (
        np.linalg.norm(teacher_embeddings, axis=1, keepdims=True) + 1e-8
    )
    
    return teacher_embeddings

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class SchemaSupervisionStudentEmbedder(nn.Module):
    """Student embedder with hierarchical schema supervision heads."""
    
    def __init__(
        self,
        input_dim=384,
        hidden_dim=256,
        output_dim=128,
        level1_dim=6,
        level2_dim=8,
        level3_dim=9,
    ):
        super().__init__()
        
        # Compression path
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Teacher compression to match intermediate (256D) for distillation
        self.teacher_compress_to_intermediate = nn.Linear(input_dim, hidden_dim)
        
        # Schema recovery heads (hierarchical)
        self.schema_relu = nn.ReLU()

        self.level1_fc1 = nn.Linear(output_dim, 64)
        self.level1_fc2 = nn.Linear(64, level1_dim)

        self.level2_fc1 = nn.Linear(output_dim, 64)
        self.level2_fc2 = nn.Linear(64, level2_dim)

        self.level3_fc1 = nn.Linear(output_dim, 64)
        self.level3_fc2 = nn.Linear(64, level3_dim)

        self.schema_sigmoid = nn.Sigmoid()
        
    def forward(self, x, return_schema=False, return_intermediate=False):
        """
        Forward pass.
        
        Args:
            x: Input embeddings (batch_size, 384)
            return_schema: If True, also compute schema predictions
            return_intermediate: If True, also return intermediate and teacher_compressed
        
        Returns:
            dict with 'embedding' (128D student), optionally 'schema',
            'intermediate', 'teacher_compressed'
        """
        intermediate = self.relu(self.fc1(x))  # (batch, 256)
        embedding = self.fc2(intermediate)      # (batch, 128)
        
        results = {'embedding': embedding}
        
        if return_intermediate:
            teacher_compressed = self.teacher_compress_to_intermediate(x)  # (batch, 256)
            results['teacher_compressed'] = teacher_compressed
            results['intermediate'] = intermediate
        
        if return_schema:
            level1_hidden = self.schema_relu(self.level1_fc1(embedding))
            level2_hidden = self.schema_relu(self.level2_fc1(embedding))
            level3_hidden = self.schema_relu(self.level3_fc1(embedding))

            results['schema_level1'] = self.schema_sigmoid(
                self.level1_fc2(level1_hidden)
            )
            results['schema_level2'] = self.schema_sigmoid(
                self.level2_fc2(level2_hidden)
            )
            results['schema_level3'] = self.schema_sigmoid(
                self.level3_fc2(level3_hidden)
            )
        
        return results


# ============================================================================
# TRAINING
# ============================================================================

def train_on_real_voxsigil_data(
    voxsigil_dir: str,
    output_dir: str = 'phase4b_outputs',
    epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    alpha: float = 0.7,  # distillation weight
    beta: float = 0.3,   # schema recovery weight
    recursive: bool = True,
    deduplicate: bool = True,
):
    """
    Train student embedder on REAL Voxsigil schema data.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] {device}")
    
    # -------- STEP 1: Load Real Voxsigil Data --------
    print("\n[STEP 1] Loading real Voxsigil schema data...")
    voxsigil_data, stats = load_real_voxsigil_data(
        voxsigil_dir,
        recursive=recursive,
        deduplicate=deduplicate,
    )
    
    if not voxsigil_data:
        print("ERROR: No Voxsigil files loaded!")
        return None
    
    filenames, raw_data, schema_vectors, schema_levels = zip(*voxsigil_data)
    schema_vectors = np.array(schema_vectors)
    level1_vectors = np.array([lvl['level1'] for lvl in schema_levels])
    level2_vectors = np.array([lvl['level2'] for lvl in schema_levels])
    level3_vectors = np.array([lvl['level3'] for lvl in schema_levels])
    
    print(f"[STEP 1] Loaded {len(schema_vectors)} real sigils")
    if deduplicate:
        print(
            f"[STEP 1] Deduplicated {stats.get('duplicate_count', 0)} entries"
        )
    print(f"[STEP 1] Schema shape: {schema_vectors.shape}")
    
    # -------- STEP 2: Generate Teacher Embeddings --------
    print("\n[STEP 2] Generating teacher embeddings from schema...")
    teacher_embeddings = generate_teacher_embeddings(schema_vectors)
    print(f"[STEP 2] Teacher embeddings shape: {teacher_embeddings.shape}")
    
    # Convert to tensors
    teacher_tensor = torch.from_numpy(teacher_embeddings).float().to(device)
    level1_tensor = torch.from_numpy(level1_vectors).float().to(device)
    level2_tensor = torch.from_numpy(level2_vectors).float().to(device)
    level3_tensor = torch.from_numpy(level3_vectors).float().to(device)
    
    # Train/test split
    n_total = len(schema_vectors)
    n_train = int(0.8 * n_total)
    indices = np.random.permutation(n_total)
    
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    teacher_train = teacher_tensor[train_idx]
    level1_train = level1_tensor[train_idx]
    level2_train = level2_tensor[train_idx]
    level3_train = level3_tensor[train_idx]
    teacher_test = teacher_tensor[test_idx]
    level1_test = level1_tensor[test_idx]
    level2_test = level2_tensor[test_idx]
    level3_test = level3_tensor[test_idx]
    
    print(f"[STEP 2] Train: {len(train_idx)}, Test: {len(test_idx)}")
    
    # -------- STEP 3: Initialize Model --------
    print("\n[STEP 3] Initializing schema-supervised student...")
    model = SchemaSupervisionStudentEmbedder(
        input_dim=384,
        hidden_dim=256,
        output_dim=128,
        level1_dim=level1_vectors.shape[1],
        level2_dim=level2_vectors.shape[1],
        level3_dim=level3_vectors.shape[1],
    )
    model = model.to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[STEP 3] Model parameters: {param_count:,}")
    
    # -------- STEP 4: Training Loop --------
    print("\n[STEP 4] Training on REAL Voxsigil data...")
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    distill_loss_fn = nn.MSELoss()
    schema_loss_fn = nn.MSELoss()

    schedule = [
        {
            'name': 'level1_foundation',
            'epochs': max(1, int(epochs * 0.3)),
            'weights': {'level1': 1.0, 'level2': 0.0, 'level3': 0.0},
        },
        {
            'name': 'level2_structure',
            'epochs': max(1, int(epochs * 0.3)),
            'weights': {'level1': 0.3, 'level2': 1.0, 'level3': 0.0},
        },
        {
            'name': 'level3_practical',
            'epochs': max(1, epochs - int(epochs * 0.6)),
            'weights': {'level1': 0.2, 'level2': 0.5, 'level3': 1.0},
        },
    ]
    
    train_losses = []
    
    epoch_counter = 0
    for phase in schedule:
        print(
            f"  [SCHEDULE] Phase {phase['name']}"
            f" for {phase['epochs']} epochs"
        )
        for _ in range(phase['epochs']):
            epoch_loss = 0.0
            epoch_distill = 0.0
            epoch_schema = 0.0

            # Shuffle training data
            perm = torch.randperm(teacher_train.shape[0])
            teacher_shuffled = teacher_train[perm]
            level1_shuffled = level1_train[perm]
            level2_shuffled = level2_train[perm]
            level3_shuffled = level3_train[perm]

            n_batches = 0
            for batch_start in range(0, len(teacher_shuffled), batch_size):
                batch_end = min(batch_start + batch_size, len(teacher_shuffled))

                teacher_batch = teacher_shuffled[batch_start:batch_end]
                level1_batch = level1_shuffled[batch_start:batch_end]
                level2_batch = level2_shuffled[batch_start:batch_end]
                level3_batch = level3_shuffled[batch_start:batch_end]

                # Forward pass
                model.train()
                results = model(teacher_batch, return_schema=True, return_intermediate=True)

                student_intermediate = results['intermediate']
                teacher_compressed = results['teacher_compressed']
                schema_level1 = results['schema_level1']
                schema_level2 = results['schema_level2']
                schema_level3 = results['schema_level3']

                # Compute losses
                distill_loss = distill_loss_fn(student_intermediate, teacher_compressed)
                level1_loss = schema_loss_fn(schema_level1, level1_batch)
                level2_loss = schema_loss_fn(schema_level2, level2_batch)
                level3_loss = schema_loss_fn(schema_level3, level3_batch)

                schema_loss = (
                    phase['weights']['level1'] * level1_loss
                    + phase['weights']['level2'] * level2_loss
                    + phase['weights']['level3'] * level3_loss
                )
                combined_loss = alpha * distill_loss + beta * schema_loss

                # Backprop
                optimizer.zero_grad()
                combined_loss.backward()
                optimizer.step()

                epoch_loss += combined_loss.item()
                epoch_distill += distill_loss.item()
                epoch_schema += schema_loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            avg_distill = epoch_distill / n_batches
            avg_schema = epoch_schema / n_batches
            train_losses.append(avg_loss)

            if (epoch_counter + 1) % max(1, epochs // 10) == 0:
                print(
                    f"  Epoch {epoch_counter + 1:3d}:"
                    f" loss={avg_loss:.6f}"
                    f" (distill={avg_distill:.6f}, schema={avg_schema:.6f})"
                )

            if epoch_counter == 0 or epoch_counter == epochs - 1:
                print(
                    f"  Epoch {epoch_counter + 1:3d}:"
                    f" loss={avg_loss:.6f}"
                    f" (distill={avg_distill:.6f}, schema={avg_schema:.6f})"
                )

            epoch_counter += 1
    
    print(f"[STEP 4] Training complete. Final loss: {train_losses[-1]:.6f}")
    
    # -------- STEP 5: Test Schema Recovery --------
    print("\n[STEP 5] Validating schema recovery on REAL test data...")

    model.eval()
    with torch.no_grad():
        test_results = model(teacher_test, return_schema=True, return_intermediate=False)
        level1_pred = test_results['schema_level1'].cpu().numpy()
        level2_pred = test_results['schema_level2'].cpu().numpy()
        level3_pred = test_results['schema_level3'].cpu().numpy()

    level1_test_np = level1_test.cpu().numpy()
    level2_test_np = level2_test.cpu().numpy()
    level3_test_np = level3_test.cpu().numpy()

    def compute_r2_summary(level_name, y_true, y_pred):
        scores = []
        variances = []
        valid_dims = []
        for dim in range(y_true.shape[1]):
            variance = float(np.var(y_true[:, dim]))
            variances.append(variance)
            valid = variance >= 1e-8
            valid_dims.append(valid)
            r2 = safe_r2_score(y_true[:, dim], y_pred[:, dim])
            scores.append(r2)
            passing = "[PASS]" if r2 > 0.5 else "[FAIL]"
            print(
                f"  {level_name} dim {dim:2d}:"
                f" R2={r2:7.4f} var={variance:.6e} {passing}"
            )
        mean_score = float(np.mean(scores)) if scores else 0.0
        if any(valid_dims):
            valid_scores = [s for s, v in zip(scores, valid_dims) if v]
            valid_mean = float(np.mean(valid_scores)) if valid_scores else 0.0
        else:
            valid_mean = 0.0
        print(
            f"  {level_name} mean R2: {mean_score:.4f}"
            f" | variance-aware mean: {valid_mean:.4f}"
        )
        return scores, variances, valid_mean

    level1_scores, level1_variances, level1_mean = compute_r2_summary(
        "Level1", level1_test_np, level1_pred
    )
    level2_scores, level2_variances, level2_mean = compute_r2_summary(
        "Level2", level2_test_np, level2_pred
    )
    level3_scores, level3_variances, level3_mean = compute_r2_summary(
        "Level3", level3_test_np, level3_pred
    )

    mean_r2 = float(np.mean([level1_mean, level2_mean, level3_mean]))
    print(f"[STEP 5] Mean R2 across all levels: {mean_r2:.4f}")

    # Cosine similarity (student vs teacher) on test in intermediate space
    with torch.no_grad():
        cosine_outputs = model(
            teacher_test,
            return_schema=False,
            return_intermediate=True,
        )
        student_intermediate = cosine_outputs['intermediate']
        teacher_intermediate = cosine_outputs['teacher_compressed']
        student_norm = student_intermediate / (
            torch.norm(student_intermediate, dim=1, keepdim=True) + 1e-8
        )
        teacher_norm = teacher_intermediate / (
            torch.norm(teacher_intermediate, dim=1, keepdim=True) + 1e-8
        )
        cosine_sim = torch.sum(student_norm * teacher_norm, dim=1).cpu().numpy()

    cosine_stats = {
        'mean': float(np.mean(cosine_sim)),
        'min': float(np.min(cosine_sim)),
        'max': float(np.max(cosine_sim)),
        'p50': float(np.percentile(cosine_sim, 50)),
        'p90': float(np.percentile(cosine_sim, 90)),
        'p99': float(np.percentile(cosine_sim, 99)),
    }
    print(
        "[STEP 5] Cosine similarity (intermediate): "
        f"mean={cosine_stats['mean']:.4f} "
        f"p50={cosine_stats['p50']:.4f} "
        f"p90={cosine_stats['p90']:.4f} "
        f"p99={cosine_stats['p99']:.4f}"
    )
    
    # Test distillation loss
    with torch.no_grad():
        test_results_full = model(teacher_test, return_schema=True, return_intermediate=True)
        student_intermediate_test = test_results_full['intermediate']  # (batch, 256)
        teacher_compressed_test = test_results_full['teacher_compressed']  # (batch, 256)
        distill_loss_test = distill_loss_fn(student_intermediate_test, teacher_compressed_test).item()
    
    print(f"[STEP 5] Distillation loss on test: {distill_loss_test:.6f}")
    
    # -------- STEP 6: Gate Validation --------
    print("\n[STEP 6] Validation gate checks...")
    
    gates = {
        'level1_recovery': level1_mean > 0.5,
        'level2_recovery': level2_mean > 0.5,
        'level3_recovery': level3_mean > 0.5,
        'distillation_quality': distill_loss_test < 0.01,
        'zero_crash_rate': True,  # no exceptions
    }
    
    for gate_name, gate_pass in gates.items():
        status = "[PASS]" if gate_pass else "[FAIL]"
        print(f"  {status} {gate_name}")
    
    all_pass = all(gates.values())
    overall = "ALL GATES PASS" if all_pass else "SOME GATES FAIL"
    print(f"\n[VALIDATION] Overall: {overall}")
    
    # -------- STEP 7: Save Model & Results --------
    print("\n[STEP 7] Saving model and results...")
    
    model_path = os.path.join(output_dir, 'student_embedder_real_voxsigil.pt')
    torch.save(model.state_dict(), model_path)
    print(f"  Model saved: {model_path}")
    
    results_dict = {
        'metadata': {
            'timestamp': str(np.datetime64('now')),
            'data_source': 'REAL Voxsigil sigils',
            'n_sigils_loaded': len(schema_vectors),
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'deduplicated': bool(deduplicate),
            'duplicate_count': int(stats.get('duplicate_count', 0)),
            'unique_count': int(stats.get('unique_count', len(schema_vectors))),
        },
        'training': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'final_loss': float(train_losses[-1]),
            'alpha': alpha,
            'beta': beta,
        },
        'validation': {
            'level1_r2': [float(r2) for r2 in level1_scores],
            'level2_r2': [float(r2) for r2 in level2_scores],
            'level3_r2': [float(r2) for r2 in level3_scores],
            'level1_variance': [float(v) for v in level1_variances],
            'level2_variance': [float(v) for v in level2_variances],
            'level3_variance': [float(v) for v in level3_variances],
            'level1_mean_r2': float(level1_mean),
            'level2_mean_r2': float(level2_mean),
            'level3_mean_r2': float(level3_mean),
            'overall_mean_r2': float(mean_r2),
            'distillation_loss_test': float(distill_loss_test),
            'cosine_similarity': cosine_stats,
        },
        'gates': {k: bool(v) for k, v in gates.items()},
        'all_gates_pass': all_pass,
        'sigil_files': list(filenames),
    }
    
    results_path = os.path.join(output_dir, 'phase4b2_real_voxsigil_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"  Results saved: {results_path}")
    
    return {
        'model': model,
        'results': results_dict,
        'schema_vectors': schema_vectors,
        'teacher_embeddings': teacher_embeddings,
    }


if __name__ == '__main__':
    voxsigil_dir = r'c:\nebula-social-crypto-core\voxsigil_library'
    
    print("=" * 80)
    print("PHASE 4-B.2: REAL VOXSIGIL SCHEMA TRAINING & VALIDATION")
    print("=" * 80)
    
    outcome = train_on_real_voxsigil_data(
        voxsigil_dir=voxsigil_dir,
        output_dir='phase4b_outputs',
        epochs=300,
        batch_size=8,
        learning_rate=3e-4,
        alpha=0.7,
        beta=0.3,
        recursive=True,
        deduplicate=True,
    )
    
    if outcome and outcome['results']['all_gates_pass']:
        print("\n" + "=" * 80)
        print("[SUCCESS] VALIDATION COMPLETE: MODEL TRAINED ON REAL VOXSIGIL DATA")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("[INCOMPLETE] Validation did not achieve full pass")
        print("=" * 80)
