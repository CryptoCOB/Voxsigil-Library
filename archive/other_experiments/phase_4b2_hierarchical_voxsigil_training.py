"""
Phase 4-B.2: Hierarchical VoxSigil Schema Training
===================================================

Trains student embedder on REAL Voxsigil data with 3-level hierarchical schema:
  Level 1: Core identification (primitive, tags, version)
  Level 2: Structure (composite type, components, temporal, math)
  Level 3: Advanced (activation, usage, SMART_MRAP, relationships)

Each level builds on previous. Multi-task learning at each level.
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
import io

# UTF-8 support on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ============================================================================
# HIERARCHICAL SCHEMA EXTRACTION (3 LEVELS)
# ============================================================================

def extract_level1_schema(sigil_data: Dict) -> np.ndarray:
    """
    Level 1: Core identification (4D)
    - 0: is_cognitive_primitive (0/1)
    - 1: tag_count (normalized 0-1)
    - 2: has_mathematical_basis (0/1)
    - 3: schema_maturity (1.4-alpha=0.4, 1.5-holo=0.8)
    """
    schema = np.zeros(4, dtype=np.float32)
    
    holo = sigil_data.get('holo_mesh', {})
    cognitive = sigil_data.get('cognitive', {})
    meta = sigil_data.get('meta', {})
    
    # Dimension 0: is_cognitive_primitive
    schema[0] = 1.0 if holo.get('is_cognitive_primitive') else 0.0
    
    # Dimension 1: tag_count (normalized)
    tags = cognitive.get('tags', [])
    schema[1] = min(len(tags) / 10.0, 1.0)
    
    # Dimension 2: has_mathematical_basis
    schema[2] = 1.0 if cognitive.get('math') else 0.0
    
    # Dimension 3: schema_maturity
    schema_version = meta.get('schema_version', '1.4')
    if '1.5' in schema_version:
        schema[3] = 0.8
    else:
        schema[3] = 0.4
    
    return schema


def extract_level2_schema(sigil_data: Dict) -> np.ndarray:
    """
    Level 2: Structure & Conceptual (8D)
    - 0-5: composite_type (one-hot encoded as 6D)
    - 6: component_count (0-1 normalized)
    - 7: temporal_dynamics (0-1: static=0, dynamic=1)
    """
    schema = np.zeros(8, dtype=np.float32)
    
    cognitive = sigil_data.get('cognitive', {})
    structure = cognitive.get('structure', {})
    
    # Dimensions 0-5: composite_type (6-class one-hot)
    composite_map = {
        'sequential': 0,
        'hierarchical': 1,
        'network': 2,
        'recursive': 3,
        'parallel': 4,
        'other': 5,  # catch-all
    }
    composite = structure.get('composite_type', 'other').lower()
    if 'recursive' in composite:
        composite_idx = 3
    elif 'parallel' in composite:
        composite_idx = 4
    elif 'network' in composite:
        composite_idx = 2
    elif 'hierarchical' in composite:
        composite_idx = 1
    elif 'sequential' in composite:
        composite_idx = 0
    else:
        composite_idx = 5
    
    schema[composite_idx] = 1.0
    
    # Dimension 6: component_count (normalized)
    components = structure.get('components', [])
    schema[6] = min(len(components) / 15.0, 1.0)
    
    # Dimension 7: temporal_dynamics
    temporal = structure.get('temporal_structure', '').lower()
    is_dynamic = any(kw in temporal for kw in ['feedback', 'loop', 'event', 'continuous', 'iterative', 'dynamic'])
    schema[7] = 1.0 if is_dynamic else 0.0
    
    return schema


def extract_level3_schema(sigil_data: Dict) -> np.ndarray:
    """
    Level 3: Advanced (10D)
    - 0: has_activation_context (0/1)
    - 1: has_usage_spec (0/1)
    - 2: has_parameterization (0/1)
    - 3: has_relationships (0/1)
    - 4: principle_richness (0-1: text length normalized)
    - 5: event_support (0/1)
    - 6: async_capable (0/1)
    - 7: registration_ready (0/1)
    - 8: has_prompt_template (0/1)
    - 9: mesh_compatibility (0-1: level of holo support)
    """
    schema = np.zeros(10, dtype=np.float32)
    
    impl = sigil_data.get('implementation', {})
    cognitive = sigil_data.get('cognitive', {})
    holo = sigil_data.get('holo_mesh', {})
    meta = sigil_data.get('meta', {})
    
    # Dimension 0: has_activation_context
    schema[0] = 1.0 if impl.get('activation_context') else 0.0
    
    # Dimension 1: has_usage_spec
    schema[1] = 1.0 if impl.get('usage') else 0.0
    
    # Dimension 2: has_parameterization
    schema[2] = 1.0 if impl.get('parameterization_schema') else 0.0
    
    # Dimension 3: has_relationships
    schema[3] = 1.0 if cognitive.get('relationships') else 0.0
    
    # Dimension 4: principle_richness
    principle = cognitive.get('principle', '')
    schema[4] = min(len(principle) / 700.0, 1.0)
    
    # Dimension 5: event_support
    vanta = holo.get('vanta_core_integration', {})
    schema[5] = 1.0 if vanta.get('event_support') else 0.0
    
    # Dimension 6: async_capable
    schema[6] = 1.0 if vanta.get('async_capable') else 0.0
    
    # Dimension 7: registration_ready
    schema[7] = 1.0 if holo.get('registration_ready') else 0.0
    
    # Dimension 8: has_prompt_template
    schema[8] = 1.0 if impl.get('prompt_template') else 0.0
    
    # Dimension 9: mesh_compatibility
    mesh_compat = holo.get('mesh_compatibility', '')
    if '1.5' in mesh_compat:
        schema[9] = 1.0
    elif '1.4' in mesh_compat:
        schema[9] = 0.7
    else:
        schema[9] = 0.0
    
    return schema


def load_hierarchical_voxsigil_data(sigil_dir: str) -> List[Tuple[str, Dict, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load all .voxsigil files and extract hierarchical schema (Level 1, 2, 3).
    Returns list of (filename, raw_data, level1, level2, level3).
    """
    
    data = []
    sigil_files = glob.glob(os.path.join(sigil_dir, '*.voxsigil'))
    
    print(f"[LOAD] Found {len(sigil_files)} .voxsigil files")
    
    for idx, filepath in enumerate(sigil_files):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                sigil_data = yaml.safe_load(f)
                if sigil_data:
                    l1 = extract_level1_schema(sigil_data)
                    l2 = extract_level2_schema(sigil_data)
                    l3 = extract_level3_schema(sigil_data)
                    filename = os.path.basename(filepath)
                    data.append((filename, sigil_data, l1, l2, l3))
                    
                    # ASCII-safe output
                    print(f"  [{idx+1:2d}] {filename:<35s} | L1={l1[0]:.0f} L2comp={np.argmax(l2[:6])} L3act={l3[0]:.0f}")
        except Exception as e:
            error_msg = str(e)[:60]
            print(f"  [ERROR] {os.path.basename(filepath)}: {error_msg}")
    
    return data


def generate_hierarchical_teacher_embeddings(
    level1_vectors: np.ndarray,
    level2_vectors: np.ndarray,
    level3_vectors: np.ndarray
) -> np.ndarray:
    """
    Generate teacher embeddings by concatenating hierarchical schema projections.
    
    L1 (4D) -> 64D block
    L2 (8D) -> 160D block
    L3 (10D) -> 160D block
    Total: 384D
    """
    n_samples = level1_vectors.shape[0]
    teacher_embeddings = np.zeros((n_samples, 384), dtype=np.float32)
    
    for i in range(n_samples):
        # L1 -> 64D block (indices 0-63)
        l1 = level1_vectors[i]
        for j, val in enumerate(l1):
            block_idx = j * 16
            for k in range(16):
                teacher_embeddings[i, block_idx + k] = (
                    np.sin((k + 1) * val * np.pi) * (val + 0.5)
                )
        
        # L2 -> 160D block (indices 64-223)
        l2 = level2_vectors[i]
        for j, val in enumerate(l2):
            block_idx = 64 + j * 20
            for k in range(20):
                teacher_embeddings[i, block_idx + k] = (
                    np.cos((k + 1) * val * np.pi) * (val + 0.5)
                )
        
        # L3 -> 160D block (indices 224-383)
        l3 = level3_vectors[i]
        for j, val in enumerate(l3):
            block_idx = 224 + j * 16
            for k in range(16):
                teacher_embeddings[i, block_idx + k] = (
                    np.sin((k + 1) * val * np.pi) + np.cos((k + 2) * val * np.pi)
                ) * (val + 0.5)
    
    # L2 normalize
    teacher_embeddings = teacher_embeddings / (
        np.linalg.norm(teacher_embeddings, axis=1, keepdims=True) + 1e-8
    )
    
    return teacher_embeddings


# ============================================================================
# MODEL ARCHITECTURE WITH HIERARCHICAL HEADS
# ============================================================================

class HierarchicalSchemaStudentEmbedder(nn.Module):
    """Student with three hierarchical schema recovery heads."""
    
    def __init__(self, input_dim=384, hidden_dim=256, output_dim=128):
        super().__init__()
        
        # Main compression
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Teacher compression (distillation)
        self.teacher_compress = nn.Linear(input_dim, hidden_dim)
        
        # Level 1 schema head (4D)
        self.l1_fc1 = nn.Linear(output_dim, 32)
        self.l1_relu = nn.ReLU()
        self.l1_fc2 = nn.Linear(32, 4)
        self.l1_sigmoid = nn.Sigmoid()
        
        # Level 2 schema head (8D)
        self.l2_fc1 = nn.Linear(output_dim, 48)
        self.l2_relu = nn.ReLU()
        self.l2_fc2 = nn.Linear(48, 8)
        self.l2_sigmoid = nn.Sigmoid()
        
        # Level 3 schema head (10D)
        self.l3_fc1 = nn.Linear(output_dim, 64)
        self.l3_relu = nn.ReLU()
        self.l3_fc2 = nn.Linear(64, 10)
        self.l3_sigmoid = nn.Sigmoid()
    
    def forward(self, x, return_schema=False, return_intermediate=False):
        """Forward pass with optional schema and intermediate returns."""
        intermediate = self.relu(self.fc1(x))
        embedding = self.fc2(intermediate)
        
        results = {'embedding': embedding}
        
        if return_intermediate:
            teacher_compressed = self.teacher_compress(x)
            results['teacher_compressed'] = teacher_compressed
            results['intermediate'] = intermediate
        
        if return_schema:
            l1_hidden = self.l1_relu(self.l1_fc1(embedding))
            l1_pred = self.l1_sigmoid(self.l1_fc2(l1_hidden))
            
            l2_hidden = self.l2_relu(self.l2_fc1(embedding))
            l2_pred = self.l2_sigmoid(self.l2_fc2(l2_hidden))
            
            l3_hidden = self.l3_relu(self.l3_fc1(embedding))
            l3_pred = self.l3_sigmoid(self.l3_fc2(l3_hidden))
            
            results['schema_l1'] = l1_pred
            results['schema_l2'] = l2_pred
            results['schema_l3'] = l3_pred
        
        return results


# ============================================================================
# TRAINING
# ============================================================================

def train_hierarchical_on_voxsigil(
    voxsigil_dir: str,
    output_dir: str = 'phase4b_outputs',
    epochs: int = 150,
    batch_size: int = 4,
    learning_rate: float = 5e-4,
):
    """Train student with hierarchical schema recovery."""
    
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] {device}\n")
    
    # -------- STEP 1: Load Hierarchical Data --------
    print("[STEP 1] Loading hierarchical Voxsigil schema data...")
    voxsigil_data = load_hierarchical_voxsigil_data(voxsigil_dir)
    
    if not voxsigil_data:
        print("ERROR: No Voxsigil files loaded!")
        return None
    
    filenames, raw_data, l1_vecs, l2_vecs, l3_vecs = zip(*voxsigil_data)
    l1_vecs = np.array(l1_vecs)
    l2_vecs = np.array(l2_vecs)
    l3_vecs = np.array(l3_vecs)
    
    print(f"[STEP 1] Loaded {len(l1_vecs)} real sigils")
    print(f"         L1 shape: {l1_vecs.shape}, L2 shape: {l2_vecs.shape}, L3 shape: {l3_vecs.shape}\n")
    
    # -------- STEP 2: Generate Teacher Embeddings --------
    print("[STEP 2] Generating hierarchical teacher embeddings...")
    teacher_embeddings = generate_hierarchical_teacher_embeddings(l1_vecs, l2_vecs, l3_vecs)
    print(f"[STEP 2] Teacher embeddings shape: {teacher_embeddings.shape}\n")
    
    # Convert to tensors
    teacher_tensor = torch.from_numpy(teacher_embeddings).float().to(device)
    l1_tensor = torch.from_numpy(l1_vecs).float().to(device)
    l2_tensor = torch.from_numpy(l2_vecs).float().to(device)
    l3_tensor = torch.from_numpy(l3_vecs).float().to(device)
    
    # Train/test split
    n_total = len(l1_vecs)
    n_train = int(0.8 * n_total)
    indices = np.random.permutation(n_total)
    
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    teacher_train = teacher_tensor[train_idx]
    l1_train, l2_train, l3_train = l1_tensor[train_idx], l2_tensor[train_idx], l3_tensor[train_idx]
    
    teacher_test = teacher_tensor[test_idx]
    l1_test, l2_test, l3_test = l1_tensor[test_idx], l2_tensor[test_idx], l3_tensor[test_idx]
    
    print(f"[STEP 2] Train: {len(train_idx)}, Test: {len(test_idx)}\n")
    
    # -------- STEP 3: Initialize Model --------
    print("[STEP 3] Initializing hierarchical schema-supervised student...")
    model = HierarchicalSchemaStudentEmbedder(input_dim=384, hidden_dim=256, output_dim=128)
    model = model.to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[STEP 3] Model parameters: {param_count:,}\n")
    
    # -------- STEP 4: Training Loop --------
    print("[STEP 4] Multi-task hierarchical training...")
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    distill_loss_fn = nn.MSELoss()
    l1_loss_fn = nn.MSELoss()
    l2_loss_fn = nn.MSELoss()
    l3_loss_fn = nn.MSELoss()
    
    # Loss weights
    w_distill = 0.4
    w_l1 = 0.2
    w_l2 = 0.2
    w_l3 = 0.2
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        perm = torch.randperm(teacher_train.shape[0])
        teacher_shuffled = teacher_train[perm]
        l1_shuffled = l1_train[perm]
        l2_shuffled = l2_train[perm]
        l3_shuffled = l3_train[perm]
        
        for batch_start in range(0, len(teacher_shuffled), batch_size):
            batch_end = min(batch_start + batch_size, len(teacher_shuffled))
            
            teacher_batch = teacher_shuffled[batch_start:batch_end]
            l1_batch = l1_shuffled[batch_start:batch_end]
            l2_batch = l2_shuffled[batch_start:batch_end]
            l3_batch = l3_shuffled[batch_start:batch_end]
            
            model.train()
            results = model(teacher_batch, return_schema=True, return_intermediate=True)
            
            # Distillation loss
            student_intermediate = results['intermediate']
            teacher_compressed = results['teacher_compressed']
            distill_loss = distill_loss_fn(student_intermediate, teacher_compressed)
            
            # Hierarchical schema losses
            l1_loss = l1_loss_fn(results['schema_l1'], l1_batch)
            l2_loss = l2_loss_fn(results['schema_l2'], l2_batch)
            l3_loss = l3_loss_fn(results['schema_l3'], l3_batch)
            
            # Combined loss
            combined_loss = (w_distill * distill_loss + 
                           w_l1 * l1_loss + 
                           w_l2 * l2_loss + 
                           w_l3 * l3_loss)
            
            optimizer.zero_grad()
            combined_loss.backward()
            optimizer.step()
            
            epoch_loss += combined_loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        
        if (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"  Epoch {epoch+1:3d}: combined_loss={avg_loss:.6f}")
    
    print(f"\n[STEP 4] Training complete. Final loss: {avg_loss:.6f}\n")
    
    # -------- STEP 5: Test Validation --------
    print("[STEP 5] Validating hierarchical schema recovery...")
    
    model.eval()
    with torch.no_grad():
        test_results = model(teacher_test, return_schema=True, return_intermediate=True)
        
        l1_pred = test_results['schema_l1'].cpu().numpy()
        l2_pred = test_results['schema_l2'].cpu().numpy()
        l3_pred = test_results['schema_l3'].cpu().numpy()
        
        l1_test_np = l1_test.cpu().numpy()
        l2_test_np = l2_test.cpu().numpy()
        l3_test_np = l3_test.cpu().numpy()
    
    # Per-level R2 scores
    print("\n[Level 1 - Core Identification]")
    l1_r2_scores = []
    l1_names = ['primitive', 'tag_count', 'has_math', 'schema_maturity']
    for dim in range(4):
        r2 = r2_score(l1_test_np[:, dim], l1_pred[:, dim])
        l1_r2_scores.append(r2)
        passing = "[PASS]" if r2 > 0.3 else "[FAIL]"
        print(f"  Dim {dim} ({l1_names[dim]:<18s}): R2={r2:7.4f} {passing}")
    
    print("\n[Level 2 - Structure & Conceptual]")
    l2_r2_scores = []
    l2_names = ['composite_0', 'composite_1', 'composite_2', 'composite_3', 
                'composite_4', 'composite_5', 'component_count', 'temporal_dynamics']
    for dim in range(8):
        r2 = r2_score(l2_test_np[:, dim], l2_pred[:, dim])
        l2_r2_scores.append(r2)
        passing = "[PASS]" if r2 > 0.3 else "[FAIL]"
        print(f"  Dim {dim} ({l2_names[dim]:<18s}): R2={r2:7.4f} {passing}")
    
    print("\n[Level 3 - Advanced Cognitive]")
    l3_r2_scores = []
    l3_names = ['activation', 'usage', 'parameterization', 'relationships', 
                'principle_rich', 'event_support', 'async', 'registration', 'prompt', 'mesh_compat']
    for dim in range(10):
        r2 = r2_score(l3_test_np[:, dim], l3_pred[:, dim])
        l3_r2_scores.append(r2)
        passing = "[PASS]" if r2 > 0.3 else "[FAIL]"
        print(f"  Dim {dim} ({l3_names[dim]:<18s}): R2={r2:7.4f} {passing}")
    
    # Aggregate metrics
    all_r2 = l1_r2_scores + l2_r2_scores + l3_r2_scores
    mean_r2 = np.mean(all_r2)
    passing_dims = sum(1 for r2 in all_r2 if r2 > 0.3)
    
    print(f"\n[Aggregate] Mean R2: {mean_r2:.4f}, Passing dims: {passing_dims}/22")
    
    # -------- STEP 6: Gates --------
    print("\n[STEP 6] Validation gate checks...")
    
    gates = {
        'level1_recovery': np.mean(l1_r2_scores) > 0.4,
        'level2_recovery': np.mean(l2_r2_scores) > 0.3,
        'level3_recovery': np.mean(l3_r2_scores) > 0.3,
        'overall_schema': mean_r2 > 0.35,
        'passing_threshold': passing_dims >= 15,
    }
    
    for gate_name, gate_pass in gates.items():
        status = "[PASS]" if gate_pass else "[FAIL]"
        print(f"  {status} {gate_name}")
    
    all_pass = all(gates.values())
    overall = "ALL GATES PASS" if all_pass else "SOME GATES FAIL"
    print(f"\n[VALIDATION] Overall: {overall}")
    
    # -------- STEP 7: Save --------
    print("\n[STEP 7] Saving model and results...")
    
    model_path = os.path.join(output_dir, 'student_embedder_hierarchical_voxsigil.pt')
    torch.save(model.state_dict(), model_path)
    print(f"  Model saved: {model_path}")
    
    results_dict = {
        'metadata': {
            'timestamp': str(np.datetime64('now')),
            'data_source': 'REAL Voxsigil sigils (35)',
            'n_train': len(train_idx),
            'n_test': len(test_idx),
            'schema_levels': 3,
            'total_schema_dims': 22,
        },
        'training': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'loss_weights': {
                'distillation': w_distill,
                'level1': w_l1,
                'level2': w_l2,
                'level3': w_l3,
            },
        },
        'validation': {
            'level1': {
                'r2_scores': [float(r2) for r2 in l1_r2_scores],
                'mean_r2': float(np.mean(l1_r2_scores)),
                'passing_dims': sum(1 for r2 in l1_r2_scores if r2 > 0.3),
            },
            'level2': {
                'r2_scores': [float(r2) for r2 in l2_r2_scores],
                'mean_r2': float(np.mean(l2_r2_scores)),
                'passing_dims': sum(1 for r2 in l2_r2_scores if r2 > 0.3),
            },
            'level3': {
                'r2_scores': [float(r2) for r2 in l3_r2_scores],
                'mean_r2': float(np.mean(l3_r2_scores)),
                'passing_dims': sum(1 for r2 in l3_r2_scores if r2 > 0.3),
            },
            'overall_mean_r2': float(mean_r2),
            'total_passing_dims': int(passing_dims),
        },
        'gates': {k: bool(v) for k, v in gates.items()},
        'all_gates_pass': all_pass,
        'sigil_files': list(filenames),
    }
    
    results_path = os.path.join(output_dir, 'phase4b2_hierarchical_voxsigil_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"  Results saved: {results_path}")
    
    return {
        'model': model,
        'results': results_dict,
        'all_r2_scores': all_r2,
    }


if __name__ == '__main__':
    voxsigil_dir = r'c:\nebula-social-crypto-core\voxsigil_library\library_sigil\sigils'
    
    print("=" * 80)
    print("PHASE 4-B.2: HIERARCHICAL VOXSIGIL SCHEMA TRAINING & VALIDATION")
    print("=" * 80 + "\n")
    
    outcome = train_hierarchical_on_voxsigil(
        voxsigil_dir=voxsigil_dir,
        output_dir='phase4b_outputs',
        epochs=150,
        batch_size=4,
        learning_rate=5e-4,
    )
    
    if outcome and outcome['results']['all_gates_pass']:
        print("\n" + "=" * 80)
        print("[SUCCESS] MODEL TRAINED ON REAL HIERARCHICAL VOXSIGIL SCHEMA")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("[INCOMPLETE] Validation gates did not all pass")
        print("=" * 80)
