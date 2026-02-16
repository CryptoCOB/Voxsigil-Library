"""
Analyze real Voxsigil schema to understand what we're actually trying to learn.
This will show us the distribution of schema characteristics across sigils.
"""

import yaml
import json
import numpy as np
from pathlib import Path
import glob

def extract_detailed_schema(sigil_data):
    """Extract more detailed schema characteristics."""
    
    meta = sigil_data.get('meta', {})
    holo = sigil_data.get('holo_mesh', {})
    cognitive = sigil_data.get('cognitive', {})
    impl = sigil_data.get('implementation', {})
    conn = sigil_data.get('connectivity', {})
    
    return {
        # Basic characteristics
        'is_primitive': holo.get('is_cognitive_primitive', False),
        'has_math': bool(cognitive.get('math')),
        'schema_version': meta.get('schema_version', 'unknown'),
        
        # Structure analysis
        'structure': cognitive.get('structure', {}),
        'composite_type': cognitive.get('structure', {}).get('composite_type', 'none'),
        'temporal_structure': cognitive.get('structure', {}).get('temporal_structure', 'static'),
        'n_components': len(cognitive.get('structure', {}).get('components', [])),
        
        # Tags and categorization
        'tags': cognitive.get('tags', []),
        'tag_count': len(cognitive.get('tags', [])),
        
        # Text richness
        'principle_len': len(cognitive.get('principle', '')),
        'math_len': len(cognitive.get('math', '')),
        
        # Features
        'has_activation': bool(impl.get('activation_context')),
        'has_usage': bool(impl.get('usage')),
        'has_params': bool(impl.get('parameterization_schema')),
        'has_relationships': bool(meta.get('relationships', [])) or bool(cognitive.get('relationships', [])),
        
        # Mesh properties
        'mesh_compatible': holo.get('mesh_compatibility', 'none'),
        'registration_ready': holo.get('registration_ready', False),
        'event_support': holo.get('vanta_core_integration', {}).get('event_support', False),
        'async_capable': holo.get('vanta_core_integration', {}).get('async_capable', False),
    }


def load_and_analyze(sigil_dir):
    """Load all sigils and analyze schema."""
    
    sigil_files = sorted(glob.glob(f"{sigil_dir}/*.voxsigil"))
    
    all_data = []
    
    for filepath in sigil_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                sigil_data = yaml.safe_load(f)
                if sigil_data:
                    filename = Path(filepath).stem
                    schema = extract_detailed_schema(sigil_data)
                    all_data.append({
                        'filename': filename,
                        'schema': schema,
                        'raw': sigil_data
                    })
        except Exception as e:
            print(f"Error loading {Path(filepath).name}: {e}")
    
    print(f"\n{'='*80}")
    print(f"VOXSIGIL SCHEMA ANALYSIS ({len(all_data)} sigils)")
    print(f"{'='*80}\n")
    
    # Analyze each dimension
    print("[1] PRIMITIVE CHARACTERISTIC")
    primitives = [d['schema']['is_primitive'] for d in all_data]
    print(f"    Cognitive primitives: {sum(primitives)}/{len(all_data)}")
    print(f"    Distribution: {[d['filename'] for d in all_data if d['schema']['is_primitive']]}\n")
    
    print("[2] MATHEMATICAL FORMALISM")
    with_math = [d for d in all_data if d['schema']['has_math']]
    print(f"    With math: {len(with_math)}/{len(all_data)}")
    print(f"    Examples: {[d['filename'] for d in with_math[:3]]}\n")
    
    print("[3] COMPOSITE TYPES")
    composite_types = {}
    for d in all_data:
        ct = d['schema']['composite_type']
        composite_types[ct] = composite_types.get(ct, 0) + 1
    for ct, count in sorted(composite_types.items(), key=lambda x: -x[1]):
        print(f"    {ct:<15s}: {count:3d}")
    print()
    
    print("[4] COMPONENT COUNTS")
    n_comps = [d['schema']['n_components'] for d in all_data]
    print(f"    Min: {min(n_comps)}, Max: {max(n_comps)}, Mean: {np.mean(n_comps):.2f}")
    print(f"    Distribution:")
    for c in sorted(set(n_comps)):
        count = sum(1 for x in n_comps if x == c)
        bar = '█' * (count // 2)
        print(f"      {c} components: {bar} ({count})")
    print()
    
    print("[5] TAG DIVERSITY")
    tag_counts = [d['schema']['tag_count'] for d in all_data]
    print(f"    Min: {min(tag_counts)}, Max: {max(tag_counts)}, Mean: {np.mean(tag_counts):.2f}\n")
    
    print("[6] TEXT RICHNESS (Principle)")
    principle_lens = [d['schema']['principle_len'] for d in all_data]
    print(f"    Min: {min(principle_lens)}, Max: {max(principle_lens)}, Mean: {np.mean(principle_lens):.0f}\n")
    
    print("[7] FEATURES")
    feats = {
        'has_activation': sum(1 for d in all_data if d['schema']['has_activation']),
        'has_usage': sum(1 for d in all_data if d['schema']['has_usage']),
        'has_params': sum(1 for d in all_data if d['schema']['has_params']),
        'has_relationships': sum(1 for d in all_data if d['schema']['has_relationships']),
        'event_support': sum(1 for d in all_data if d['schema']['event_support']),
        'async_capable': sum(1 for d in all_data if d['schema']['async_capable']),
    }
    for feat, count in sorted(feats.items(), key=lambda x: -x[1]):
        print(f"    {feat:<20s}: {count:2d}/{len(all_data)}")
    print()
    
    print("[8] TEMPORAL PATTERNS")
    temporal = {}
    for d in all_data:
        tp = d['schema']['temporal_structure']
        temporal[tp] = temporal.get(tp, 0) + 1
    for tp, count in sorted(temporal.items(), key=lambda x: -x[1]):
        print(f"    {tp:<20s}: {count:2d}")
    print()
    
    # Show some example sigils with different characteristics
    print("[9] CHARACTERISTIC EXAMPLES")
    
    # Most complex
    most_complex = max(all_data, key=lambda d: d['schema']['n_components'])
    print(f"\n    Most complex (components): {most_complex['filename']}")
    print(f"      Components: {most_complex['schema']['n_components']}")
    print(f"      Type: {most_complex['schema']['composite_type']}")
    
    # Richest text
    richest = max(all_data, key=lambda d: d['schema']['principle_len'])
    print(f"\n    Richest principle: {richest['filename']}")
    print(f"      Principle length: {richest['schema']['principle_len']} chars")
    
    # Most featured
    most_featured = max(all_data, key=lambda d: sum([
        d['schema']['has_activation'],
        d['schema']['has_usage'],
        d['schema']['has_params'],
        d['schema']['event_support'],
    ]))
    print(f"\n    Most featured: {most_featured['filename']}")
    print(f"      Features: activation={most_featured['schema']['has_activation']}, "
          f"usage={most_featured['schema']['has_usage']}, "
          f"params={most_featured['schema']['has_params']}")
    
    # Minimal
    minimal = min(all_data, key=lambda d: sum([
        d['schema']['is_primitive'],
        d['schema']['has_math'],
        d['schema']['n_components'],
        d['schema']['has_activation'],
    ]))
    print(f"\n    Most minimal: {minimal['filename']}")
    print(f"      Primitive: {minimal['schema']['is_primitive']}, "
          f"Math: {minimal['schema']['has_math']}, "
          f"Components: {minimal['schema']['n_components']}")
    
    return all_data


if __name__ == '__main__':
    sigil_dir = r'c:\nebula-social-crypto-core\voxsigil_library\library_sigil\sigils'
    data = load_and_analyze(sigil_dir)
    
    # Save analysis
    with open('voxsigil_schema_analysis.json', 'w') as f:
        json.dump({
            'total_sigils': len(data),
            'sample_schemas': [{'file': d['filename'], 'schema': d['schema']} for d in data[:5]]
        }, f, indent=2)
