"""
Discover available Ollama models and generate Phase 6 benchmark report
"""

import requests
import json
import time
from datetime import datetime
from pathlib import Path

def get_available_models():
    """Query Ollama for installed models"""
    try:
        resp = requests.get('http://localhost:11434/api/tags', timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            models = [m['name'] for m in data.get('models', [])]
            return sorted(models)
    except:
        pass
    return []

def discover_and_report():
    """Discover available models and create initial report"""
    
    print("\n" + "="*80)
    print("🔍 DISCOVERING AVAILABLE OLLAMA MODELS FOR PHASE 6")
    print("="*80)
    
    models = get_available_models()
    
    if not models:
        print("❌ No Ollama models found. Is Ollama running?")
        print("   Start Ollama and pull models first:")
        print("   > ollama pull llama3.2:latest")
        print("   > ollama pull mistral:latest")
        print("   > ollama pull phi3:mini")
        return
    
    print(f"\n✅ Found {len(models)} available models:\n")
    for i, model in enumerate(models, 1):
        print(f"   {i}. {model}")
    
    # Create discovery report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("c:\\UBLT\\phase6_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    discovery_file = output_dir / f"phase6_model_discovery_{timestamp}.json"
    
    discovery_report = {
        "timestamp": datetime.now().isoformat(),
        "phase": "Phase 6 Model Discovery",
        "available_models": models,
        "model_count": len(models),
        "status": "ready_for_benchmark" if len(models) >= 3 else "insufficient_models",
        "next_step": "Run phase6_parallel_benchmarking.py with discovered models"
    }
    
    with open(discovery_file, 'w') as f:
        json.dump(discovery_report, f, indent=2)
    
    print(f"\n📋 Discovery report saved to: {discovery_file}")
    print("\n" + "="*80)
    print("✅ READY FOR PHASE 6 BENCHMARKING")
    print("="*80)
    print(f"Available models for benchmarking: {len(models)}")
    print(f"Minimum required: 3 models")
    print(f"Status: {'✓ READY' if len(models) >= 3 else '✗ INSUFFICIENT MODELS'}")
    print("="*80 + "\n")
    
    return models

if __name__ == "__main__":
    get_available_models()
    discover_and_report()
