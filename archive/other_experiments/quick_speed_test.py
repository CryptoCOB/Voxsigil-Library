"""
Quick Speed Test - Find fastest Ollama model
Just measure tokens/second, pick the winner
"""

import requests
import time
from datetime import datetime

MODELS = [
    "mistral:latest",
    "phi3:mini",
    "qwen2:7b",
    "llama3:8b",
    "deepseek-coder:6.7b",
    "wizard-math:latest",
    "mathstral:latest",
    "kimi-k2.5:cloud",
    "gpt-oss:20b",
]

OLLAMA_BASE_URL = "http://localhost:11434"

TEST_PROMPT = """Generate a behavioral profile with these attributes:
- Focus level (0-1)
- Adaptability (0-1)
- Collaboration (0-1)
- Innovation (0-1)
Provide numerical scores."""

def test_speed(model: str) -> dict:
    """Test single model speed"""
    print(f"⚡ Testing {model}...", end=" ", flush=True)
    
    try:
        start = time.time()
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": TEST_PROMPT,
                "stream": False,
                "options": {"num_predict": 256, "temperature": 0.7}
            },
            timeout=60
        )
        elapsed = time.time() - start
        
        data = response.json()
        tokens = data.get("eval_count", 0)
        tps = tokens / elapsed if elapsed > 0 else 0
        
        print(f"{tps:.1f} tok/s ({elapsed:.2f}s)")
        return {"model": model, "tokens_per_sec": tps, "time": elapsed, "tokens": tokens, "success": True}
    except Exception as e:
        print(f"❌ {str(e)[:50]}")
        return {"model": model, "tokens_per_sec": 0, "time": 0, "tokens": 0, "success": False}

def main():
    print("\n" + "="*60)
    print("⚡ QUICK SPEED TEST - Finding Fastest Model")
    print("="*60)
    
    results = []
    for model in MODELS:
        result = test_speed(model)
        if result["success"]:
            results.append(result)
    
    if not results:
        print("\n❌ No successful tests")
        return
    
    results.sort(key=lambda x: x["tokens_per_sec"], reverse=True)
    
    print("\n" + "="*60)
    print("📊 RESULTS (Sorted by Speed)")
    print("="*60)
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['model']:<30} {r['tokens_per_sec']:>8.1f} tok/s")
    
    winner = results[0]
    print("\n" + "="*60)
    print(f"🏆 FASTEST MODEL: {winner['model']}")
    print(f"   Speed: {winner['tokens_per_sec']:.1f} tokens/second")
    print("="*60)
    print(f"\n✅ Use {winner['model']} for fast BLT testing\n")

if __name__ == "__main__":
    main()
