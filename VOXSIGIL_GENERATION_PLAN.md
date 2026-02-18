# VoxSigil Generation & Evaluation Plan (v0.1)

## Objectives
- Generate a large-scale, legally valid VoxSigil dataset aligned with schema 2.0-omega.
- Validate every sigil before training (hard fail on invalid).
- Evaluate baseline vs Llama generation before scaling to full dataset.

## Scaled Dataset Targets (10×)
| Sigil Type | Previous Robust | 10× Target | Purpose |
| --- | --- | --- | --- |
| Primitive / Atomic | 256–512 | 2,560–5,120 | Category grounding, glyph semantics |
| Standard Organism | 2,000–4,000 | 20,000–40,000 | Core symbolic grammar |
| Flow (ordered / causal) | 1,000–2,000 | 10,000–20,000 | Process + prediction logic |
| Assembly / Meta | 500–1,000 | 5,000–10,000 | Orchestration + non-acting structures |
| Mutation Variants | 1,000–2,000 | 10,000–20,000 | Entropy + evolution rules |

**Total Target:** 47,500–95,000 sigils

## Distribution Rule (Required)
- 70% canonical / stable
- 20% edge-case but valid
- 10% near-boundary (almost invalid but legal)

## Generation Directive (Payload)
Use this as the generator instruction for Llama/Ollama:

```
Objective:
Generate a large-scale, legally valid VoxSigil dataset for training BLT compression and VML symbolic grammar.

Generation Targets
Total sigils: 50k–100k

Distribution by type:
Primitive / Atomic: ~5k
Standard Organism: ~30k
Flow (ordered): ~15k
Assembly / Meta: ~7k
Mutation Variants: ~15k

Mandatory Constraints
Every sigil MUST declare:
scaffold_type
tags (domain, function, polarity, temporal, epistemic, lifecycle)

Glyph count per sigil:
Minimum: 1
Maximum: 11

Category limits enforced:
NOETIC ≤ 2
PHYSICS ≤ 2
LOGIC ≤ 2
ASTRAL ≤ 1
STRUCTURAL ≤ 2 (paired)
ENTROPY ≤ 1 (≤2 only for omega)
EMERGENCE (𐑒) ≤ 1

Flow scaffolds MUST be ordered and non-emergent.
Assembly scaffolds MUST NOT act directly.
Mutation variants MUST differ by ENTROPY-mediated change only.
All sigils MUST pass validation or be discarded.

Diversity Rules
70% canonical
20% edge-case
10% near-boundary legal

Output Format
Canonical JSON
Deterministic ordering
Hashable
One sigil per record
Do NOT generate prose explanations.
Do NOT invent new glyph categories.
Do NOT violate scaffold-tag compatibility.
```

## Workflow (Clean + Organized)
1. **Generate (Llama, parallel)**
   - Script: `scripts/training/generate_voxsigil_corpus.py`
2. **Validate + Normalize (hard gate)**
   - Uses `normalize_to_2_0_omega` + `validate_interconnected_schema`.
3. **Smoke Test (small sample)**
   - Script: `scripts/training/evaluate_sigil_generation.py`
4. **Scale to full dataset**
   - Run generator with full counts.
5. **Train and benchmark**
   - Use existing VME benchmarks in `benchmarks/`.

## Commands
### 1) Smoke test (baseline vs Llama)

```bash
PYTHONPATH=C:\UBLT;C:\UBLT\src python scripts/training/evaluate_sigil_generation.py --count 30 --model llama3.2:latest
```

If Ollama is not running yet, run baseline-only:

```bash
PYTHONPATH=C:\UBLT;C:\UBLT\src python scripts/training/evaluate_sigil_generation.py --count 30 --skip-llama
```

### 2) Generate full corpus (parallel)

```bash
PYTHONPATH=C:\UBLT\src python scripts/training/generate_voxsigil_corpus.py \
   --model llama3.2:latest \
   --workers 6 \
   --primitive 5000 \
   --standard 30000 \
   --flow 15000 \
   --assembly 7000 \
   --mutation 15000
```

### 3) Build training dataset (with validation gate)

```bash
python scripts/training/build_voxsigil_training_dataset.py \
   --repo-root C:/UBLT \
   --output-dir training/datasets \
   --target-size 36000 \
   --shard-size 4000 \
   --llama-prepass \
   --require-valid
```

## Notes

- Llama/Ollama parallel generation is used for speed (`--workers`).
- Validation is enforced before training artifacts are produced.
- Tag order is preserved; glyph ordering is preserved within scaffold types.

---

## Symbolic RAG Middleware Directive

**Status**: REQUIRED — This layer is mandatory for long-term memory and symbolic continuity.

### Architecture

```
[ VME Orchestration ]
          ↓
[ SymbolicRAGMiddleware ]  ← src/voxsigil_library/rag/
          ↓
[ BLT Compression / Validation ]
          ↓
[ Model / Agent / Generator ]
```

### Why It Exists

Without RAG, BLT is a language compression engine.
With RAG, BLT becomes a symbolic cognition engine.

| Without RAG | With RAG |
|---|---|
| Static symbolic structure only | Long-horizon memory |
| No sigil recall | Schema-grounded retrieval |
| No lineage reinforcement | Lineage continuity |
| In-context only | Cross-session symbolic persistence |

### Components Built

| File | Role |
|---|---|
| `src/voxsigil_library/rag/embedder.py` | Deterministic 768D sigil embeddings |
| `src/voxsigil_library/rag/retriever.py` | `SigilRetriever` interface + FAISS backend |
| `src/voxsigil_library/rag/blt_bridge.py` | BLT compress / score / filter |
| `src/voxsigil_library/rag/middleware.py` | Full retrieve → enrich → BLT_validate → generate loop |

### Embedding Dimensions

Each sigil is embedded as a **768D unit-normalized vector** composed of three 256D sub-encodings:

1. **Glyph sequence** — sha256 hash projection of ordered glyphs → 256D
2. **Category distribution** — normalized counts over 8 categories → 256D
3. **Scaffold + tags** — hash projection of scaffold type + flattened tag values → 256D

### Retrieval Query Interface

```python
from src.voxsigil_library.rag import SymbolicRAGMiddleware, QueryContext

query = QueryContext(
    scaffold_type="flow",
    tags={"domain": ["cognition"], "function": ["evaluate"]},
    intent="evaluate prediction signal",
    entropy_budget=0.7,
    lineage="oracle-chain-v2",
)
results = middleware.retrieve_and_enrich(query, top_k=10)
```

### Pipeline Loop

```
retrieve(query_context)
    → filter_illegal()
    → blt_compress_and_score()
    → enrich_context()
    → inject_into_generator()
    → store_lineage()
```

### Vector Store Strategy

- **Phase 1 (now)**: FAISS local, deterministic, zero-dependency fallback to numpy cosine similarity
- **Phase 2 (multi-agent)**: Qdrant remote, shared symbolic memory, agent-aware retrieval
- Avoid Weaviate — too opinionated for symbolic schema constraints

### Constraints (Hard Rules)

- Do NOT retrieve prose. Only sigil objects.
- Do NOT pass invalid sigils to BLT.
- Preserve all `lifecycle` and `lineage` tags through the pipeline.
- Retrieval is by **structure**, not keywords: scaffold_type + tags + glyph patterns.

### Commands

```bash
# Index sigil library into FAISS
PYTHONPATH=C:\UBLT;C:\UBLT\src python -c "
from src.voxsigil_library.rag import FAISSRetriever, SigilEmbedder
import json, pathlib
sigils = [json.loads(l) for l in pathlib.Path('training/datasets').glob('*.jsonl') for l in open(l)]
retriever = FAISSRetriever()
retriever.index(sigils)
retriever.save('storage/rag_index.faiss')
print(f'Indexed {len(sigils)} sigils')
"

# Test retrieval
PYTHONPATH=C:\UBLT;C:\UBLT\src python -c "
from src.voxsigil_library.rag import SymbolicRAGMiddleware, QueryContext
mw = SymbolicRAGMiddleware.from_index('storage/rag_index.faiss')
results = mw.retrieve_and_enrich(QueryContext(scaffold_type='flow', entropy_budget=0.7))
print(f'Retrieved {len(results)} sigils')
for r in results[:3]: print(r['sigil'], r['scaffold']['scaffold_type'])
"
```
