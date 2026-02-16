# VoxSigil Benchmark Quick Start

**Goal**: Run BEIR benchmark to compare VoxSigil vs MiniLM and get public validation

---

## 📦 Installation

```bash
# Install BEIR
pip install beir sentence-transformers

# Verify installation
python -c "from beir import util; print('BEIR installed successfully')"
```

---

## 🚀 Quick Run (5 minutes)

Run fast evaluation on 5 small datasets:

```bash
cd c:\UBLT
python benchmarks/beir_voxsigil_comparison.py --tasks fast
```

**Expected output**:
- `benchmarks/results/beir_comparison_results.json`
- `benchmarks/results/beir_comparison_report.md`

**Tasks included** (fast mode):
- NFCorpus (Medical) - 3,633 docs
- SciFact (Scientific) - 5,183 docs  
- ArguAna (Arguments) - 8,674 docs
- SCIDOCS (Citations) - 25,657 docs
- FiQA (Financial) - 57,638 docs

**Runtime**: ~5-10 minutes total

---

## 📊 Full Benchmark (2-3 hours)

Run complete evaluation on all 15 BEIR tasks:

```bash
python benchmarks/beir_voxsigil_comparison.py --tasks all
```

**Tasks included** (all mode):
- All 5 fast tasks above
- Plus 10 larger datasets (NQ, HotpotQA, MS MARCO, etc.)

**Runtime**: 2-3 hours depending on hardware

---

## 🎯 Expected Results

### Target Metrics

```
Model              Params   Dim   NDCG@10   MAP@10   Recall@10
MiniLM-L6-v2       22M      384D  0.380     0.320    0.520     (baseline)
VoxSigil-Student   3.7M     128D  0.340+    0.280+   0.470+    (target: 90% retention)
```

### Success Criteria

- ✅ NDCG@10: >0.34 (90% of baseline 0.38)
- ✅ MAP@10: >0.28 (88% of baseline 0.32)
- ✅ Recall@10: >0.47 (90% of baseline 0.52)
- ✅ Encoding speed: 2-3x faster than MiniLM
- ✅ Model size: 6x smaller (3.7M vs 22M params)

---

## 🔧 Troubleshooting

### Issue: "BEIR not installed"
```bash
pip install beir sentence-transformers
```

### Issue: "VoxSigil student embedder not found"
First train the student embedder:
```bash
python phase_4b1_student_embedder.py
```
This creates: `phase4b_outputs/student_embedder_128d.pkl`

### Issue: "Dataset download failed"
BEIR will auto-download datasets from:
```
https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/
```
If download fails, check internet connection or download manually.

### Issue: Out of memory
Reduce batch size in the code:
- MiniLM: `batch_size=16` → `batch_size=8`
- VoxSigil: `batch_size=8` → `batch_size=4`

---

## 📈 Interpreting Results

### Good Result Example
```json
{
  "baseline": {
    "avg_ndcg_at_10": 0.3800,
    "avg_map_at_10": 0.3200
  },
  "voxsigil": {
    "avg_ndcg_at_10": 0.3450,  ← 91% retention (GOOD!)
    "avg_map_at_10": 0.2920    ← 91% retention (GOOD!)
  }
}
```

### Marginal Result Example
```json
{
  "voxsigil": {
    "avg_ndcg_at_10": 0.3100,  ← 82% retention (MARGINAL)
    "avg_map_at_10": 0.2560    ← 80% retention (MARGINAL)
  }
}
```
**Action**: Increase student model to 256D, accept 3x compression instead of 6x

### Poor Result Example
```json
{
  "voxsigil": {
    "avg_ndcg_at_10": 0.2500,  ← 66% retention (POOR)
    "avg_map_at_10": 0.2000    ← 62% retention (POOR)
  }
}
```
**Action**: Retrain with better distillation loss or more training data

---

## 📝 Next Steps After Benchmarking

### 1. If Results Are Good (>85% retention)
- [ ] Publish results to BEIR leaderboard
- [ ] Write blog post: "VoxSigil: 6x Smaller Embedder with 90% Accuracy"
- [ ] Submit to relevant conferences (ACL, EMNLP, NeurIPS)
- [ ] Open-source the model on HuggingFace
- [ ] Create pip package: `pip install voxsigil`

### 2. If Results Are Marginal (80-85% retention)
- [ ] Increase model to 256D (accept 3x vs 6x compression)
- [ ] Try different distillation temperatures
- [ ] Add more training data
- [ ] Experiment with different student architectures
- [ ] Re-run benchmarks and compare

### 3. If Results Are Poor (<80% retention)
- [ ] Debug the embedding quality
- [ ] Check if teacher embeddings are correct
- [ ] Validate training loss convergence
- [ ] Consider hybrid approach (adaptive dimension)

---

## 🎓 Understanding BEIR Metrics

### NDCG@10 (Normalized Discounted Cumulative Gain)
- **Range**: 0.0 to 1.0
- **Meaning**: Quality of top 10 rankings (position matters)
- **Good score**: >0.35
- **Excellent score**: >0.45

### MAP@10 (Mean Average Precision)
- **Range**: 0.0 to 1.0
- **Meaning**: Precision across all relevant results in top 10
- **Good score**: >0.30
- **Excellent score**: >0.40

### Recall@10
- **Range**: 0.0 to 1.0
- **Meaning**: % of relevant docs found in top 10
- **Good score**: >0.45
- **Excellent score**: >0.60

---

## 📞 Support

If benchmarks fail or results are unexpected:
1. Check logs in console output
2. Verify student embedder exists: `phase4b_outputs/student_embedder_128d.pkl`
3. Test on single task first: `--tasks nfcorpus`
4. Share results for analysis

---

## 🔗 References

- BEIR Paper: https://arxiv.org/abs/2104.08663
- BEIR GitHub: https://github.com/beir-cellar/beir
- BEIR Leaderboard: https://eval.ai/web/challenges/challenge-page/1897/leaderboard

---

**Ready to run?** Execute:
```bash
python benchmarks/beir_voxsigil_comparison.py --tasks fast
```
