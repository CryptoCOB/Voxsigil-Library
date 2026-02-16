# Workspace Cleanup & Sync Plan

**Status:** Ready to Execute  
**Date:** Feb 16, 2026  
**Goal:** Clean up local workspace, organize files, and sync with GitHub

---

## Phase 1: Current State Analysis

### What We Have Locally
- **~170 untracked files** (all recent work)
- **45+ directories** (mixed production, experimental, data)
- **Multiple phase documentation files** (Phase 0-6)
- **Experimental code** (various algorithms, testing scripts)
- **Generated content** (sigils, outputs, benchmarks)

### What's in GitHub (origin/main)
- **Core library**: `src/`, `scaffolds/`, `bin/`
- **VME 2.0**: `vme/` with Phase 4-6 components
- **Documentation**: `docs/`, main `README.md`
- **GitHub workflows**: `.github/workflows/`
- **License & governance**: `LICENSE`, `CONTRIBUTING.md`, etc.

---

## Phase 2: Organization Strategy

### Keep (Production-Ready)
These files should be tracked in git and pushed to GitHub:

1. **Core VME Components**
   - `vme/` directory (Phase 4-6 already in repo)
   - Any new optimization improvements
   
2. **New Documentation**
   - `AGENT_INTEGRATION_GUIDE.md` (just created) ✅
   - Updated API documentation
   - Agent examples

3. **Core Agent Integration**
   - Scripts that show how agents can use the system
   - Authentication/connection examples
   - Production benchmarks

### Archive (Experimental/Historical)
These should go into a versioned `archive/` or `legacy/` directory:
- Phase 0-3 documentation (context & history)
- Experimental algorithms (behavioral NAS, energy adaptation, etc.)
- Multiple iterations of the same scripts
- Test outputs and temporary files

### Ignore (Development Only)
These should NOT be tracked in git (add to `.gitignore`):
- `.venv/` - Virtual environment
- `__pycache__/`, `.pytest_cache/` - Python cache
- `*.pyc` - Compiled Python
- Checkpoint/model weights (unless explicitly production)
- Local configuration files with secrets
- Data files (maybe)

---

## Phase 3: File Categorization

### Production Files (Keep in Root/Subdirs)
```
AGENT_INTEGRATION_GUIDE.md           ✅ New - Agent onboarding
docs/                                 ✅ Exists - API docs, etc.
vme/                                  ✅ Exists - VME 2.0 phases
src/                                  ✅ Exists - Core SDK
bin/                                  ✅ Exists - CLI tools
scaffolds/                            ✅ Exists - Sigil templates
README.md                             ✅ Exists - Main entry point
VOXSIGIL_2.0_DEPLOYMENT_COMPLETE.md  ✅ Exists - Deployment record
```

### Archive Directory (Organize Old Work)
```
archive/
├── phase_0_roadmap/               # Original planning
│   ├── PHASE_0_ROADMAP.md
│   └── PHASE_3_5_EVALUATION_PLAN.md
├── phase_1_foundation/            # Foundation work (Phase A)
│   ├── phase_a2-a5_scripts.py
│   └── notes/
├── phase_2_integration/           # Integration work (Phase B)
│   ├── phase_b1-b3_scripts.py
│   └── notes/
├── phase_3_operation/             # Continuous operation (Phase C)
│   ├── phase_c_scripts.py
│   └── notes/
├── phase_4_cognitive/             # Phase 4 experiments (before VME 2.0)
│   ├── phase_4a_validate.py
│   ├── phase_4b1_*.py
│   └── outputs/
├── other_experiments/             # Other algorithms tested
│   ├── behavioral_nas_nextgen.py
│   ├── neural_architecture_search.py
│   ├── quantum_behavioral_nas.py
│   └── [other experimental .py files]
├── generated_outputs/             # Benchmark outputs
│   ├── phase4b_outputs/
│   ├── phase5_outputs/
│   ├── phase6_outputs/
│   └── generated_sigils/
└── ARCHIVE_README.md              # Index of what's here & why
```

### Directories to Clean
```
These will be removed or consolidated:
- MetaConsciousness/              → archive/
- blt_category_flows/             → archive/ 
- blt_modules_reconstructed/      → archive/
- blt_persona_flows/              → archive/
- blt_training_data/              → archive/
- blt_training_generator.py        → archive/
- [and 50+ other experimental dirs/files]
```

---

## Phase 4: Implementation Steps

### Step 1: Create Archive Structure
```bash
mkdir -p archive/{phase_0_roadmap,phase_1_foundation,phase_2_integration,phase_3_operation,phase_4_cognitive,other_experiments,generated_outputs}
```

### Step 2: Move Experimental Files
```bash
# Move phase documentation
mv PHASE_0_ROADMAP.md archive/phase_0_roadmap/
mv PHASE_3_5_*.md archive/phase_0_roadmap/
mv PHASE_4A_COMPLETION.md archive/phase_4_cognitive/
mv PHASE_4B_*.md archive/phase_4_cognitive/
mv PHASE_6_COMPLETION_SUMMARY.md archive/phase_4_cognitive/
mv PHASE_D_COMPLETION_SUMMARY.md archive/phase_4_cognitive/

# Move experimental Python files
mv behavioral_nas_nextgen.py archive/other_experiments/
mv neural_architecture_search.py archive/other_experiments/
mv quantum_behavioral_nas.py archive/other_experiments/
# [... more files ...]

# Move metadata/state files
mv voxsigil_schema_analysis.json archive/generated_outputs/
mv voxsigil_context_summary.json archive/generated_outputs/
mv voxsigil_inventory.json archive/generated_outputs/
```

### Step 3: Create .gitignore
```
# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
.pytest_cache/
.venv/
venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Data & Checkpoints (unless tracked intentionally)
checkpoints/
weights/
*.bin
*.pt

# Local config with secrets
.env
.env.local
*.key
config.local.json

# Logs
*.log

# Temporary files
temp_*
tmp_*
*.tmp
```

### Step 4: Create Archive README
```
archive/README.md - Index explaining:
- Why these files are here
- Phase timeline and what each phase accomplished
- How to navigate the archive
- Which files might be useful for future reference
```

### Step 5: Commit and Push
```bash
git add AGENT_INTEGRATION_GUIDE.md
git add archive/
git add .gitignore
git commit -m "Cleanup: Organize workspace and add comprehensive agent integration guide"
git push origin main
```

---

## Phase 5: Post-Cleanup Structure

### Local Workspace Will Look Like:
```
voxsigil-library/
├── .git/                          # Git repo
├── .github/workflows/             # CI/CD
├── archive/                       # Historical/experimental work ← NEW
│   ├── phase_0_roadmap/
│   ├── phase_1_foundation/
│   ├── phase_2_integration/
│   ├── phase_3_operation/
│   ├── phase_4_cognitive/
│   ├── other_experiments/
│   ├── generated_outputs/
│   └── README.md
├── bin/                           # CLI tools
├── docs/                          # Documentation
├── src/                           # Core SDK
├── vme/                           # VME 2.0 (Phase 4-6)
├── scaffolds/                     # Sigil templates
├── README.md                      # Main entry point
├── AGENT_INTEGRATION_GUIDE.md     # NEW - Agent onboarding ← NEW
├── .gitignore                     # Git ignore rules ← NEW
├── package.json                   # npm metadata
├── setup.py                       # Python metadata
└── [other config files]
```

### Benefits:
- ✅ **Clean production repository** - Easy to understand at a glance
- ✅ **Preserved history** - All experimental work is still available in `archive/`
- ✅ **Agent-friendly** - Clear documentation on how to use the system
- ✅ **Maintainable** - New contributors understand the structure
- ✅ **Professional** - Looks production-ready for investors/partners

---

## Phase 6: Sync Verification

After cleanup, verify everything is synced:

```bash
# Check status
git status

# Verify everything is committed
git log --oneline -5

# Push to GitHub
git push origin main

# Verify on GitHub
# https://github.com/CryptoCOB/Voxsigil-Library
```

---

## Timeline & Effort

| Phase | Task | Effort | Time |
|-------|------|--------|------|
| 1 | Analyze & plan | 5 min | ✅ Done |
| 2 | Finalize organization | 5 min | In progress |
| 3 | Create .gitignore | 5 min | Ready |
| 4 | Move files to archive/ | 10 min | Ready |
| 5 | Create Archive README | 10 min | Ready |
| 6 | Commit & push | 5 min | Ready |
| - | **Total** | **40 min** | Ready |

---

## Decision Points

### Q1: Should we delete old phase files or archive them?
**A:** Archive. They document the development journey and might be useful for:
- Understanding why certain design decisions were made
- Recovering techniques if needed
- Showing the evolution of the system
- Reference for future work

### Q2: What about model checkpoints and generated sigils?
**A:** 
- If under 10MB and useful: keep in repo
- If over 10MB: store in `archive/generated_outputs/` and consider external storage
- Document where large files are stored

### Q3: Should we include the .venv in git?
**A:** No. Add to `.gitignore` and document installation in `docs/INSTALLATION.md`

---

## Next Actions

**Ready to execute when you say "GO":**
1. Execute Phase 4 (move files to archive/)
2. Create `.gitignore`
3. Create `archive/README.md`
4. Commit everything
5. Push to GitHub

**Then:**
1. Verify on GitHub
2. Create quick summary of new organization
3. Update `docs/README.md` to link to `AGENT_INTEGRATION_GUIDE.md`

---

**Questions?** Let me know before we execute!
