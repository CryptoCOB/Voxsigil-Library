# UBLT Repository - Critical Files Inventory

## Summary
- **Total Source .py Files:** 70
- **Total Bytecode .pyc Files:** 126
- **Status:** We have source for ~37% of files, need to recover 63% from bytecode

## SOURCE FILES WE HAVE (70 files)

### Core BLT Files (Recovered)
- `temp_recovered_blt.py` - Core BLT implementation ✓
- `temp_recovered_blt_interface.py` - BLT Communication Interface ✓
- `temp_recovered_init.py` - Init module ✓
- `temp_recovered_blockchain_utils.py` - Blockchain utilities ✓
- `temp_recovered_mesh.py` - Mesh network ✓
- `temp_recovered_ledger_bridge.py` - Ledger bridge ✓

### Other Source Files
- Root level Python scripts (15+ precompute and test scripts)
- API module: `api/dual_chain_api_server.py`
- Blockchain: `blockchain/dual_chain_blockchain.py`
- Core: `core/echo_state_network.py`, `core/qcnn_advanced_processor.py`, `core/task_fabric.py`
- Nebula modules: Utils, training adapters, and various initialization files
- External: Full Higgs-Audio implementation (50+ files)

## CRITICAL MISSING SOURCE FILES (Bytecode Only) - PRIORITY: HIGH

### 🔴 BLT Core Modules (`c:\UBLT\modules\__pycache__\`)
These are ESSENTIAL for the Byte Latency Transformer:
- `blt.cpython-313.pyc` → **blt.py** (missing)
- `blt_encoder.cpython-313.pyc` → **blt_encoder.py** (missing)
- `blt_encoder_interface.cpython-313.pyc` → **blt_encoder_interface.py** (missing)
- `bidirectional_blt_system.cpython-313.pyc` → **bidirectional_blt_system.py** (missing)
- `hybrid_blt.cpython-313.pyc` → **hybrid_blt.py** (missing)
- `mesh.cpython-313.pyc` → **mesh.py** (missing)
- `eon.cpython-313.pyc` → **eon.py** (missing - "Enhanced Ontological Network"?)
- `recap.cpython-313.pyc` → **recap.py** (missing)
- `art.cpython-313.pyc` → **art.py** (missing)

### 🔴 Consciousness/Core BLT Modules (`c:\UBLT\modules\blt_modules\__pycache__\`)
Critical for consciousness management in BLT:
- `consciousness_manager.cpython-313.pyc` → **consciousness_manager.py** (missing)
- `consciousness_scaffold.cpython-313.pyc` → **consciousness_scaffold.py** (missing)
- `core_processor.cpython-313.pyc` → **core_processor.py** (missing)
- `memory_reflector.cpython-313.pyc` → **memory_reflector.py** (missing)
- `mesh_coordinator.cpython-313.pyc` → **mesh_coordinator.py** (missing)
- `semantic_engine.cpython-313.pyc` → **semantic_engine.py** (missing)

### 🟠 Orchestration/Task Management (`c:\UBLT\orchestration\__pycache__\`)
- `task_reward_tracker.cpython-313.pyc`
- `task_fabric_engine.cpython-313.pyc`
- `ghost_detection_protocol.cpython-313.pyc`
- `ghost_detection_models.cpython-313.pyc`

### 🟠 Input Modules (`c:\UBLT\input_modules\__pycache__\`)
All input modalities are bytecode-only:
- `AuditoryInputModule.cpython-313.pyc`
- `ImageInputModuleV1.cpython-313.pyc`
- `TabularInputModuleV1.cpython-313.pyc`
- `TextInputModuleV1.cpython-313.pyc`
- `VideoInputModule.cpython-313.pyc`
- `ProprioceptiveInputModule.cpython-313.pyc`
- `OlfactoryInputModule.cpython-313.pyc`
- `TactileInputModule.cpython-313.pyc`

### 🟠 Research/NAS (`c:\UBLT\research\__pycache__\`)
- `nas_search_space.cpython-313.pyc`
- `nas_orchestrator.cpython-313.pyc`
- `nas_memory.cpython-313.pyc`
- `nas_layers_specialized.cpython-313.pyc`
- `nas_evolution_loop.cpython-313.pyc`
- `nas_controller.cpython-313.pyc`

### 🟠 Storage & Connectors
- `storage/PineconeManager.cpython-313.pyc`
- `integration/service_registry.cpython-313.pyc`
- `integration/orchestrator.cpython-313.pyc`

### 🟠 Nebula Core Components
- `nebula/core/meta_cognitive.cpython-313.pyc`
- `nebula/core/memory_router.cpython-313.pyc`
- `nebula/core/inference_only_core.cpython-313.pyc`
- `nebula/core/quantum/qcnn_core.cpython-313.pyc`
- `nebula/core/quantum/quantum_gates.cpython-313.pyc`
- `nebula/core/quantum/encoding.cpython-313.pyc`
- `nebula/blockchain/orionbelt_client.cpython-313.pyc`
- `nebula/crypto/sigilwallet.cpython-313.pyc`
- `nebula/rag/rag_engine.cpython-313.pyc`
- `nebula/rag/local_vector_store.cpython-313.pyc`
- And more...

## RECOVERY STRATEGY

### Step 1: Install Decompiler
```bash
pip install uncompyle6
# OR
pip install pycdc
```

### Step 2: Recover Critical BLT Module Source Files
Priority order:
1. `modules/__pycache__/*.pyc` - Core BLT system
2. `modules/blt_modules/__pycache__/*.pyc` - Consciousness managers
3. `orchestration/__pycache__/*.pyc` - Task orchestration
4. `input_modules/__pycache__/*.pyc` - Input handling
5. All other modules

### Step 3: Verify Recovered Files
- Check for decompilation quality
- Test imports and basic functionality
- Compare with recovered_*.py patterns

### Step 4: Cleanup Bytecode
Once verified, can optionally consolidate or remove __pycache__ directories

## File Status Legend
- ✓ Source file exists
- ❌ Only bytecode exists (need decompilation)
- 🔴 CRITICAL - Core BLT functionality
- 🟠 IMPORTANT - Key system components
- 🟡 Useful - Supporting modules
