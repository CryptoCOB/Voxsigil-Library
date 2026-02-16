# 🌌 Nebula Dual-Chain Blockchain Implementation
# Enhanced OrionBelt with MainChain/TestChain support

import threading
import logging
import json
import time
import os
from typing import List, Dict, Any
from config.chain_config import ChainConfig, ChainConfigManager, ChainType, get_current_chain

class DualChainBlockchain:
    """Enhanced Blockchain with dual-chain architecture support"""
    
    def __init__(self, config: ChainConfig = None):
        """Initialize blockchain with chain-specific configuration"""
        self.config = config or get_current_chain()
        self._lock = threading.RLock()
        
        # Chain identification
        self.chain_id = self.config.chain_id
        self.network_name = self.config.network_name
        self.chain_type = self.config.chain_type
        
        # Core blockchain data
        self.chain: List = []  # Will contain Block objects
        self.pending_transactions: List = []
        
        # Chain-specific parameters
        self.difficulty = self.config.difficulty
        self.mining_reward = self.config.base_reward
        self.max_reorg_depth = self.config.max_reorg_depth
        self.block_time_target = self.config.block_time_target
        
        # Validators and consensus
        self.validators = self.config.validators
        self.min_validators = self.config.min_validators
        self.consensus_threshold = self.config.consensus_threshold
        self.consensus = None  # Initialized lazily
        
        # Task Fabric integration (V1 features)
        self.compute_tasks: List = []
        self.task_claims: Dict[str, List] = {}
        self.task_results: Dict[str, List] = {}
        
        # Task Fabric Engine (initialized after blockchain setup)
        self.task_fabric = None
        
        # dVPN integration
        self.dvpn_nodes: Dict[str, Dict] = {}
        self.bandwidth_receipts: List = []
        
        # dVPN Overlay Network (enhanced)
        self.dvpn_overlay = None
        
        # Persistence
        self.persistence_path = self.config.blockchain_file
        
        # Initialize genesis block
        with self._lock:
            self._initialize_chain()
            
        # Initialize Task Fabric after blockchain is ready
        self._initialize_task_fabric()
        
        # Initialize PoUW consensus engine
        self._initialize_pouw_engine()
        self._initialize_dvpn_overlay()
        
        # Initialize dVPN overlay network
        self._initialize_dvpn_overlay()
            
        logging.info(f"🔗 Initialized {self.network_name} ({self.chain_id})")
        logging.info(f"   Chain Type: {self.chain_type.value.upper()}")
        logging.info(f"   Difficulty: {self.difficulty}")
        logging.info(f"   Block Time: {self.block_time_target}s")
        logging.info(f"   Mining Reward: {self.mining_reward}")
    
    def _initialize_task_fabric(self):
        """Initialize Task Fabric engine for this blockchain"""
        try:
            # Import here to avoid circular imports
            from task_fabric import TaskFabricEngine
            self.task_fabric = TaskFabricEngine(self)
            logging.info(f"✅ Task Fabric initialized for {self.network_name}")
        except Exception as e:
            logging.error(f"Failed to initialize Task Fabric: {e}")
            self.task_fabric = None
    
    def _initialize_pouw_engine(self):
        """Initialize Proof-of-Useful-Work consensus engine"""
        try:
            from proof_of_useful_work import integrate_pouw_with_blockchain
            self.pouw_engine = integrate_pouw_with_blockchain(self, self.task_fabric)
            logging.info(f"🔥 PoUW consensus initialized for {self.network_name}")
        except Exception as e:
            logging.error(f"Failed to initialize PoUW engine: {e}")
            self.pouw_engine = None
    
    def _initialize_dvpn_overlay(self):
        """Initialize dVPN overlay network with advanced features"""
        try:
            from production_dvpn_overlay import create_dvpn_overlay
            self.dvpn_overlay = create_dvpn_overlay(self)
            logging.info(f"🌐 Production dVPN overlay network initialized for {self.network_name}")
        except Exception as e:
            logging.error(f"Failed to initialize dVPN overlay: {e}")
            self.dvpn_overlay = None
    
    def _initialize_chain(self):
        """Initialize blockchain with genesis block"""
        try:
            # Try to load existing chain
            self._load_from_disk()
            if len(self.chain) == 0:
                raise FileNotFoundError("No existing chain")
            logging.info(f"📁 Loaded existing {self.network_name} with {len(self.chain)} blocks")
        except (FileNotFoundError, json.JSONDecodeError):
            # Create genesis block for new chain
            genesis_block = self._create_genesis_block()
            self.chain.append(genesis_block)
            self._save_to_disk()
            logging.info(f"🆕 Created genesis block for {self.network_name}")
    
    def _create_genesis_block(self):
        """Create deterministic genesis block specific to chain type"""
        # Import Block here to avoid circular imports
        from OrionBelt import Block
        
        genesis_tx = {
            "genesis": True,
            "chain_id": self.chain_id,
            "network": self.network_name,
            "chain_type": self.chain_type.value,
            "timestamp": 0.0,
            "initial_supply": 1_000_000_000,  # 1B tokens
            "validators": self.validators,
            "note": f"Genesis block for {self.network_name}"
        }
        
        return Block(
            index=0,
            previous_hash="0" * 64,  # Clean 64-char hash
            timestamp=0.0,
            transactions=[genesis_tx],
            validator="System",
            nonce=0
        )
    
    def _load_from_disk(self):
        """Load blockchain from persistent storage"""
        with open(self.persistence_path, 'r') as f:
            chain_data = json.load(f)
            
        # Validate chain belongs to correct network, with legacy ID auto-migration support
        on_disk_chain_id = chain_data.get('chain_id')
        if on_disk_chain_id != self.chain_id:
            legacy_ids = {
                ChainType.MAIN: {"nebula-main-1", "nebula-main"},
                ChainType.TEST: {"nebula-test-1", "nebula-test"},
            }
            allow_migration = os.getenv("NEBULA_ALLOW_CHAIN_ID_MIGRATION", "1").lower() in {"1", "true", "yes"}
            if allow_migration and on_disk_chain_id in legacy_ids.get(self.chain_type, set()):
                logging.warning(
                    "Legacy chain ID detected (%s) for %s; migrating to %s",
                    on_disk_chain_id,
                    self.chain_type.value,
                    self.chain_id,
                )
                # Load existing data, then persist with the new chain_id via _save_to_disk
                self.chain = chain_data.get('chain', [])
                self.compute_tasks = chain_data.get('compute_tasks', [])
                self.task_claims = chain_data.get('task_claims', {})
                self.task_results = chain_data.get('task_results', {})
                # Best-effort: capture optional persisted dvpn fields
                self.dvpn_nodes = chain_data.get('dvpn_nodes', {})
                self.bandwidth_receipts = chain_data.get('bandwidth_receipts', [])
                # Rewrite file with the expected chain_id and naming
                self._save_to_disk()
                return
            else:
                raise ValueError(
                    f"Chain ID mismatch: expected {self.chain_id}, got {on_disk_chain_id}"
                )
        
        # Load blocks (would need proper Block deserialization)
        self.chain = chain_data.get('chain', [])
        self.compute_tasks = chain_data.get('compute_tasks', [])
        self.task_claims = chain_data.get('task_claims', {})
        self.task_results = chain_data.get('task_results', {})
    
    def _save_to_disk(self):
        """Save blockchain to persistent storage"""
        chain_data = {
            'chain_id': self.chain_id,
            'network_name': self.network_name,
            'chain_type': self.chain_type.value,
            'chain': self.chain,
            'compute_tasks': self.compute_tasks,
            'task_claims': self.task_claims,
            'task_results': self.task_results,
            'dvpn_nodes': self.dvpn_nodes,
            'bandwidth_receipts': self.bandwidth_receipts[-1000:]  # Keep last 1000 receipts
        }
        
        with open(self.persistence_path, 'w') as f:
            json.dump(chain_data, f, indent=2, default=str)
    
    def get_chain_info(self) -> Dict[str, Any]:
        """Get information about current chain"""
        return {
            'chain_id': self.chain_id,
            'network_name': self.network_name,
            'chain_type': self.chain_type.value,
            'block_height': len(self.chain),
            'difficulty': self.difficulty,
            'mining_reward': self.mining_reward,
            'block_time_target': self.block_time_target,
            'validators': len(self.validators),
            'pending_transactions': len(self.pending_transactions),
            'compute_tasks': len(self.compute_tasks),
            'dvpn_nodes': len(self.dvpn_nodes)
        }
    
    def is_main_chain(self) -> bool:
        """Check if this is the MainChain"""
        return self.chain_type == ChainType.MAIN
    
    def is_test_chain(self) -> bool:
        """Check if this is the TestChain"""
        return self.chain_type == ChainType.TEST
    
    def is_chain_valid(self) -> bool:
        """Validate blockchain integrity"""
        with self._lock:
            if len(self.chain) == 0:
                return True  # Empty chain is valid
            
            # Check chain continuity
            for i in range(1, len(self.chain)):
                current_block = self.chain[i]
                previous_block = self.chain[i-1]
                
                # Check if previous hash matches
                current_prev_hash = getattr(current_block, 'previous_hash', '')
                previous_hash = getattr(previous_block, 'hash', '')
                
                if str(current_prev_hash) != str(previous_hash):
                    return False
                
                # Check index continuity
                current_index = getattr(current_block, 'index', i)
                previous_index = getattr(previous_block, 'index', i-1)
                
                if current_index != previous_index + 1:
                    return False
            
            return True
    
    def add_transaction(self, transaction):
        """Add transaction to pending pool with chain validation"""
        with self._lock:
            # Add chain ID to transaction for validation
            if isinstance(transaction, dict):
                transaction['chain_id'] = self.chain_id
            self.pending_transactions.append(transaction)
    
    def validate_cross_chain_tx(self, transaction: Dict) -> bool:
        """Validate transaction belongs to correct chain"""
        tx_chain_id = transaction.get('chain_id')
        if tx_chain_id and tx_chain_id != self.chain_id:
            logging.warning(f"Cross-chain transaction rejected: {tx_chain_id} != {self.chain_id}")
            return False
        return True
    
    # Task Fabric methods (V1 features)
    def add_compute_task(self, task: Dict) -> bool:
        """Add compute task to blockchain with Task Fabric integration"""
        with self._lock:
            task['chain_id'] = self.chain_id
            task['created_at'] = task.get('created_at', time.time())
            
            # Add to blockchain storage
            self.compute_tasks.append(task)
            
            # If Task Fabric is available, register with it
            if self.task_fabric:
                try:
                    self.task_fabric.create_task_from_dict(task)
                    logging.info(f"Task {task.get('task_id')} registered with Task Fabric")
                except Exception as e:
                    logging.error(f"Failed to register task with Task Fabric: {e}")
            
            self._save_to_disk()
            return True
    
    def claim_task(self, task_id: str, miner: str) -> bool:
        """Claim task for processing with enhanced validation"""
        with self._lock:
            if task_id not in self.task_claims:
                self.task_claims[task_id] = []
            
            claim = {
                'task_id': task_id,
                'miner': miner,
                'timestamp': time.time(),
                'chain_id': self.chain_id,
                'status': 'active'
            }
            self.task_claims[task_id].append(claim)
            
            # Integrate with Task Fabric if available
            if self.task_fabric:
                try:
                    success = self.task_fabric.claim_task(task_id, miner)
                    if not success:
                        logging.warning(f"Task Fabric rejected claim for {task_id} by {miner}")
                        return False
                except Exception as e:
                    logging.error(f"Task Fabric claim error: {e}")
            
            return True
    
    def submit_task_result(self, result: Dict) -> bool:
        """Submit task computation result with verification"""
        with self._lock:
            task_id = result.get('task_id')
            if task_id not in self.task_results:
                self.task_results[task_id] = []
            
            result['chain_id'] = self.chain_id
            result['submitted_at'] = result.get('submitted_at', time.time())
            self.task_results[task_id].append(result)
            
            # Integrate with Task Fabric for verification
            if self.task_fabric:
                try:
                    # Submit to Task Fabric for verification
                    # Extract required parameters for TaskFabricEngine
                    task_id = result.get('task_id', 'unknown_task')
                    worker_id = result.get('miner', result.get('worker_id', 'unknown_worker'))
                    result_data = result.get('result', result)
                    
                    # Call TaskFabricEngine with proper signature (non-async)
                    success = self.task_fabric.submit_result(task_id, worker_id, result_data)
                    logging.info(f"TaskFabric result submission: {success}")
                except Exception as e:
                    logging.error(f"Task Fabric result submission error: {e}")
            
            return True
    
    def get_task_fabric(self):
        """Get Task Fabric engine instance"""
        return self.task_fabric
    
    def get_pending_tasks_enhanced(self, miner_capabilities: Dict[str, Any] = None) -> List[Dict]:
        """Get pending tasks with Task Fabric intelligence"""
        if self.task_fabric:
            try:
                tasks = self.task_fabric.get_pending_tasks(miner_capabilities)
                return [task.to_dict() for task in tasks]
            except Exception as e:
                logging.error(f"Task Fabric pending tasks error: {e}")
                
        # Fallback to basic pending tasks
        return [task for task in self.compute_tasks 
                if task.get('status', 'pending') == 'pending']
    
    def get_task_status_enhanced(self, task_id: str) -> Dict[str, Any]:
        """Get comprehensive task status via Task Fabric"""
        if self.task_fabric:
            try:
                return self.task_fabric.get_task_status(task_id)
            except Exception as e:
                logging.error(f"Task Fabric status error: {e}")
        
        # Fallback to basic status
        task = next((t for t in self.compute_tasks if t.get('task_id') == task_id), None)
        if not task:
            return {"error": "Task not found"}
            
        claims = self.task_claims.get(task_id, [])
        results = self.task_results.get(task_id, [])
        
        return {
            "task": task,
            "claims": claims,
            "results": results,
            "status": task.get('status', 'pending')
        }
    
    def mine_pouw_block(self, transactions: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Mine a new block using Proof-of-Useful-Work"""
        if not self.pouw_engine:
            logging.warning("PoUW engine not available, falling back to standard mining")
            return self._mine_standard_block(transactions)
        
        with self._lock:
            previous_block = self.chain[-1] if self.chain else self._get_genesis_block()
            height = len(self.chain)
            
            # Get previous hash depending on block type
            if hasattr(previous_block, 'hash'):
                previous_hash = previous_block.hash
            elif isinstance(previous_block, dict):
                previous_hash = previous_block['hash']
            else:
                previous_hash = "0" * 64  # Fallback genesis hash
            
            # Create PoUW block with useful work proofs
            pouw_block = self.pouw_engine.create_pouw_block(
                previous_hash=previous_hash,
                height=height,
                transactions=transactions or []
            )
            
            # Validate the PoUW block
            if not self.pouw_engine.validate_pouw_block(pouw_block):
                logging.error("Failed to validate PoUW block")
                return {"success": False, "error": "Block validation failed"}
            
            # Convert to standard block format and add hash/nonce
            block_dict = pouw_block.to_dict()
            block_dict['hash'] = self._calculate_block_hash(block_dict)
            
            # Add to chain
            self.chain.append(block_dict)
            self._save_to_disk()
            
            logging.info(f"⛏️ Mined PoUW block {height} with {len(pouw_block.useful_work_proofs)} work proofs, total value: {pouw_block.total_work_value:.2f}")
            
            return {
                "success": True,
                "block": block_dict,
                "work_proofs": len(pouw_block.useful_work_proofs),
                "work_value": pouw_block.total_work_value,
                "miner_rewards": pouw_block.miner_rewards
            }
    
    def _calculate_block_hash(self, block_dict: Dict[str, Any]) -> str:
        """Calculate hash for a block dictionary"""
        import hashlib
        import json
        
        # Create a copy without the hash field for calculation
        hash_data = {k: v for k, v in block_dict.items() if k != 'hash'}
        
        # Convert to deterministic JSON
        block_string = json.dumps(hash_data, sort_keys=True, separators=(',', ':'))
        
        # Calculate SHA-256 hash
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def _mine_standard_block(self, transactions: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Fallback standard mining without PoUW"""
        with self._lock:
            previous_block = self.chain[-1] if self.chain else self._get_genesis_block()
            height = len(self.chain)
            
            block = {
                'height': height,
                'previous_hash': previous_block['hash'],
                'timestamp': time.time(),
                'transactions': transactions or [],
                'useful_work_proofs': [],
                'total_work_value': 0.0,
                'work_quality_score': 0.0,
                'miner_rewards': {},
                'nonce': str(time.time()),
                'difficulty': self.config.difficulty
            }
            
            block['hash'] = self._calculate_block_hash(block)
            self.chain.append(block)
            self._save_to_disk()
            
            return {"success": True, "block": block}
    
    def get_pouw_stats(self) -> Dict[str, Any]:
        """Get PoUW consensus statistics"""
        if not self.pouw_engine:
            return {"error": "PoUW engine not available"}
        
        pouw_stats = self.pouw_engine.get_stats()
        
        # Add blockchain-specific stats
        pouw_blocks = []
        for block in self.chain:
            # Handle both Block objects and dictionaries
            block_dict = block.to_dict() if hasattr(block, 'to_dict') else block
            if 'useful_work_proofs' in block_dict and len(block_dict['useful_work_proofs']) > 0:
                pouw_blocks.append(block_dict)
        
        total_work_value = sum(block.get('total_work_value', 0) for block in pouw_blocks)
        total_work_proofs = sum(len(block.get('useful_work_proofs', [])) for block in pouw_blocks)
        
        pouw_stats.update({
            "pouw_blocks_count": len(pouw_blocks),
            "total_blocks": len(self.chain),
            "pouw_adoption_rate": len(pouw_blocks) / max(len(self.chain), 1),
            "total_work_value_processed": total_work_value,
            "total_work_proofs_processed": total_work_proofs
        })
        
        return pouw_stats
    
    # dVPN methods
    def register_dvpn_node(self, node_info: Dict) -> bool:
        """Register dVPN node"""
        with self._lock:
            node_id = node_info.get('node_id')
            node_info['chain_id'] = self.chain_id
            self.dvpn_nodes[node_id] = node_info
            return True
    
    def submit_bandwidth_receipt(self, receipt: Dict) -> bool:
        """Submit bandwidth proof receipt"""
        with self._lock:
            receipt['chain_id'] = self.chain_id
            self.bandwidth_receipts.append(receipt)
            return True

class ChainManager:
    """Manages multiple blockchain instances"""
    
    def __init__(self):
        self._chains: Dict[ChainType, DualChainBlockchain] = {}
        self._active_chain: ChainType = ChainType.TEST  # Default to test
    
    def get_chain(self, chain_type: ChainType = None) -> DualChainBlockchain:
        """Get blockchain instance for specific chain type"""
        if chain_type is None:
            chain_type = self._active_chain
            
        if chain_type not in self._chains:
            config = ChainConfigManager.get_config(chain_type, from_env=False)
            self._chains[chain_type] = DualChainBlockchain(config)
            
        return self._chains[chain_type]
    
    def get_main_chain(self) -> DualChainBlockchain:
        """Get MainChain instance"""
        return self.get_chain(ChainType.MAIN)
    
    def get_test_chain(self) -> DualChainBlockchain:
        """Get TestChain instance"""
        return self.get_chain(ChainType.TEST)
    
    def set_active_chain(self, chain_type: ChainType):
        """Set active chain for operations"""
        self._active_chain = chain_type
        ChainConfigManager.set_chain_environment(chain_type)
        logging.info(f"🔗 Switched to {chain_type.value.upper()} chain")
    
    def get_active_chain(self) -> DualChainBlockchain:
        """Get currently active chain"""
        return self.get_chain(self._active_chain)
    
    def get_chain_status(self) -> Dict[str, Any]:
        """Get status of all chains"""
        status = {}
        for chain_type in [ChainType.MAIN, ChainType.TEST]:
            try:
                chain = self.get_chain(chain_type)
                status[chain_type.value] = chain.get_chain_info()
            except Exception as e:
                status[chain_type.value] = {"error": str(e)}
        
        status['active_chain'] = self._active_chain.value
        return status

# Global chain manager instance
chain_manager = ChainManager()

# Convenience functions
def get_main_chain() -> DualChainBlockchain:
    """Get MainChain instance"""
    return chain_manager.get_main_chain()

def get_test_chain() -> DualChainBlockchain:
    """Get TestChain instance"""  
    return chain_manager.get_test_chain()

def get_active_chain() -> DualChainBlockchain:
    """Get currently active chain"""
    return chain_manager.get_active_chain()

def switch_to_main():
    """Switch to MainChain"""
    chain_manager.set_active_chain(ChainType.MAIN)

def switch_to_test():
    """Switch to TestChain"""
    chain_manager.set_active_chain(ChainType.TEST)

if __name__ == "__main__":
    import time
    
    # Test dual-chain setup
    print("🧪 Testing Dual-Chain Architecture")
    
    # Test TestChain
    switch_to_test()
    test_chain = get_test_chain()
    print(f"✅ TestChain: {test_chain.network_name}")
    print(f"   Chain ID: {test_chain.chain_id}")
    print(f"   Difficulty: {test_chain.difficulty}")
    print(f"   Block Time: {test_chain.block_time_target}s")
    
    # Test MainChain  
    switch_to_main()
    main_chain = get_main_chain()
    print(f"✅ MainChain: {main_chain.network_name}")
    print(f"   Chain ID: {main_chain.chain_id}")
    print(f"   Difficulty: {main_chain.difficulty}")
    print(f"   Block Time: {main_chain.block_time_target}s")
    
    # Show status
    print("\n📊 Chain Status:")
    status = chain_manager.get_chain_status()
    for chain_name, info in status.items():
        if chain_name != 'active_chain':
            print(f"   {chain_name.upper()}: {info.get('block_height', 0)} blocks")
    
    print(f"   Active: {status['active_chain'].upper()}")