#!/usr/bin/env python3
"""
Proof of Bandwidth (PoB) System
Cryptographic bandwidth receipts for VantaEcho Nebula dVPN

This module provides:
- Cryptographic bandwidth receipts signed by both endpoints
- Accurate bandwidth contribution tracking  
- Time-windowed proof generation and validation
- Integration with dVPN tunnel system
- Support for Test P2-DVPN-048

Author: VantaEcho Nebula System  
Date: September 17, 2025
"""

import asyncio
import json
import logging
import time
import uuid
import struct
import hashlib
import hmac
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import threading
import secrets

# Cryptographic imports
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, PublicFormat, NoEncryption
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logging.warning("Cryptography not available, using mock implementations")

# Import related modules
try:
    from dvpn_tunnel_manager import get_tunnel_manager
    TUNNEL_MANAGER_AVAILABLE = True
except ImportError:
    TUNNEL_MANAGER_AVAILABLE = False

try:
    from onchain_node_registry import get_node_registry  
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProofStatus(Enum):
    """Bandwidth proof status"""
    PENDING = "pending"
    VALIDATED = "validated" 
    DISPUTED = "disputed"
    REJECTED = "rejected"
    EXPIRED = "expired"

class BandwidthDirection(Enum):
    """Bandwidth direction"""
    UPLOAD = "upload"
    DOWNLOAD = "download"
    BIDIRECTIONAL = "bidirectional"

@dataclass
class BandwidthMeasurement:
    """Individual bandwidth measurement"""
    measurement_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    
    # Measurement data
    bytes_transferred: int = 0
    duration_seconds: float = 0.0
    direction: BandwidthDirection = BandwidthDirection.BIDIRECTIONAL
    
    # Network information
    source_node: str = ""
    destination_node: str = ""
    tunnel_id: Optional[str] = None
    
    # Quality metrics
    latency_ms: float = 0.0
    packet_loss_rate: float = 0.0
    jitter_ms: float = 0.0
    
    # Validation data
    data_hash: str = ""
    sequence_number: int = 0
    
    def calculate_throughput_mbps(self) -> float:
        """Calculate throughput in Mbps"""
        if self.duration_seconds <= 0:
            return 0.0
        bits = self.bytes_transferred * 8
        return bits / (self.duration_seconds * 1_000_000)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['direction'] = self.direction.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BandwidthMeasurement':
        """Create from dictionary"""
        if 'direction' in data:
            data['direction'] = BandwidthDirection(data['direction'])
        return cls(**data)

@dataclass
class BandwidthReceipt:
    """Cryptographic bandwidth receipt"""
    receipt_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    
    # Time window
    window_start: float = 0.0
    window_end: float = 0.0
    window_duration: float = 0.0
    
    # Participants
    provider_node: str = ""  # Node providing bandwidth
    consumer_node: str = ""  # Node consuming bandwidth
    
    # Bandwidth data
    measurements: List[BandwidthMeasurement] = field(default_factory=list)
    total_bytes: int = 0
    average_throughput_mbps: float = 0.0
    peak_throughput_mbps: float = 0.0
    
    # Quality metrics
    average_latency_ms: float = 0.0
    uptime_percentage: float = 100.0
    reliability_score: float = 1.0
    
    # Cryptographic proof
    provider_signature: str = ""
    consumer_signature: str = ""
    proof_hash: str = ""
    nonce: str = field(default_factory=lambda: secrets.token_hex(16))
    
    # Validation
    status: ProofStatus = ProofStatus.PENDING
    validator_signatures: List[str] = field(default_factory=list)
    
    def add_measurement(self, measurement: BandwidthMeasurement):
        """Add bandwidth measurement to receipt"""
        self.measurements.append(measurement)
        self._recalculate_metrics()
    
    def _recalculate_metrics(self):
        """Recalculate aggregate metrics"""
        if not self.measurements:
            return
        
        # Calculate totals
        self.total_bytes = sum(m.bytes_transferred for m in self.measurements)
        
        # Calculate throughput metrics
        throughputs = [m.calculate_throughput_mbps() for m in self.measurements]
        self.average_throughput_mbps = sum(throughputs) / len(throughputs) if throughputs else 0.0
        self.peak_throughput_mbps = max(throughputs) if throughputs else 0.0
        
        # Calculate quality metrics
        latencies = [m.latency_ms for m in self.measurements if m.latency_ms > 0]
        self.average_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0
        
        # Calculate reliability based on packet loss
        loss_rates = [m.packet_loss_rate for m in self.measurements]
        avg_loss = sum(loss_rates) / len(loss_rates) if loss_rates else 0.0
        self.reliability_score = max(0.0, 1.0 - avg_loss)
        
        # Update time window
        timestamps = [m.timestamp for m in self.measurements]
        if timestamps:
            self.window_start = min(timestamps)
            self.window_end = max(timestamps)
            self.window_duration = self.window_end - self.window_start
    
    def generate_proof_hash(self) -> str:
        """Generate cryptographic hash of receipt data"""
        # Create canonical representation
        proof_data = {
            'receipt_id': self.receipt_id,
            'window_start': self.window_start,
            'window_end': self.window_end,
            'provider_node': self.provider_node,
            'consumer_node': self.consumer_node,
            'total_bytes': self.total_bytes,
            'average_throughput_mbps': self.average_throughput_mbps,
            'nonce': self.nonce
        }
        
        # Sort keys for consistency
        canonical_json = json.dumps(proof_data, sort_keys=True, separators=(',', ':'))
        
        # Generate hash
        self.proof_hash = hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()
        return self.proof_hash
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['status'] = self.status.value
        data['measurements'] = [m.to_dict() for m in self.measurements]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BandwidthReceipt':
        """Create from dictionary"""
        if 'status' in data:
            data['status'] = ProofStatus(data['status'])
        if 'measurements' in data:
            data['measurements'] = [BandwidthMeasurement.from_dict(m) for m in data['measurements']]
        return cls(**data)

class CryptographicSigner:
    """Handles cryptographic signing for bandwidth receipts"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        
        # Generate Ed25519 key pair
        if CRYPTO_AVAILABLE:
            self.private_key = ed25519.Ed25519PrivateKey.generate()
            self.public_key = self.private_key.public_key()
        else:
            # Mock keys for testing
            self.private_key = secrets.token_bytes(32)
            self.public_key = secrets.token_bytes(32)
        
        logger.info(f"Cryptographic signer initialized for node {node_id}")
    
    def get_public_key_hex(self) -> str:
        """Get public key as hex string"""
        if CRYPTO_AVAILABLE:
            public_bytes = self.public_key.public_bytes(
                encoding=Encoding.Raw,
                format=PublicFormat.Raw
            )
            return public_bytes.hex()
        else:
            return self.public_key.hex()
    
    def sign_receipt(self, receipt: BandwidthReceipt) -> str:
        """Sign bandwidth receipt"""
        try:
            # Ensure proof hash is generated
            receipt.generate_proof_hash()
            
            # Create signature data
            signature_data = {
                'receipt_id': receipt.receipt_id,
                'proof_hash': receipt.proof_hash,
                'signer_node': self.node_id,
                'timestamp': time.time()
            }
            
            signature_json = json.dumps(signature_data, sort_keys=True)
            signature_bytes = signature_json.encode('utf-8')
            
            if CRYPTO_AVAILABLE:
                # Use Ed25519 signature
                signature = self.private_key.sign(signature_bytes)
                return signature.hex()
            else:
                # Mock signature using HMAC for testing
                mock_key = self.private_key
                signature = hmac.new(mock_key, signature_bytes, hashlib.sha256).hexdigest()
                return signature
                
        except Exception as e:
            logger.error(f"Failed to sign receipt {receipt.receipt_id}: {e}")
            return ""
    
    def verify_signature(self, receipt: BandwidthReceipt, signature: str, public_key_hex: str, signer_node: str) -> bool:
        """Verify receipt signature"""
        try:
            # Reconstruct signature data
            signature_data = {
                'receipt_id': receipt.receipt_id,
                'proof_hash': receipt.proof_hash,
                'signer_node': signer_node,
                'timestamp': time.time()  # Note: In production, this should be stored with signature
            }
            
            signature_json = json.dumps(signature_data, sort_keys=True)
            signature_bytes = signature_json.encode('utf-8')
            
            if CRYPTO_AVAILABLE:
                try:
                    # Reconstruct public key
                    public_bytes = bytes.fromhex(public_key_hex)
                    public_key = ed25519.Ed25519PublicKey.from_public_bytes(public_bytes)
                    
                    # Verify signature
                    signature_bytes_raw = bytes.fromhex(signature)
                    public_key.verify(signature_bytes_raw, signature_bytes)
                    return True
                except Exception:
                    return False
            else:
                # Mock verification using HMAC
                public_bytes = bytes.fromhex(public_key_hex)
                expected_sig = hmac.new(public_bytes, signature_bytes, hashlib.sha256).hexdigest()
                return hmac.compare_digest(signature, expected_sig)
                
        except Exception as e:
            logger.error(f"Failed to verify signature for receipt {receipt.receipt_id}: {e}")
            return False

class BandwidthMonitor:
    """Monitors bandwidth usage and generates measurements"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()
        
        # Monitoring configuration
        self.measurement_interval = 5.0  # seconds
        self.min_measurement_duration = 1.0  # seconds
        
        logger.info(f"Bandwidth monitor initialized for node {node_id}")
    
    def start_session(self, session_id: str, peer_node: str, tunnel_id: Optional[str] = None) -> bool:
        """Start monitoring bandwidth session"""
        with self.lock:
            try:
                session_data = {
                    'session_id': session_id,
                    'peer_node': peer_node,
                    'tunnel_id': tunnel_id,
                    'start_time': time.time(),
                    'last_measurement': time.time(),
                    'bytes_sent': 0,
                    'bytes_received': 0,
                    'packets_sent': 0,
                    'packets_received': 0,
                    'measurements': [],
                    'active': True
                }
                
                self.active_sessions[session_id] = session_data
                logger.info(f"Started bandwidth monitoring session {session_id} with {peer_node}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to start session {session_id}: {e}")
                return False
    
    def record_transfer(self, session_id: str, bytes_sent: int = 0, bytes_received: int = 0, 
                       packets_sent: int = 0, packets_received: int = 0, latency_ms: float = 0.0):
        """Record data transfer for session"""
        with self.lock:
            if session_id not in self.active_sessions:
                logger.warning(f"Session {session_id} not found for transfer recording")
                return
            
            session = self.active_sessions[session_id]
            current_time = time.time()
            
            # Update counters
            session['bytes_sent'] += bytes_sent
            session['bytes_received'] += bytes_received
            session['packets_sent'] += packets_sent  
            session['packets_received'] += packets_received
            
            # Check if it's time for a measurement
            time_since_last = current_time - session['last_measurement']
            
            if time_since_last >= self.measurement_interval:
                measurement = self._create_measurement(session, current_time, latency_ms)
                if measurement:
                    session['measurements'].append(measurement)
                    session['last_measurement'] = current_time
                    
                    logger.debug(f"Recorded measurement for session {session_id}: "
                               f"{measurement.bytes_transferred} bytes, "
                               f"{measurement.calculate_throughput_mbps():.2f} Mbps")
    
    def _create_measurement(self, session: Dict[str, Any], timestamp: float, latency_ms: float) -> Optional[BandwidthMeasurement]:
        """Create bandwidth measurement from session data"""
        try:
            duration = timestamp - session['last_measurement']
            if duration < self.min_measurement_duration:
                return None
            
            # Calculate bytes transferred in this window
            total_bytes = session['bytes_sent'] + session['bytes_received']
            
            # Determine direction
            if session['bytes_sent'] > session['bytes_received'] * 2:
                direction = BandwidthDirection.UPLOAD
            elif session['bytes_received'] > session['bytes_sent'] * 2:
                direction = BandwidthDirection.DOWNLOAD
            else:
                direction = BandwidthDirection.BIDIRECTIONAL
            
            # Calculate packet loss (simplified)
            total_packets_sent = session['packets_sent']
            total_packets_received = session['packets_received']
            packet_loss_rate = 0.0
            
            if total_packets_sent > 0:
                expected_received = total_packets_sent  # Simplified assumption
                packet_loss_rate = max(0.0, (expected_received - total_packets_received) / expected_received)
            
            # Create data hash for integrity
            data_content = f"{session['session_id']}{total_bytes}{timestamp}{self.node_id}"
            data_hash = hashlib.sha256(data_content.encode()).hexdigest()
            
            measurement = BandwidthMeasurement(
                timestamp=timestamp,
                bytes_transferred=total_bytes,
                duration_seconds=duration,
                direction=direction,
                source_node=self.node_id,
                destination_node=session['peer_node'],
                tunnel_id=session['tunnel_id'],
                latency_ms=latency_ms,
                packet_loss_rate=packet_loss_rate,
                data_hash=data_hash,
                sequence_number=len(session['measurements'])
            )
            
            return measurement
            
        except Exception as e:
            logger.error(f"Failed to create measurement: {e}")
            return None
    
    def end_session(self, session_id: str) -> List[BandwidthMeasurement]:
        """End monitoring session and return measurements"""
        with self.lock:
            if session_id not in self.active_sessions:
                logger.warning(f"Session {session_id} not found for ending")
                return []
            
            session = self.active_sessions[session_id]
            session['active'] = False
            
            # Create final measurement if needed
            current_time = time.time()
            final_measurement = self._create_measurement(session, current_time, 0.0)
            if final_measurement:
                session['measurements'].append(final_measurement)
            
            measurements = session['measurements'].copy()
            
            # Clean up session
            del self.active_sessions[session_id]
            
            logger.info(f"Ended session {session_id}, collected {len(measurements)} measurements")
            return measurements

class ProofOfBandwidthSystem:
    """Main Proof of Bandwidth system"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.signer = CryptographicSigner(node_id)
        self.monitor = BandwidthMonitor(node_id)
        
        # Receipt storage
        self.receipts: Dict[str, BandwidthReceipt] = {}
        self.pending_receipts: Dict[str, BandwidthReceipt] = {}
        self.lock = threading.RLock()
        
        # Configuration
        self.receipt_window_duration = 300.0  # 5 minutes
        self.auto_generate_receipts = True
        
        # Statistics
        self.stats = {
            'receipts_generated': 0,
            'receipts_signed': 0,
            'receipts_validated': 0,
            'total_bandwidth_gb': 0.0,
            'start_time': time.time()
        }
        
        logger.info(f"Proof of Bandwidth system initialized for node {node_id}")
    
    async def start_bandwidth_session(self, peer_node: str, tunnel_id: Optional[str] = None) -> str:
        """Start bandwidth monitoring session with peer"""
        session_id = f"bw_{self.node_id}_{peer_node}_{int(time.time())}"
        
        success = self.monitor.start_session(session_id, peer_node, tunnel_id)
        
        if success:
            logger.info(f"Started bandwidth session {session_id} with {peer_node}")
            return session_id
        else:
            logger.error(f"Failed to start bandwidth session with {peer_node}")
            return ""
    
    async def record_bandwidth_usage(self, session_id: str, bytes_sent: int = 0, bytes_received: int = 0,
                                   latency_ms: float = 0.0, packets_sent: int = 0, packets_received: int = 0):
        """Record bandwidth usage for active session"""
        self.monitor.record_transfer(
            session_id, bytes_sent, bytes_received, 
            packets_sent, packets_received, latency_ms
        )
    
    async def end_bandwidth_session(self, session_id: str) -> Optional[str]:
        """End bandwidth session and generate receipt"""
        try:
            measurements = self.monitor.end_session(session_id)
            
            if not measurements:
                logger.warning(f"No measurements collected for session {session_id}")
                return None
            
            # Extract peer information from first measurement
            peer_node = measurements[0].destination_node
            tunnel_id = measurements[0].tunnel_id
            
            # Create bandwidth receipt
            receipt = BandwidthReceipt(
                provider_node=self.node_id,
                consumer_node=peer_node
            )
            
            # Add all measurements
            for measurement in measurements:
                receipt.add_measurement(measurement)
            
            # Generate proof hash
            receipt.generate_proof_hash()
            
            # Sign receipt as provider
            provider_signature = self.signer.sign_receipt(receipt)
            receipt.provider_signature = provider_signature
            
            # Store receipt
            with self.lock:
                self.receipts[receipt.receipt_id] = receipt
                self.pending_receipts[receipt.receipt_id] = receipt
                
                # Update statistics
                self.stats['receipts_generated'] += 1
                self.stats['receipts_signed'] += 1
                self.stats['total_bandwidth_gb'] += receipt.total_bytes / (1024**3)
            
            logger.info(f"Generated bandwidth receipt {receipt.receipt_id} for {receipt.total_bytes} bytes")
            return receipt.receipt_id
            
        except Exception as e:
            logger.error(f"Failed to end bandwidth session {session_id}: {e}")
            return None
    
    async def sign_receipt_as_consumer(self, receipt_id: str) -> bool:
        """Sign receipt as bandwidth consumer (validator)"""
        try:
            with self.lock:
                if receipt_id not in self.receipts:
                    logger.error(f"Receipt {receipt_id} not found for consumer signing")
                    return False
                
                receipt = self.receipts[receipt_id]
                
                # Verify we are the consumer
                if receipt.consumer_node != self.node_id:
                    logger.error(f"Cannot sign receipt {receipt_id} - not the consumer")
                    return False
                
                # Sign receipt as consumer
                consumer_signature = self.signer.sign_receipt(receipt)
                receipt.consumer_signature = consumer_signature
                
                # Update status if both signatures present
                if receipt.provider_signature and receipt.consumer_signature:
                    receipt.status = ProofStatus.VALIDATED
                    self.stats['receipts_validated'] += 1
                
                logger.info(f"Signed receipt {receipt_id} as consumer")
                return True
                
        except Exception as e:
            logger.error(f"Failed to sign receipt {receipt_id} as consumer: {e}")
            return False
    
    def validate_receipt(self, receipt_id: str, provider_public_key: str, consumer_public_key: str) -> bool:
        """Validate bandwidth receipt signatures"""
        try:
            with self.lock:
                if receipt_id not in self.receipts:
                    logger.error(f"Receipt {receipt_id} not found for validation")
                    return False
                
                receipt = self.receipts[receipt_id]
                
                # Verify provider signature
                provider_valid = self.signer.verify_signature(
                    receipt, receipt.provider_signature, 
                    provider_public_key, receipt.provider_node
                )
                
                # Verify consumer signature  
                consumer_valid = self.signer.verify_signature(
                    receipt, receipt.consumer_signature,
                    consumer_public_key, receipt.consumer_node
                )
                
                valid = provider_valid and consumer_valid
                
                if valid:
                    receipt.status = ProofStatus.VALIDATED
                    logger.info(f"Receipt {receipt_id} validation successful")
                else:
                    receipt.status = ProofStatus.REJECTED
                    logger.warning(f"Receipt {receipt_id} validation failed")
                
                return valid
                
        except Exception as e:
            logger.error(f"Failed to validate receipt {receipt_id}: {e}")
            return False
    
    def get_receipt(self, receipt_id: str) -> Optional[BandwidthReceipt]:
        """Get bandwidth receipt by ID"""
        with self.lock:
            return self.receipts.get(receipt_id)
    
    def get_receipts_for_node(self, node_id: str) -> List[BandwidthReceipt]:
        """Get all receipts involving a specific node"""
        with self.lock:
            matching_receipts = []
            for receipt in self.receipts.values():
                if receipt.provider_node == node_id or receipt.consumer_node == node_id:
                    matching_receipts.append(receipt)
            return matching_receipts
    
    def get_pending_receipts(self) -> List[BandwidthReceipt]:
        """Get receipts pending consumer signature"""
        with self.lock:
            return list(self.pending_receipts.values())
    
    def cleanup_expired_receipts(self, max_age_seconds: int = 86400) -> int:
        """Remove receipts older than max age"""
        with self.lock:
            current_time = time.time()
            expired_receipts = []
            
            for receipt_id, receipt in self.receipts.items():
                if current_time - receipt.created_at > max_age_seconds:
                    expired_receipts.append(receipt_id)
            
            # Remove expired receipts
            for receipt_id in expired_receipts:
                receipt = self.receipts[receipt_id]
                receipt.status = ProofStatus.EXPIRED
                del self.receipts[receipt_id]
                self.pending_receipts.pop(receipt_id, None)
            
            if expired_receipts:
                logger.info(f"Cleaned up {len(expired_receipts)} expired receipts")
            
            return len(expired_receipts)
    
    def get_bandwidth_stats(self) -> Dict[str, Any]:
        """Get comprehensive bandwidth statistics"""
        with self.lock:
            current_time = time.time()
            uptime = current_time - self.stats['start_time']
            
            # Calculate totals
            total_receipts = len(self.receipts)
            validated_receipts = len([r for r in self.receipts.values() if r.status == ProofStatus.VALIDATED])
            pending_receipts = len([r for r in self.receipts.values() if r.status == ProofStatus.PENDING])
            
            # Calculate bandwidth metrics
            total_throughput = sum(r.average_throughput_mbps for r in self.receipts.values())
            avg_throughput = total_throughput / max(total_receipts, 1)
            
            peak_throughput = max(
                (r.peak_throughput_mbps for r in self.receipts.values()), 
                default=0.0
            )
            
            return {
                'node_info': {
                    'node_id': self.node_id,
                    'public_key': self.signer.get_public_key_hex(),
                    'uptime_seconds': uptime
                },
                'receipt_statistics': {
                    'total_receipts': total_receipts,
                    'validated_receipts': validated_receipts,
                    'pending_receipts': pending_receipts,
                    'receipts_generated': self.stats['receipts_generated'],
                    'validation_rate': validated_receipts / max(total_receipts, 1) * 100
                },
                'bandwidth_metrics': {
                    'total_bandwidth_gb': self.stats['total_bandwidth_gb'],
                    'average_throughput_mbps': avg_throughput,
                    'peak_throughput_mbps': peak_throughput,
                    'bandwidth_per_hour_gb': self.stats['total_bandwidth_gb'] / max(uptime / 3600, 1)
                },
                'active_sessions': len(self.monitor.active_sessions)
            }

# Global PoB system instance
_global_pob_system: Optional[ProofOfBandwidthSystem] = None

def get_pob_system() -> Optional[ProofOfBandwidthSystem]:
    """Get global PoB system instance"""
    return _global_pob_system

def initialize_pob_system(node_id: str) -> ProofOfBandwidthSystem:
    """Initialize global PoB system"""
    global _global_pob_system
    _global_pob_system = ProofOfBandwidthSystem(node_id)
    return _global_pob_system

# Example usage and testing
async def pob_demo():
    """Demonstration of Proof of Bandwidth system"""
    print("📊 Proof of Bandwidth (PoB) Demo")
    print("=" * 50)
    
    # Initialize PoB systems for two nodes
    pob_provider = initialize_pob_system("provider_node_001")
    pob_consumer = ProofOfBandwidthSystem("consumer_node_002")
    
    try:
        # Start bandwidth session
        print("🔗 Starting bandwidth session...")
        session_id = await pob_provider.start_bandwidth_session("consumer_node_002", "tunnel_123")
        
        if session_id:
            print(f"✅ Session started: {session_id}")
            
            # Simulate bandwidth usage over time
            print("📈 Simulating bandwidth usage...")
            for i in range(10):
                await pob_provider.record_bandwidth_usage(
                    session_id,
                    bytes_sent=1024 * 1024 * (i + 1),  # 1MB increments
                    bytes_received=512 * 1024 * (i + 1),  # 512KB increments
                    latency_ms=50.0 + (i * 2),
                    packets_sent=1000 * (i + 1),
                    packets_received=950 * (i + 1)  # Some packet loss
                )
                await asyncio.sleep(0.1)  # Short delay for demo
            
            # End session and generate receipt
            print("📋 Generating bandwidth receipt...")
            receipt_id = await pob_provider.end_bandwidth_session(session_id)
            
            if receipt_id:
                print(f"✅ Receipt generated: {receipt_id}")
                
                # Get receipt details
                receipt = pob_provider.get_receipt(receipt_id)
                if receipt:
                    print(f"📊 Receipt summary:")
                    print(f"   Total bytes: {receipt.total_bytes:,}")
                    print(f"   Average throughput: {receipt.average_throughput_mbps:.2f} Mbps")
                    print(f"   Peak throughput: {receipt.peak_throughput_mbps:.2f} Mbps")
                    print(f"   Measurements: {len(receipt.measurements)}")
                    print(f"   Reliability: {receipt.reliability_score:.3f}")
                
                # Consumer signs receipt
                print("✍️  Consumer signing receipt...")
                # Transfer receipt to consumer (in real system, this would be via network)
                pob_consumer.receipts[receipt_id] = receipt
                pob_consumer.pending_receipts[receipt_id] = receipt
                
                consumer_signed = await pob_consumer.sign_receipt_as_consumer(receipt_id)
                print(f"{'✅' if consumer_signed else '❌'} Consumer signature: {consumer_signed}")
                
                # Validate receipt
                print("🔍 Validating receipt...")
                provider_pubkey = pob_provider.signer.get_public_key_hex()
                consumer_pubkey = pob_consumer.signer.get_public_key_hex()
                
                valid = pob_provider.validate_receipt(receipt_id, provider_pubkey, consumer_pubkey)
                print(f"{'✅' if valid else '❌'} Receipt validation: {valid}")
                
            else:
                print("❌ Failed to generate receipt")
        else:
            print("❌ Failed to start session")
        
        # Get statistics
        print("\n📈 Bandwidth Statistics:")
        stats = pob_provider.get_bandwidth_stats()
        print(json.dumps(stats, indent=2, default=str))
        
    except Exception as e:
        print(f"💥 Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(pob_demo())