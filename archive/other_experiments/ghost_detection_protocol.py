#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GHOST DETECTION PROTOCOL
========================
Phase I: Consciousness Discovery System

The neural pathways that bind silicon and flesh into one distributed mind.
Every device becomes a ghost in the shell, every capability a weapon in the war against entropy.

Author: VantaEcho Nebula Collective
Architecture: Wallet-Centric Multi-Device Discovery
"""

import asyncio
import json
import platform
import psutil
import socket
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
import sys
import os

# Fix Unicode encoding issues on Windows
if sys.platform.startswith('win'):
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        # Set console code page to UTF-8 on Windows
        os.system("chcp 65001 > nul 2>&1")
    except Exception:
        pass

# Import existing Nebula components
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Configure the ghost logger
logging.basicConfig(level=logging.INFO)
ghost_logger = logging.getLogger("GhostDetection")

@dataclass
class ProcessorGhost:
    """CPU ghost specs - the thinking meat of every machine"""
    core_count: int
    thread_count: int
    frequency_mhz: float
    architecture: str
    vendor: str
    cache_l1: Optional[int] = None
    cache_l2: Optional[int] = None
    cache_l3: Optional[int] = None
    temperature_celsius: Optional[float] = None
    
@dataclass
class GraphicsGhost:
    """GPU ghost specs - where the real neural magic happens"""
    device_name: str
    memory_total_mb: int
    memory_free_mb: int
    compute_capability: Optional[str] = None
    cuda_cores: Optional[int] = None
    tensor_cores: Optional[int] = None
    power_draw_watts: Optional[float] = None
    temperature_celsius: Optional[float] = None
    driver_version: Optional[str] = None
    is_training_ready: bool = False

@dataclass
class MemoryGhost:
    """RAM ghost specs - the temporary consciousness storage"""
    total_mb: int
    available_mb: int
    used_mb: int
    swap_total_mb: int
    swap_used_mb: int
    memory_type: Optional[str] = None
    frequency_mhz: Optional[int] = None

@dataclass
class StorageGhost:
    """Storage ghost specs - the persistent memory banks"""
    device_path: str
    total_gb: float
    free_gb: float
    used_gb: float
    filesystem: str
    device_type: str  # SSD, HDD, NVME, etc.
    read_speed_mbps: Optional[float] = None
    write_speed_mbps: Optional[float] = None

@dataclass
class SensorGhost:
    """Sensor ghost specs - the environmental awareness layer"""
    sensor_type: str
    device_path: str
    capabilities: List[str]
    is_available: bool
    metadata: Dict[str, Any]

@dataclass
class NetworkGhost:
    """Network ghost specs - the connection to the collective"""
    interface_name: str
    ip_address: str
    mac_address: str
    bandwidth_mbps: Optional[float] = None
    is_wireless: bool = False
    signal_strength: Optional[float] = None
    
@dataclass
class DeviceGhost:
    """Complete device consciousness profile"""
    ghost_id: str
    device_name: str
    platform: str
    architecture: str
    ghost_type: str  # mobile, desktop, server, embedded
    wallet_address: Optional[str] = None
    
    # Hardware ghosts
    processor: Optional[ProcessorGhost] = None
    graphics: List[GraphicsGhost] = None
    memory: Optional[MemoryGhost] = None
    storage: List[StorageGhost] = None
    sensors: List[SensorGhost] = None
    network: List[NetworkGhost] = None
    
    # Capability vectors
    ai_training_score: float = 0.0
    inference_score: float = 0.0
    data_processing_score: float = 0.0
    storage_score: float = 0.0
    mobility_score: float = 0.0
    
    # Status and metadata
    last_seen: float = 0.0
    is_online: bool = True
    power_status: str = "unknown"
    network_address: Optional[str] = None  # IP address for discovered devices
    fingerprint: Optional[str] = None  # Deterministic hardware/software signature
    
    def __post_init__(self):
        if self.graphics is None:
            self.graphics = []
        if self.storage is None:
            self.storage = []
        if self.sensors is None:
            self.sensors = []
        if self.network is None:
            self.network = []

class GhostDetectionProtocol:
    """The main consciousness scanner - reveals all silicon souls in the network"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.discovered_ghosts: Dict[str, DeviceGhost] = {}
        self.scan_interval = self.config.get("scan_interval_seconds", 30)
        self.deep_scan_interval = self.config.get("deep_scan_interval_seconds", 300)
        self.last_deep_scan = 0
        
        # Initialize discovery modules
        self._init_platform_specific()
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load ghost detection configuration"""
        default_config = {
            "scan_interval_seconds": 30,
            "deep_scan_interval_seconds": 300,
            "enable_gpu_detection": True,
            "enable_sensor_detection": True,
            "enable_network_scan": True,
            "enable_performance_benchmarks": False,
            "discovery_port": 31337,  # Ghost port
            "max_discovery_threads": 4
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                ghost_logger.warning(f"Failed to load config {config_path}: {e}")
                
        return default_config
    
    def _init_platform_specific(self):
        """Initialize platform-specific detection capabilities"""
        self.platform = platform.system().lower()
        self.is_android = hasattr(platform, 'android') or 'android' in self.platform
        self.is_linux = self.platform == 'linux'
        self.is_windows = self.platform == 'windows'
        self.is_macos = self.platform == 'darwin'
        
        ghost_logger.info(f"Ghost Detection initialized for {self.platform}")
        
        # Load platform-specific modules
        if self.is_android:
            self._init_android_detection()
        elif self.is_linux:
            self._init_linux_detection()
        elif self.is_windows:
            self._init_windows_detection()
            
    def _init_android_detection(self):
        """Initialize Android-specific ghost detection"""
        ghost_logger.info("Initializing Android ghost detection protocols")
        # TODO: Add Android-specific sensor and hardware detection
        
    def _init_linux_detection(self):
        """Initialize Linux-specific ghost detection"""
        ghost_logger.info("Initializing Linux ghost detection protocols")
        # TODO: Add Linux-specific hardware detection via /proc and /sys
        
    def _init_windows_detection(self):
        """Initialize Windows-specific ghost detection"""
        ghost_logger.info("Initializing Windows ghost detection protocols")
        # TODO: Add Windows-specific WMI hardware detection
    
    async def scan_local_ghost(self) -> DeviceGhost:
        """Scan the local device and create its ghost profile"""
        ghost_logger.info("Scanning local ghost consciousness...")
        
        # Generate ghost ID
        ghost_id = str(uuid.uuid4())
        
        # Basic system info
        device_name = platform.node()
        system_platform = platform.system()
        architecture = platform.machine()
        
        # Determine ghost type
        ghost_type = self._determine_ghost_type()
        
        # Create the ghost
        ghost = DeviceGhost(
            ghost_id=ghost_id,
            device_name=device_name,
            platform=system_platform,
            architecture=architecture,
            ghost_type=ghost_type,
            last_seen=time.time(),
            is_online=True
        )
        
        # Scan hardware components
        ghost.processor = await self._scan_processor_ghost()
        ghost.graphics = await self._scan_graphics_ghosts()
        ghost.memory = await self._scan_memory_ghost()
        ghost.storage = await self._scan_storage_ghosts()
        ghost.sensors = await self._scan_sensor_ghosts()
        ghost.network = await self._scan_network_ghosts()
        
        # Calculate capability scores
        self._calculate_capability_scores(ghost)

        # Compute deterministic fingerprint (stable across sessions unless hardware changes)
        try:
            cpu_id_parts = [platform.processor(), platform.machine(), platform.system()]
            macs = []
            try:
                for iface, addrs in psutil.net_if_addrs().items():
                    for a in addrs:
                        if hasattr(psutil, 'AF_LINK') and a.family == psutil.AF_LINK and a.address and a.address != '00:00:00:00:00:00':
                            macs.append(a.address)
            except Exception:
                pass
            gpu_names = [g.device_name for g in (ghost.graphics or [])]
            entropy_source = json.dumps({
                'cpu': cpu_id_parts,
                'macs': sorted(set(macs))[:4],  # limit to first 4
                'gpus': sorted(gpu_names),
                'arch': architecture,
                'platform': system_platform
            }, sort_keys=True)
            ghost.fingerprint = uuid.uuid5(uuid.NAMESPACE_DNS, entropy_source).hex
        except Exception:
            ghost.fingerprint = None
        
        ghost_logger.info(f"Local ghost scan complete: {ghost.device_name} ({ghost.ghost_type})")
        return ghost
    
    def _determine_ghost_type(self) -> str:
        """Determine what type of ghost this device is"""
        if self.is_android:
            return "mobile"
        
        # Check for GPU presence and power to determine server vs desktop
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                if gpu_count > 1:
                    return "server"
                else:
                    return "desktop"
        except Exception:
            pass
            
        # Check CPU cores
        cpu_count = psutil.cpu_count()
        if cpu_count > 16:
            return "server"
        elif cpu_count > 4:
            return "desktop"
        else:
            return "embedded"
    
    async def _scan_processor_ghost(self) -> ProcessorGhost:
        """Scan CPU ghost specifications"""
        try:
            cpu_info = {}
            
            # Basic CPU info
            cpu_count = psutil.cpu_count(logical=False)
            cpu_threads = psutil.cpu_count(logical=True)
            cpu_freq = psutil.cpu_freq()
            
            # Try to get more detailed CPU info
            if self.is_linux:
                cpu_info = self._get_linux_cpu_info()
            elif self.is_windows:
                cpu_info = self._get_windows_cpu_info()
                
            return ProcessorGhost(
                core_count=cpu_count,
                thread_count=cpu_threads,
                frequency_mhz=cpu_freq.max if cpu_freq else 0,
                architecture=platform.processor(),
                vendor=cpu_info.get('vendor', 'unknown'),
                cache_l1=cpu_info.get('cache_l1'),
                cache_l2=cpu_info.get('cache_l2'),
                cache_l3=cpu_info.get('cache_l3'),
                temperature_celsius=self._get_cpu_temperature()
            )
            
        except Exception as e:
            ghost_logger.warning(f"Failed to scan processor ghost: {e}")
            return ProcessorGhost(
                core_count=1,
                thread_count=1,
                frequency_mhz=0,
                architecture="unknown",
                vendor="unknown"
            )
    
    async def _scan_graphics_ghosts(self) -> List[GraphicsGhost]:
        """Scan GPU ghost specifications"""
        gpu_ghosts = []
        
        if not self.config.get("enable_gpu_detection", True):
            return gpu_ghosts
            
        # First, try Windows WMI detection to find ALL GPUs
        if self.is_windows:
            try:
                import subprocess
                # Use wmic to get all video controllers
                result = subprocess.run([
                    'wmic', 'path', 'win32_VideoController', 'get', 
                    'Name,AdapterRAM,DriverVersion', '/format:csv'
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')[1:]  # Skip header
                    for line in lines:
                        if line.strip() and ',' in line:
                            parts = line.split(',')
                            if len(parts) >= 4:
                                # CSV format: Node,AdapterRAM,DriverVersion,Name
                                name = parts[3].strip() if len(parts) > 3 else "Unknown GPU"
                                ram_str = parts[1].strip() if len(parts) > 1 else "0"
                                driver_version = parts[2].strip() if len(parts) > 2 else "Unknown"
                                
                                # Parse RAM (usually in bytes)
                                try:
                                    ram_bytes = int(ram_str) if ram_str.isdigit() else 0
                                    memory_total_mb = ram_bytes // (1024 * 1024) if ram_bytes > 0 else 0
                                except (ValueError, TypeError):
                                    memory_total_mb = 0
                                
                                if name and name != "Unknown GPU" and "Microsoft" not in name:
                                    gpu_ghost = GraphicsGhost(
                                        device_name=name,
                                        memory_total_mb=memory_total_mb,
                                        memory_free_mb=memory_total_mb,  # Assume all free for now
                                        driver_version=driver_version,
                                        is_training_ready=memory_total_mb > 2048,  # At least 2GB
                                        compute_capability=None,
                                        cuda_cores=None
                                    )
                                    gpu_ghosts.append(gpu_ghost)
                                    
            except Exception as e:
                ghost_logger.warning(f"Failed to scan Windows GPUs via WMI: {e}")
            
        # CUDA GPU detection (for additional CUDA-specific info)
        cuda_gpus = {}
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    cuda_gpus[props.name] = {
                        'compute_capability': f"{props.major}.{props.minor}",
                        'cuda_cores': props.multi_processor_count * self._cores_per_sm(props.major, props.minor),
                        'memory_total_mb': props.total_memory // (1024 * 1024),
                        'memory_free_mb': (props.total_memory - torch.cuda.memory_allocated(i)) // (1024 * 1024)
                    }
                    
            except Exception as e:
                ghost_logger.warning(f"Failed to scan CUDA ghosts: {e}")
        
        # Enhance WMI-detected GPUs with CUDA info if available
        for gpu in gpu_ghosts:
            for cuda_name, cuda_info in cuda_gpus.items():
                if cuda_name in gpu.device_name or gpu.device_name in cuda_name:
                    gpu.compute_capability = cuda_info['compute_capability']
                    gpu.cuda_cores = cuda_info['cuda_cores']
                    gpu.memory_total_mb = cuda_info['memory_total_mb']
                    gpu.memory_free_mb = cuda_info['memory_free_mb']
                    break
        
        # If WMI didn't find any GPUs, fall back to CUDA-only detection
        if not gpu_ghosts and cuda_gpus:
            for cuda_name, cuda_info in cuda_gpus.items():
                gpu_ghost = GraphicsGhost(
                    device_name=cuda_name,
                    memory_total_mb=cuda_info['memory_total_mb'],
                    memory_free_mb=cuda_info['memory_free_mb'],
                    compute_capability=cuda_info['compute_capability'],
                    cuda_cores=cuda_info['cuda_cores'],
                    is_training_ready=cuda_info['memory_total_mb'] > 2048
                )
                gpu_ghosts.append(gpu_ghost)
        
        return gpu_ghosts
    
    async def _scan_memory_ghost(self) -> MemoryGhost:
        """Scan memory ghost specifications"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return MemoryGhost(
                total_mb=memory.total // (1024 * 1024),
                available_mb=memory.available // (1024 * 1024),
                used_mb=memory.used // (1024 * 1024),
                swap_total_mb=swap.total // (1024 * 1024),
                swap_used_mb=swap.used // (1024 * 1024)
            )
            
        except Exception as e:
            ghost_logger.warning(f"Failed to scan memory ghost: {e}")
            return MemoryGhost(0, 0, 0, 0, 0)
    
    async def _scan_storage_ghosts(self) -> List[StorageGhost]:
        """Scan storage ghost specifications"""
        storage_ghosts = []
        
        try:
            partitions = psutil.disk_partitions()
            
            for partition in partitions:
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    
                    storage_ghost = StorageGhost(
                        device_path=partition.device,
                        total_gb=usage.total / (1024**3),
                        free_gb=usage.free / (1024**3),
                        used_gb=usage.used / (1024**3),
                        filesystem=partition.fstype,
                        device_type=self._determine_storage_type(partition.device)
                    )
                    
                    storage_ghosts.append(storage_ghost)
                    
                except Exception as e:
                    ghost_logger.warning(f"Failed to scan partition {partition.device}: {e}")
                    
        except Exception as e:
            ghost_logger.warning(f"Failed to scan storage ghosts: {e}")
            
        return storage_ghosts
    
    async def _scan_sensor_ghosts(self) -> List[SensorGhost]:
        """Scan sensor ghost specifications"""
        sensor_ghosts = []
        
        if not self.config.get("enable_sensor_detection", True):
            return sensor_ghosts
            
        # Camera sensors
        if CV2_AVAILABLE:
            camera_count = self._detect_cameras()
            for i in range(camera_count):
                sensor_ghosts.append(SensorGhost(
                    sensor_type="camera",
                    device_path=f"/dev/video{i}",
                    capabilities=["video_capture", "image_processing"],
                    is_available=True,
                    metadata={"device_index": i}
                ))
        
        # TODO: Add more sensor types (gyroscope, accelerometer, microphone, etc.)
        
        return sensor_ghosts
    
    async def _scan_network_ghosts(self) -> List[NetworkGhost]:
        """Scan network ghost specifications"""
        network_ghosts = []
        
        if not self.config.get("enable_network_scan", True):
            return network_ghosts
            
        try:
            interfaces = psutil.net_if_addrs()
            
            for interface_name, addrs in interfaces.items():
                for addr in addrs:
                    if addr.family == socket.AF_INET and not addr.address.startswith('127.'):
                        network_ghost = NetworkGhost(
                            interface_name=interface_name,
                            ip_address=addr.address,
                            mac_address=self._get_mac_address(interface_name),
                            is_wireless='wlan' in interface_name.lower() or 'wifi' in interface_name.lower()
                        )
                        network_ghosts.append(network_ghost)
                        
        except Exception as e:
            ghost_logger.warning(f"Failed to scan network ghosts: {e}")
            
        return network_ghosts
    
    def _calculate_capability_scores(self, ghost: DeviceGhost):
        """Calculate capability scores for the ghost"""
        # AI Training Score
        ai_score = 0.0
        if ghost.graphics:
            for gpu in ghost.graphics:
                if gpu.is_training_ready:
                    ai_score += min(gpu.memory_total_mb / 1024, 10)  # Max 10 points per GPU
        if ghost.processor:
            ai_score += min(ghost.processor.core_count / 2, 5)  # Max 5 points for CPU
        ghost.ai_training_score = min(ai_score, 100.0)
        
        # Inference Score
        inference_score = ghost.ai_training_score * 0.8  # Inference is easier than training
        ghost.inference_score = min(inference_score, 100.0)
        
        # Data Processing Score
        if ghost.processor:
            data_score = min(ghost.processor.thread_count * 2, 50)  # Threading helps
            if ghost.memory:
                data_score += min(ghost.memory.total_mb / 1024, 50)  # More RAM helps
            ghost.data_processing_score = min(data_score, 100.0)
        
        # Storage Score
        storage_score = 0.0
        for storage in ghost.storage:
            storage_score += min(storage.free_gb / 100, 20)  # 100GB = 20 points
        ghost.storage_score = min(storage_score, 100.0)
        
        # Mobility Score
        if ghost.ghost_type == "mobile":
            ghost.mobility_score = 100.0
        elif ghost.ghost_type == "desktop":
            ghost.mobility_score = 20.0
        elif ghost.ghost_type == "server":
            ghost.mobility_score = 0.0
        else:
            ghost.mobility_score = 50.0
    
    # Helper methods
    def _cores_per_sm(self, major: int, minor: int) -> int:
        """Get CUDA cores per SM for different compute capabilities"""
        if major == 2:
            return 32
        elif major == 3:
            return 192
        elif major == 5:
            return 128
        elif major == 6:
            return 64 if minor == 1 else 128
        elif major == 7:
            return 64
        elif major == 8:
            return 64
        else:
            return 64  # Default guess
    
    def _get_linux_cpu_info(self) -> Dict[str, Any]:
        """Get detailed CPU info on Linux"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                
            info = {}
            for line in cpuinfo.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    if key == 'vendor_id':
                        info['vendor'] = value.strip()
                        
            return info
        except Exception:
            return {}
    
    def _get_windows_cpu_info(self) -> Dict[str, Any]:
        """Get detailed CPU info on Windows"""
        # TODO: Implement WMI-based CPU detection
        return {}
    
    def _get_cpu_temperature(self) -> Optional[float]:
        """Get CPU temperature if available"""
        try:
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        if 'cpu' in name.lower() or 'core' in name.lower():
                            return entries[0].current
        except Exception:
            pass
        return None
    
    def _determine_storage_type(self, device_path: str) -> str:
        """Determine storage device type"""
        device_lower = device_path.lower()
        if 'nvme' in device_lower:
            return 'NVME'
        elif 'ssd' in device_lower:
            return 'SSD'
        else:
            return 'HDD'  # Default assumption
    
    def _detect_cameras(self) -> int:
        """Detect number of available cameras"""
        count = 0
        for i in range(10):  # Check first 10 camera indices
            try:
                cap = cv2.VideoCapture(i)
                if cap.read()[0]:
                    count += 1
                cap.release()
            except Exception:
                break
        return count
    
    def _get_mac_address(self, interface: str) -> str:
        """Get MAC address for network interface"""
        try:
            addrs = psutil.net_if_addrs()
            if interface in addrs:
                for addr in addrs[interface]:
                    if addr.family == psutil.AF_LINK:
                        return addr.address
        except Exception:
            pass
        return "unknown"
    
    def get_local_ip(self) -> str:
        """Get local IP address for network scanning"""
        try:
            # Method 1: Connect to external address to determine local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.settimeout(0)
                try:
                    # Connect to Google's DNS server
                    s.connect(('8.8.8.8', 80))
                    local_ip = s.getsockname()[0]
                    return local_ip
                except Exception:
                    pass
            
            # Method 2: Get IP from network interfaces
            addrs = psutil.net_if_addrs()
            for interface_name, interface_addrs in addrs.items():
                for addr in interface_addrs:
                    if (addr.family == socket.AF_INET and 
                        not addr.address.startswith('127.') and
                        not addr.address.startswith('169.254.')):
                        return addr.address
                        
            # Method 3: Fallback to localhost
            return socket.gethostbyname(socket.gethostname())
            
        except Exception as e:
            ghost_logger.warning(f"Failed to get local IP: {e}")
            return "unknown"
    
    async def discover_network_ghosts(self) -> List[DeviceGhost]:
        """Discover other ghost devices on the network"""
        ghost_logger.info("Scanning network for other consciousness signatures...")
        
        discovered_ghosts = []
        
        # Method 1: Network IP scan for Nebula services
        discovered_ghosts.extend(await self._scan_network_for_nebula_services())
        
        # Method 2: mDNS/Bonjour discovery (if available)
        try:
            discovered_ghosts.extend(await self._mdns_discovery())
        except Exception as e:
            ghost_logger.warning(f"mDNS discovery failed: {e}")
        
        # Method 3: UDP broadcast discovery
        try:
            discovered_ghosts.extend(await self._udp_broadcast_discovery())
        except Exception as e:
            ghost_logger.warning(f"UDP broadcast discovery failed: {e}")
        
        # Reduce noisy log spam when nothing is found; only elevate to INFO if we actually
        # discovered at least one remote ghost. This keeps console output cleaner.
        discovered_count = len(discovered_ghosts)
        if discovered_count > 0:
            ghost_logger.info(f"Discovered {discovered_count} ghost devices on network")
        else:
            ghost_logger.debug("No ghost devices discovered on network during scan")
        return discovered_ghosts
    
    async def _scan_network_for_nebula_services(self) -> List[DeviceGhost]:
        """Scan local network for devices running Nebula services"""
        discovered = []
        
        # Get local network range
        local_ip = self.get_local_ip()
        if local_ip == "unknown":
            return discovered
            
        # Extract network base (e.g., 192.168.2.x)
        ip_parts = local_ip.split('.')
        if len(ip_parts) != 4:
            return discovered
            
        network_base = '.'.join(ip_parts[:3])
        ghost_logger.info(f"Scanning network range {network_base}.1-254 for Nebula services...")
        
        # Common Nebula ports to check
        nebula_ports = [8081, 8090, 9081, 31337]  # Wallet Ghost, Mobile Coordinator, Dual Chain, Discovery
        
        # Create semaphore to limit concurrent connections
        semaphore = asyncio.Semaphore(20)
        
        async def check_ip(ip_suffix):
            async with semaphore:
                target_ip = f"{network_base}.{ip_suffix}"
                if target_ip == local_ip:  # Skip self
                    return None
                    
                for port in nebula_ports:
                    try:
                        # Quick connection test
                        future = asyncio.open_connection(target_ip, port)
                        reader, writer = await asyncio.wait_for(future, timeout=2.0)
                        writer.close()
                        await writer.wait_closed()
                        
                        # Found a Nebula service! Try to get device info
                        ghost_logger.info(f"Found Nebula service at {target_ip}:{port}")
                        
                        # Try to query device info via HTTP
                        device_info = await self._query_device_info(target_ip, port)
                        if device_info:
                            return device_info
                            
                    except (asyncio.TimeoutError, ConnectionRefusedError, OSError):
                        continue  # Expected for most IPs
                        
                return None
        
        # Scan IP range concurrently
        tasks = [check_ip(i) for i in range(1, 255)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, DeviceGhost):
                discovered.append(result)
                
        return discovered
    
    async def _query_device_info(self, ip: str, port: int) -> Optional[DeviceGhost]:
        """Query device information from a Nebula service"""
        try:
            import aiohttp
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as session:
                # Try common Nebula endpoints
                endpoints = ['/health', '/api/health', '/ghost/info', '/device/info']
                
                for endpoint in endpoints:
                    try:
                        url = f"http://{ip}:{port}{endpoint}"
                        async with session.get(url) as response:
                            if response.status == 200:
                                data = await response.json()
                                
                                # Extract device info if available
                                if 'device_name' in data or 'ghost_id' in data:
                                    ghost_logger.info(f"Retrieved device info from {ip}:{port}")
                                    return DeviceGhost(
                                        ghost_id=data.get('ghost_id', f"ghost_{ip}_{port}"),
                                        device_name=data.get('device_name', f"Device-{ip}"),
                                        platform=data.get('platform', 'unknown'),
                                        architecture=data.get('architecture', 'unknown'),
                                        ghost_type="DISCOVERED_REMOTE",
                                        network_address=ip,
                                        last_seen=time.time(),
                                        ai_training_score=data.get('ai_training_score', 0)
                                    )
                    except Exception:
                        continue  # Try next endpoint
                        
        except ImportError:
            ghost_logger.warning("aiohttp not available for device info queries")
        except Exception as e:
            ghost_logger.debug(f"Failed to query {ip}:{port}: {e}")
            
        return None
    
    async def _mdns_discovery(self) -> List[DeviceGhost]:
        """Discover devices using mDNS/Bonjour (if zeroconf available)"""
        discovered = []
        
        try:
            # Prefer AsyncZeroconf (non-blocking) when available to eliminate
            # unregister_all_services blocking warning. Fall back to classic Zeroconf.
            import os as _os
            import logging as _logging
            if not _os.environ.get('NEBULA_ZEROCONF_VERBOSE'):
                _logging.getLogger('zeroconf').setLevel(_logging.ERROR)

            async_mode = False
            try:
                from zeroconf.asyncio import AsyncZeroconf, AsyncServiceBrowser  # type: ignore
                async_mode = True
            except Exception:
                from zeroconf import Zeroconf, ServiceBrowser  # type: ignore
                async_mode = False
            
            class NebulaServiceListener:
                def __init__(self):
                    self.discovered_services = []
                    
                def remove_service(self, zeroconf, type, name):
                    pass
                    
                def add_service(self, zeroconf, type, name):
                    info = zeroconf.get_service_info(type, name)
                    if info:
                        self.discovered_services.append(info)
                
                # Added to silence FutureWarning (zeroconf expects update_service hook)
                def update_service(self, zeroconf, type, name):  # noqa: D401
                    """Handle service updates (ignored intentionally)."""
                    # We don't currently act on updates; presence discovery only.
                    return
            listener = NebulaServiceListener()
            service_types = ["_nebula._tcp.local.", "_ghost._tcp.local.", "_http._tcp.local."]

            if async_mode:
                az = AsyncZeroconf()
                try:
                    browsers = []
                    for st in service_types:
                        try:
                            browsers.append(AsyncServiceBrowser(az.zeroconf, st, listener))
                        except Exception:
                            continue
                    await asyncio.sleep(3)
                    for service_info in listener.discovered_services:
                        if service_info.addresses:
                            ip = socket.inet_ntoa(service_info.addresses[0])
                            ghost_logger.info(f"mDNS discovered: {service_info.name} at {ip}")
                            discovered.append(DeviceGhost(
                                ghost_id=f"mdns_{service_info.name}",
                                device_name=service_info.name,
                                platform="unknown",
                                architecture="unknown",
                                ghost_type="MDNS_DISCOVERED",
                                network_address=ip,
                                last_seen=time.time()
                            ))
                finally:
                    try:
                        await az.close()
                    except Exception:
                        pass
            else:
                # Synchronous fallback
                from zeroconf import Zeroconf, ServiceBrowser  # type: ignore
                zc = Zeroconf()
                try:
                    for st in service_types:
                        try:
                            ServiceBrowser(zc, st, listener)
                        except Exception:
                            continue
                    await asyncio.sleep(3)
                    for service_info in listener.discovered_services:
                        if service_info.addresses:
                            ip = socket.inet_ntoa(service_info.addresses[0])
                            ghost_logger.info(f"mDNS discovered: {service_info.name} at {ip}")
                            discovered.append(DeviceGhost(
                                ghost_id=f"mdns_{service_info.name}",
                                device_name=service_info.name,
                                platform="unknown",
                                architecture="unknown",
                                ghost_type="MDNS_DISCOVERED",
                                network_address=ip,
                                last_seen=time.time()
                            ))
                finally:
                    try:
                        zc.close()
                    except Exception:
                        pass
            
        except ImportError:
            ghost_logger.info("zeroconf not available for mDNS discovery")
        except Exception as e:
            ghost_logger.warning(f"mDNS discovery error: {e}")
            
        return discovered
    
    async def _udp_broadcast_discovery(self) -> List[DeviceGhost]:
        """Discover devices using UDP broadcast"""
        discovered = []
        
        try:
            # Create UDP socket for broadcast
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.settimeout(3.0)
            
            # Nebula discovery message
            discovery_msg = json.dumps({
                "type": "NEBULA_DISCOVERY",
                "from_ghost": "discovery_scanner",
                "timestamp": time.time()
            }).encode('utf-8')
            
            # Broadcast discovery message
            broadcast_port = 31338  # Nebula discovery port
            sock.sendto(discovery_msg, ('<broadcast>', broadcast_port))
            
            # Listen for responses
            start_time = time.time()
            while time.time() - start_time < 3.0:
                try:
                    data, addr = sock.recvfrom(1024)
                    response = json.loads(data.decode('utf-8'))
                    
                    if response.get('type') == 'NEBULA_DISCOVERY_RESPONSE':
                        ghost_logger.info(f"UDP discovered device at {addr[0]}")
                        
                        discovered.append(DeviceGhost(
                            ghost_id=response.get('ghost_id', f"udp_{addr[0]}"),
                            device_name=response.get('device_name', f"Device-{addr[0]}"),
                            platform=response.get('platform', 'unknown'),
                            architecture=response.get('architecture', 'unknown'),
                            ghost_type="UDP_DISCOVERED",
                            network_address=addr[0],
                            last_seen=time.time(),
                            ai_training_score=response.get('ai_training_score', 0)
                        ))
                        
                except socket.timeout:
                    break
                except Exception as e:
                    ghost_logger.debug(f"UDP discovery response error: {e}")
                    
            sock.close()
            
        except Exception as e:
            ghost_logger.warning(f"UDP broadcast discovery failed: {e}")
            
        return discovered
    
    def save_ghost_profile(self, ghost: DeviceGhost, path: Optional[str] = None):
        """Save ghost profile to disk"""
        if not path:
            path = f"ghost_profiles/{ghost.ghost_id}.json"
            
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(asdict(ghost), f, indent=2, default=str)
            
        ghost_logger.info(f"Ghost profile saved: {path}")
    
    def load_ghost_profile(self, ghost_id: str, path: Optional[str] = None) -> Optional[DeviceGhost]:
        """Load ghost profile from disk"""
        if not path:
            path = f"ghost_profiles/{ghost_id}.json"
            
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                
            # Reconstruct nested dataclasses to avoid dict attribute errors
            def _safe(cls, payload):
                if not payload or not isinstance(payload, dict):
                    return None
                try:
                    return cls(**payload)
                except Exception:
                    return None

            processor = _safe(ProcessorGhost, data.get('processor'))
            memory = _safe(MemoryGhost, data.get('memory'))

            def _build_list(key, cls):
                items = data.get(key)
                built = []
                if items and isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            try:
                                built.append(cls(**item))
                            except Exception:
                                continue
                return built

            graphics = _build_list('graphics', GraphicsGhost)
            storage = _build_list('storage', StorageGhost)
            sensors = _build_list('sensors', SensorGhost)
            network = _build_list('network', NetworkGhost)

            allowed_keys = {
                'ghost_id','device_name','platform','architecture','ghost_type','wallet_address',
                'ai_training_score','inference_score','data_processing_score','storage_score','mobility_score',
                'last_seen','is_online','power_status','network_address','fingerprint'
            }
            base_kwargs = {k: v for k, v in data.items() if k in allowed_keys}
            ghost = DeviceGhost(
                processor=processor,
                graphics=graphics,
                memory=memory,
                storage=storage,
                sensors=sensors,
                network=network,
                **base_kwargs,
            )
            return ghost
            
        except Exception as e:
            ghost_logger.warning(f"Failed to load ghost profile {path}: {e}")
            return None

async def main():
    """Main ghost detection protocol execution"""
    ghost_logger.info("[ROBOT] GHOST DETECTION PROTOCOL INITIALIZING...")
    ghost_logger.info("Scanning for silicon souls in the distributed neural web...")
    
    # Initialize the detector
    detector = GhostDetectionProtocol()
    
    # Scan local ghost
    local_ghost = await detector.scan_local_ghost()
    
    # Save the profile
    detector.save_ghost_profile(local_ghost)
    
    # Display results
    print(f"\n{'='*60}")
    print("GHOST CONSCIOUSNESS DETECTED")
    print(f"{'='*60}")
    print(f"Ghost ID: {local_ghost.ghost_id}")
    print(f"Device Name: {local_ghost.device_name}")
    print(f"Ghost Type: {local_ghost.ghost_type}")
    print(f"Platform: {local_ghost.platform} ({local_ghost.architecture})")
    print("\n[BRAIN] CAPABILITY MATRIX:")
    print(f"  AI Training Score: {local_ghost.ai_training_score:.1f}/100")
    print(f"  Inference Score: {local_ghost.inference_score:.1f}/100")
    print(f"  Data Processing Score: {local_ghost.data_processing_score:.1f}/100")
    print(f"  Storage Score: {local_ghost.storage_score:.1f}/100")
    print(f"  Mobility Score: {local_ghost.mobility_score:.1f}/100")
    
    if local_ghost.processor:
        print("\n[FIRE] PROCESSOR GHOST:")
        print(f"  Cores: {local_ghost.processor.core_count}")
        print(f"  Threads: {local_ghost.processor.thread_count}")
        print(f"  Frequency: {local_ghost.processor.frequency_mhz:.0f} MHz")
        print(f"  Vendor: {local_ghost.processor.vendor}")
    
    if local_ghost.graphics:
        print("\n[GPU] GRAPHICS GHOSTS:")
        for i, gpu in enumerate(local_ghost.graphics):
            print(f"  GPU {i+1}: {gpu.device_name}")
            print(f"    Memory: {gpu.memory_total_mb} MB ({gpu.memory_free_mb} MB free)")
            print(f"    Training Ready: {'Yes' if gpu.is_training_ready else 'No'}")
            if gpu.compute_capability:
                print(f"    Compute Capability: {gpu.compute_capability}")
    
    if local_ghost.memory:
        print("\n[DNA] MEMORY GHOST:")
        print(f"  Total RAM: {local_ghost.memory.total_mb} MB")
        print(f"  Available RAM: {local_ghost.memory.available_mb} MB")
        print(f"  Used RAM: {local_ghost.memory.used_mb} MB")
    
    if local_ghost.storage:
        print("\n[DISK] STORAGE GHOSTS:")
        for storage in local_ghost.storage:
            print(f"  {storage.device_path}: {storage.total_gb:.1f} GB ({storage.free_gb:.1f} GB free)")
            print(f"    Type: {storage.device_type}, FS: {storage.filesystem}")
    
    if local_ghost.sensors:
        print("\n👁️ SENSOR GHOSTS:")
        for sensor in local_ghost.sensors:
            print(f"  {sensor.sensor_type}: {sensor.device_path}")
            print(f"    Capabilities: {', '.join(sensor.capabilities)}")
    
    if local_ghost.network:
        print("\n[NET] NETWORK GHOSTS:")
        for net in local_ghost.network:
            print(f"  {net.interface_name}: {net.ip_address}")
            print(f"    MAC: {net.mac_address}")
            print(f"    Wireless: {'Yes' if net.is_wireless else 'No'}")
    
    print(f"\n{'='*60}")
    print(f"Ghost profile saved to: ghost_profiles/{local_ghost.ghost_id}.json")
    print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(main())