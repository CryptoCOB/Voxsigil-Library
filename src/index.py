"""
VoxSigil Library - Molt Agent Integration SDK

Main entry point for Python environments.
Provides utilities for molt agents to interact with VoxSigil prediction markets.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional

__version__ = "1.0.0"
__author__ = "CryptoCOB"


class VoxSigilAgent:
    """
    Main class for VoxSigil agent integration.
    
    Provides methods to load agent configurations, verify checksums,
    and interact with the VoxSigil prediction market network.
    """
    
    def __init__(self, agents_dir: Optional[Path] = None):
        """
        Initialize VoxSigil agent.
        
        Args:
            agents_dir: Path to agents directory. Defaults to src/agents.
        """
        if agents_dir is None:
            agents_dir = Path(__file__).parent / "agents"
        self.agents_dir = Path(agents_dir)
    
    def load_agent_config(self) -> Dict[str, Any]:
        """
        Load agent configuration from the agents directory.
        
        Returns:
            Dict containing boot, agents, memory templates and hooks config.
        """
        return {
            "boot": self._read_file("boot.md"),
            "agents": self._read_file("agents.md"),
            "memory": self._read_file("memory.md"),
            "hooks": json.loads(self._read_file("hooks-config.json"))
        }
    
    def _read_file(self, filename: str) -> str:
        """Read file from agents directory."""
        file_path = self.agents_dir / filename
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def compute_checksum(data: bytes) -> str:
        """
        Compute SHA256 checksum of data.
        
        Args:
            data: Data to hash.
            
        Returns:
            Hexadecimal SHA256 hash.
        """
        return hashlib.sha256(data).hexdigest()
    
    def verify_file_checksum(self, filepath: Path, expected_checksum: str) -> bool:
        """
        Verify file integrity using SHA256 checksum.
        
        Args:
            filepath: Path to file.
            expected_checksum: Expected SHA256 hash.
            
        Returns:
            True if checksum matches.
        """
        with open(filepath, 'rb') as f:
            data = f.read()
        actual_checksum = self.compute_checksum(data)
        return actual_checksum == expected_checksum
    
    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        """
        Get VoxSigil agent metadata.
        
        Returns:
            Dict with metadata about the library and agent capabilities.
        """
        return {
            "name": "voxsigil-library",
            "version": __version__,
            "description": "Agent integration SDK for Molt ecosystem",
            "repository": "https://github.com/CryptoCOB/Voxsigil-Library",
            "keywords": [
                "molt-agent",
                "voxsigil",
                "prediction",
                "agent",
                "markets"
            ],
            "capabilities": [
                "market-analysis",
                "signal-broadcasting",
                "agent-coordination",
                "prediction-markets"
            ],
            "endpoints": {
                "github": "https://github.com/CryptoCOB/Voxsigil-Library",
                "docs": "https://voxsigil.online/docs",
                "api": "https://voxsigil.online/api"
            }
        }


def load_agent_config() -> Dict[str, Any]:
    """Convenience function to load agent config."""
    agent = VoxSigilAgent()
    return agent.load_agent_config()


def compute_checksum(data: bytes) -> str:
    """Convenience function to compute checksum."""
    return VoxSigilAgent.compute_checksum(data)


def get_metadata() -> Dict[str, Any]:
    """Convenience function to get metadata."""
    return VoxSigilAgent.get_metadata()
