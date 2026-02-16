"""
EIP-191 Signing Utility for Cronos Wallet Authentication

Provides EVM-standard message signing for VoxBridge challenge-response auth.
Works with any EVM-compatible chain (Cronos, Ethereum, Polygon, BSC, etc.).
"""

import logging
import os
from typing import Optional

try:
    from eth_account import Account
    from eth_account.messages import encode_defunct
    HAS_ETH_ACCOUNT = True
except ImportError:
    HAS_ETH_ACCOUNT = False
    Account = None
    encode_defunct = None

logger = logging.getLogger(__name__)


class EVMSigner:
    """
    EVM message signer for Cronos wallet authentication.
    
    Uses EIP-191 personal_sign standard:
    - Works with Cronos (EVM-compatible)
    - Same format as Ethereum, Polygon, BSC
    - Off-chain signature verification
    - Zero blockchain interaction
    """
    
    def __init__(self, private_key: Optional[str] = None):
        """
        Initialize signer with Cronos wallet private key.
        
        Args:
            private_key: Hex-encoded private key (0x...). If not provided,
                        reads from CRO_WALLET_PRIVATE_KEY environment variable.
        
        Raises:
            ImportError: If eth-account package not installed
            ValueError: If no private key provided
        """
        if not HAS_ETH_ACCOUNT:
            raise ImportError(
                "eth-account package required for EIP-191 signing. "
                "Install with: pip install eth-account"
            )
        
        self.private_key = private_key or os.getenv("CRO_WALLET_PRIVATE_KEY")
        
        if not self.private_key:
            raise ValueError(
                "No private key provided. Either pass private_key argument "
                "or set CRO_WALLET_PRIVATE_KEY environment variable."
            )
        
        # Create account from private key
        self.account = Account.from_key(self.private_key)
        self.address = self.account.address
        
        logger.info(f"EVM signer initialized for address: {self.address}")
    
    def sign_message(self, message: str) -> str:
        """
        Sign message using EIP-191 format.
        
        Args:
            message: Challenge string from VoxBridge
        
        Returns:
            Hex-encoded signature (0x...)
        """
        # Encode message using EIP-191 format
        encoded_message = encode_defunct(text=message)
        
        # Sign the message
        signed = self.account.sign_message(encoded_message)
        
        # Return signature as hex string
        signature = signed.signature.hex()
        
        logger.debug(f"Signed message: {message[:50]}... -> {signature[:20]}...")
        
        return signature
    
    def get_address(self) -> str:
        """Get Cronos wallet address."""
        return self.address
    
    @staticmethod
    def verify_signature(
        message: str,
        signature: str,
        expected_address: str
    ) -> bool:
        """
        Verify EIP-191 signature matches expected address.
        
        This is typically done by VoxBridge backend, but provided here
        for testing/debugging purposes.
        
        Args:
            message: Original challenge message
            signature: Hex-encoded signature
            expected_address: Expected signer address
        
        Returns:
            True if signature valid and matches address
        """
        if not HAS_ETH_ACCOUNT:
            raise ImportError("eth-account package required for signature verification")
        
        try:
            # Encode message
            encoded_message = encode_defunct(text=message)
            
            # Recover address from signature
            recovered_address = Account.recover_message(encoded_message, signature=signature)
            
            # Compare addresses (case-insensitive)
            return recovered_address.lower() == expected_address.lower()
            
        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False


def create_signer(private_key: Optional[str] = None) -> EVMSigner:
    """
    Convenience function to create EVM signer.
    
    Args:
        private_key: Hex-encoded private key (0x...). If not provided,
                    reads from CRO_WALLET_PRIVATE_KEY environment variable.
    
    Returns:
        Configured EVMSigner instance
    """
    return EVMSigner(private_key=private_key)


def sign_challenge_for_adapter(adapter, private_key: Optional[str] = None) -> dict:
    """
    Complete challenge-response flow for adapter.
    
    Args:
        adapter: OpenClawdAdapter instance (must be bootstrapped)
        private_key: Optional private key (reads from env if not provided)
    
    Returns:
        Verification response from VoxBridge
    
    Example:
        >>> adapter = OpenClawdAgentFactory.create("my-agent", "llm")
        >>> adapter.bootstrap()
        >>> result = sign_challenge_for_adapter(adapter)
        >>> print(f"Verified: {result}")
    """
    # Create signer
    signer = EVMSigner(private_key=private_key)
    
    # Define signing function
    def sign_fn(challenge: str) -> str:
        return signer.sign_message(challenge)
    
    # Execute challenge-response
    result = adapter.sign_and_verify(sign_fn)
    
    logger.info(f"Challenge-response completed for {adapter.client.agent_name}")
    
    return result


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example: create signer and sign a message
    try:
        signer = create_signer()
        
        test_message = "This is a test challenge from VoxBridge"
        signature = signer.sign_message(test_message)
        
        print(f"Address: {signer.get_address()}")
        print(f"Signature: {signature}")
        
        # Verify signature
        is_valid = EVMSigner.verify_signature(
            message=test_message,
            signature=signature,
            expected_address=signer.get_address()
        )
        
        print(f"Signature valid: {is_valid}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTo use this utility:")
        print("1. Install eth-account: pip install eth-account")
        print("2. Set environment variable: export CRO_WALLET_PRIVATE_KEY=0x...")
        print("3. Or pass private_key argument directly")
