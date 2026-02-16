"""
EIP-191 Sigil Authentication

Challenge-response pattern using Ethereum signing.
Implements personal_sign (EIP-191) for agent identity verification.

Usage:
  signer = EIP191Signer.from_private_key("0x...")
  challenge = client.challenge()
  proof = signer.sign_message(challenge["challenge"])
  verified = client.verify(sigil_proof=proof, challenge=challenge["challenge"])

Dependencies:
  pip install eth-account  # For signing
  
If eth-account not available, falls back to mock signer (dev/testing only).
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================================
# EIP-191 SIGNER
# ============================================================================


class EIP191Signer:
    """Sign messages using EIP-191 personal_sign."""

    def __init__(self, account=None):
        self.account = account
        self.address = account.address if account else None

    @staticmethod
    def from_private_key(private_key_hex: str) -> "EIP191Signer":
        """Create signer from private key.

        Args:
            private_key_hex: Hex string (with or without 0x prefix)

        Returns:
            EIP191Signer instance
        """
        try:
            from eth_account import Account
            from eth_account.messages import encode_defunct

            # Ensure 0x prefix
            if not private_key_hex.startswith("0x"):
                private_key_hex = "0x" + private_key_hex

            account = Account.from_key(private_key_hex)
            logger.info(f"✅ Loaded signer: {account.address}")
            return EIP191Signer(account)

        except ImportError:
            logger.warning(
                "eth-account not installed. Falling back to mock signer (dev only)."
            )
            return EIP191MockSigner(private_key_hex)

    def sign_message(self, message: str) -> str:
        """Sign message using EIP-191.

        Args:
            message: Message text

        Returns:
            Signature (hex string)
        """
        from eth_account.messages import encode_defunct

        msg = encode_defunct(text=message)
        signed = self.account.sign_message(msg)
        return signed.signature.hex()

    def get_address(self) -> Optional[str]:
        """Get signer address."""
        return self.address


class EIP191MockSigner:
    """Mock signer for development/testing when eth-account unavailable."""

    def __init__(self, private_key_hex: str):
        self.private_key = private_key_hex
        self.address = "0xMOCK_ADDRESS"
        logger.warning(
            "Mock signer active (eth-account not available). "
            "This returns fake signatures for testing only."
        )

    def sign_message(self, message: str) -> str:
        """Return mock signature."""
        import hashlib

        # Deterministic mock based on private key + message
        digest = hashlib.sha256(f"{self.private_key}:{message}".encode()).hexdigest()
        return f"0x{digest}test_signature_only"

    def get_address(self) -> str:
        """Get mock address."""
        return self.address


# ============================================================================
# CHALLENGE-RESPONSE FLOW
# ============================================================================


def do_challenge_response(client, signer: EIP191Signer) -> dict:
    """Execute full challenge-response authentication.

    Args:
        client: VoxBridgeClient instance
        signer: EIP191Signer for message signing

    Returns:
        Verification result from server
    """
    try:
        # 1) Request challenge
        logger.info("📝 Requesting authentication challenge...")
        challenge_resp = client.challenge()
        challenge_text = challenge_resp.get("challenge")

        if not challenge_text:
            raise ValueError(
                f"Invalid challenge response: {challenge_resp}"
            )

        logger.info(f"Got challenge: {challenge_text[:20]}...")

        # 2) Sign challenge
        logger.info("🔐 Signing challenge...")
        signature = signer.sign_message(challenge_text)
        logger.info(f"Signed: {signature[:40]}...")

        # 3) Verify with server
        logger.info("✓ Submitting verification...")
        verified = client.verify(sigil_proof=signature, challenge=challenge_text)

        logger.info(f"✅ Authentication verified: {verified}")
        return verified

    except Exception as e:
        logger.error(f"❌ Challenge-response failed: {e}", exc_info=True)
        raise


# ============================================================================
# SECURE INTENT SUBMISSION
# ============================================================================


def submit_intent_with_proof(
    client,
    intent_name: str,
    intent_params: dict,
    signer: EIP191Signer,
    expires_in_hours: int = 24,
) -> dict:
    """Submit intent with EIP-191 proof.

    Executes challenge-response, then submits intent with signature.

    Args:
        client: VoxBridgeClient
        intent_name: Name of intent (e.g., "execute_trade")
        intent_params: Intent parameters
        signer: EIP191Signer
        expires_in_hours: Intent expiration

    Returns:
        Intent submission result
    """
    try:
        # Verify identity first
        verify_result = do_challenge_response(client, signer)

        # Generate proof for intent
        import json

        intent_blob = json.dumps(intent_params, sort_keys=True)
        intent_proof = signer.sign_message(intent_blob)

        logger.info(f"📮 Submitting intent: {intent_name}")
        result = client.submit_intent(
            intent_name=intent_name,
            intent_params=intent_params,
            sigil_proof=intent_proof,
            expires_in_hours=expires_in_hours,
        )

        logger.info(f"✅ Intent submitted: {result}")
        return result

    except Exception as e:
        logger.error(f"❌ Intent submission failed: {e}", exc_info=True)
        raise
