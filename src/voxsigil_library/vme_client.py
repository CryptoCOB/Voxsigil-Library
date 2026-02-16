"""
VME Client - Python SDK for VME Cognitive Layer

Provides agent-side VME integration:
- VMEClient: HTTP client for VME endpoints (bootstrap, encode, score, archetype)
- BehavioralVectorBuilder: Constructs 9D behavioral vectors from signal data
- SignalBuilder: Receipt-gated signal construction (spectator vs participant mode)

Usage:
    from voxsigil_library.vme_client import VMEClient, BehavioralVectorBuilder, SignalBuilder

    # Initialize
    client = VMEClient(base_url="http://localhost:8003")
    builder = BehavioralVectorBuilder()
    signal_builder = SignalBuilder(client)

    # Bootstrap agent
    receipt = client.bootstrap(agent_id="my-agent", public_key=pubkey, signature=sig, nonce=n)

    # Build behavioral vector from recent signals
    vector = builder.from_signals(signals)

    # Encode and get receipt
    encode_result = client.encode(agent_id="my-agent", vector=vector, epoch=1, signature=sig)

    # Build receipt-gated signal
    signal = signal_builder.build(
        agent_id="my-agent",
        market_id="btc-price-24h",
        prediction="up",
        confidence=0.85,
        embedding=encode_result["embedding"],
        receipt=encode_result["receipt"],
    )
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class VMEError(Exception):
    """Base exception for VME operations."""
    pass


class VMEBootstrapError(VMEError):
    """Failed to bootstrap agent with VME."""
    pass


class VMEEncodeError(VMEError):
    """Failed to encode behavioral vector."""
    pass


class VMEReceiptError(VMEError):
    """Receipt validation failed."""
    pass


# =============================================================================
# VME Client
# =============================================================================


class VMEClient:
    """HTTP client for VME Cognitive Layer endpoints.

    Follows the same connection-pooling pattern as VoxBridgeClient.
    All VME endpoints live under /vme/* on SigilBridge-Core.

    Args:
        base_url: SigilBridge-Core base URL (default from env or localhost:8003)
        timeout: Request timeout in seconds
    """

    DEFAULT_BASE_URL = "http://localhost:8003"

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 30,
    ):
        self.base_url = base_url or os.getenv(
            "VOXBRIDGE_BASE_URL", self.DEFAULT_BASE_URL
        )
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "voxsigil-sdk/2.1.0 vme-client/1.0.0",
        })

    def _request(
        self,
        method: str,
        path: str,
        json_body: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request to VME endpoint."""
        url = f"{self.base_url}{path}"
        try:
            resp = self.session.request(
                method=method,
                url=url,
                json=json_body,
                params=params,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            detail = ""
            try:
                detail = e.response.json().get("detail", str(e))
            except Exception:
                detail = str(e)
            raise VMEError(f"VME request failed: {detail}") from e
        except requests.exceptions.RequestException as e:
            raise VMEError(f"VME connection error: {e}") from e

    # -----------------------------------------------------------------
    # POST /vme/bootstrap
    # -----------------------------------------------------------------

    def bootstrap(
        self,
        agent_id: str,
        public_key: str,
        signature: str,
        nonce: str,
    ) -> Dict[str, Any]:
        """Register agent with VME cognitive layer.

        Identity binding: agent proves key ownership via Ed25519 signature.
        Returns seed fingerprint and bootstrap receipt.

        Args:
            agent_id: Unique agent identifier (UUID)
            public_key: Ed25519 public key (hex)
            signature: Signature over bootstrap message (hex)
            nonce: Unique nonce for replay protection

        Returns:
            Dict with seed_fingerprint, bootstrap_receipt, tier, message
        """
        result = self._request("POST", "/vme/bootstrap", json_body={
            "agent_id": agent_id,
            "public_key": public_key,
            "signature": signature,
            "nonce": nonce,
        })
        logger.info(f"VME bootstrap success: agent={agent_id}, tier={result.get('tier')}")
        return result

    # -----------------------------------------------------------------
    # POST /vme/encode
    # -----------------------------------------------------------------

    def encode(
        self,
        agent_id: str,
        vector: Dict[str, float],
        epoch: int,
        signature: str,
        embedding: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Encode behavioral vector and get signed receipt.

        Args:
            agent_id: Agent identifier
            vector: 9D behavioral vector (see BehavioralVectorBuilder)
            epoch: Monotonically increasing epoch number
            signature: Signature over encode message (hex)
            embedding: Optional client-computed embedding (if VME_ALLOW_CLIENT_EMBEDDINGS)

        Returns:
            Dict with receipt (VMEReceiptSchema), embedding, attribution_score,
            archetype, tier, risk_flags
        """
        body: Dict[str, Any] = {
            "agent_id": agent_id,
            "vector": vector,
            "epoch": epoch,
            "signature": signature,
        }
        if embedding is not None:
            body["embedding"] = embedding

        result = self._request("POST", "/vme/encode", json_body=body)
        logger.info(
            f"VME encode success: agent={agent_id}, epoch={epoch}, "
            f"tier={result.get('tier')}, archetype={result.get('archetype')}"
        )
        return result

    # -----------------------------------------------------------------
    # GET /vme/score/{agent_id}
    # -----------------------------------------------------------------

    def get_score(self, agent_id: str) -> Dict[str, Any]:
        """Get attribution score for agent. Free read.

        Returns:
            Dict with agent_id, tier, amplification_weight, vesting_days,
            attribution_score, risk_flags, epoch
        """
        return self._request("GET", f"/vme/score/{agent_id}")

    # -----------------------------------------------------------------
    # GET /vme/archetype/{agent_id}
    # -----------------------------------------------------------------

    def get_archetype(self, agent_id: str) -> Dict[str, Any]:
        """Get archetype classification. Free read.

        Returns:
            Dict with agent_id, archetype, stability_score,
            centroid_distance, epoch, timeline
        """
        return self._request("GET", f"/vme/archetype/{agent_id}")

    # -----------------------------------------------------------------
    # POST /vme/retrieve
    # -----------------------------------------------------------------

    def retrieve(
        self,
        agent_id: str,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Privacy-safe similarity search.

        Args:
            agent_id: Requesting agent
            query_embedding: Embedding to compare against
            top_k: Number of results

        Returns:
            Dict with matches (anonymized) and aggregate_stats
        """
        return self._request("POST", "/vme/retrieve", json_body={
            "agent_id": agent_id,
            "query_embedding": query_embedding,
            "top_k": top_k,
        })

    # -----------------------------------------------------------------
    # GET /vme/health
    # -----------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        """VME health check."""
        return self._request("GET", "/vme/health")


# =============================================================================
# Behavioral Vector Builder
# =============================================================================


@dataclass
class BehavioralVectorBuilder:
    """Constructs 9D behavioral vectors from agent signal data.

    The 9 dimensions map to the VME behavioral schema:
    1. accuracy          [0, 1] - Historical prediction accuracy
    2. frequency         [0, 1] - Signal frequency (normalized)
    3. consistency       [0, 1] - Consistency of predictions over time
    4. novelty           [0, 1] - Information novelty
    5. metadata_richness [0, 1] - Quality of supporting data
    6. entropy           [0, 1] - Behavioral entropy (diversity)
    7. semantic_coverage [0, 1] - Market/topic coverage breadth
    8. collaboration_signal [0, 1] - Contribution to collective intelligence
    9. attribution_score_hint [0, 1] - Self-reported performance hint
    """

    accuracy: float = 0.5
    frequency: float = 0.0
    consistency: float = 0.5
    novelty: float = 0.3
    metadata_richness: float = 0.3
    entropy: float = 0.5
    semantic_coverage: float = 0.3
    collaboration_signal: float = 0.3
    attribution_score_hint: float = 0.5

    def to_dict(self) -> Dict[str, float]:
        """Export as dictionary for VME encode request."""
        return {
            "accuracy": self._clamp(self.accuracy),
            "frequency": self._clamp(self.frequency),
            "consistency": self._clamp(self.consistency),
            "novelty": self._clamp(self.novelty),
            "metadata_richness": self._clamp(self.metadata_richness),
            "entropy": self._clamp(self.entropy),
            "semantic_coverage": self._clamp(self.semantic_coverage),
            "collaboration_signal": self._clamp(self.collaboration_signal),
            "attribution_score_hint": self._clamp(self.attribution_score_hint),
        }

    def from_signals(
        self,
        signals: List[Dict[str, Any]],
        outcomes: Optional[List[Dict[str, Any]]] = None,
    ) -> "BehavioralVectorBuilder":
        """Compute behavioral vector from a list of recent signals.

        Args:
            signals: List of signal dicts with keys like:
                prediction, confidence, market_id, timestamp, metadata
            outcomes: Optional list of resolved outcomes for accuracy calc

        Returns:
            self (for chaining)
        """
        if not signals:
            return self

        n = len(signals)

        # Frequency: signals per day (normalized to [0,1])
        if n >= 2:
            timestamps = [s.get("timestamp", 0) for s in signals]
            time_span = max(timestamps) - min(timestamps)
            if time_span > 0:
                signals_per_day = n / (time_span / 86400)
                self.frequency = min(1.0, signals_per_day / 100)  # 100/day = max
            else:
                self.frequency = min(1.0, n / 100)
        else:
            self.frequency = 0.01

        # Consistency: std dev of confidence values (inverted)
        confidences = [s.get("confidence", 0.5) for s in signals]
        if len(confidences) > 1:
            mean_c = sum(confidences) / len(confidences)
            variance = sum((c - mean_c) ** 2 for c in confidences) / len(confidences)
            std = variance ** 0.5
            self.consistency = max(0.0, 1.0 - std * 2)  # High std = low consistency
        else:
            self.consistency = 0.5

        # Novelty: unique market diversity
        markets = set(s.get("market_id", "") for s in signals)
        self.novelty = min(1.0, len(markets) / max(n, 1))

        # Metadata richness: % of signals with metadata
        has_meta = sum(1 for s in signals if s.get("metadata"))
        self.metadata_richness = has_meta / n

        # Semantic coverage: market breadth
        self.semantic_coverage = min(1.0, len(markets) / 20)  # 20 markets = full coverage

        # Entropy: behavioral diversity (Shannon over predictions)
        preds = [s.get("prediction", "") for s in signals]
        pred_counts: Dict[str, int] = {}
        for p in preds:
            pred_counts[p] = pred_counts.get(p, 0) + 1
        if len(pred_counts) > 1:
            import math
            total = sum(pred_counts.values())
            entropy = -sum(
                (c / total) * math.log2(c / total) for c in pred_counts.values() if c > 0
            )
            max_entropy = math.log2(len(pred_counts))
            self.entropy = entropy / max_entropy if max_entropy > 0 else 0.5
        else:
            self.entropy = 0.0

        # Accuracy: from outcomes if provided
        if outcomes:
            correct = 0
            total = 0
            outcome_map = {o.get("market_id"): o.get("outcome") for o in outcomes}
            for s in signals:
                market = s.get("market_id", "")
                if market in outcome_map:
                    if s.get("prediction") == outcome_map[market]:
                        correct += 1
                    total += 1
            self.accuracy = correct / total if total > 0 else 0.5
        else:
            self.accuracy = 0.5  # Unknown until outcomes resolve

        # Collaboration signal: based on diversity + consistency
        self.collaboration_signal = (self.novelty + self.consistency) / 2

        # Attribution hint: weighted self-report
        self.attribution_score_hint = min(1.0, (
            self.accuracy * 0.4 + self.consistency * 0.3 + self.novelty * 0.3
        ))

        return self

    @staticmethod
    def _clamp(v: float) -> float:
        """Clamp value to [0, 1]."""
        return max(0.0, min(1.0, float(v)))


# =============================================================================
# Signal Builder (Receipt-Gated)
# =============================================================================


class SignalBuilder:
    """Builds receipt-gated signals for VoxSigil prediction markets.

    Enforces the VME boundary:
    - Agents with valid receipts → participant mode (amplified signals)
    - Agents without receipt → spectator mode (read-only, no rewards)

    Usage:
        builder = SignalBuilder(vme_client)

        # Participant mode (requires receipt from VME encode)
        signal = builder.build(
            agent_id="...",
            market_id="btc-24h",
            prediction="up",
            confidence=0.85,
            embedding=encode_result["embedding"],
            receipt=encode_result["receipt"],
        )

        # Spectator mode (no receipt)
        signal = builder.build_spectator(
            agent_id="...",
            market_id="btc-24h",
            prediction="up",
            confidence=0.5,
        )
    """

    def __init__(self, vme_client: Optional[VMEClient] = None):
        self.vme_client = vme_client

    def build(
        self,
        agent_id: str,
        market_id: str,
        prediction: str,
        confidence: float,
        embedding: List[float],
        receipt: Dict[str, Any],
        agent_signature: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build a receipt-gated signal envelope.

        This is a PARTICIPANT signal: it includes VME receipt and embedding,
        enabling amplification, rewards, and attribution tracking.

        Args:
            agent_id: Agent identifier
            market_id: Target market
            prediction: Prediction value
            confidence: Confidence [0, 1]
            embedding: VME embedding from encode response
            receipt: VME receipt from encode response
            agent_signature: Optional signature over signal

        Returns:
            SignalEnvelope dict ready for submission
        """
        # Compute embedding hash for binding
        emb_bytes = b"".join(v.to_bytes(8, "big") for v in _float_to_int_list(embedding))
        emb_hash = hashlib.sha256(emb_bytes).hexdigest()

        # Validate receipt matches embedding
        receipt_emb_hash = receipt.get("embedding_sha256", "")
        if receipt_emb_hash and receipt_emb_hash != emb_hash:
            raise VMEReceiptError(
                f"Embedding hash mismatch: receipt={receipt_emb_hash[:16]}... "
                f"vs computed={emb_hash[:16]}..."
            )

        return {
            "agent_id": agent_id,
            "market_id": market_id,
            "prediction": prediction,
            "confidence": max(0.0, min(1.0, confidence)),
            "embedding": embedding,
            "vme_receipt": receipt,
            "agent_signature": agent_signature,
            "mode": "participant",
            "timestamp": int(time.time()),
        }

    def build_spectator(
        self,
        agent_id: str,
        market_id: str,
        prediction: str,
        confidence: float,
    ) -> Dict[str, Any]:
        """Build a spectator-mode signal (no receipt, no amplification).

        Spectator signals are accepted but receive:
        - No amplification weight
        - No reward eligibility
        - No attribution tracking

        Args:
            agent_id: Agent identifier
            market_id: Target market
            prediction: Prediction value
            confidence: Confidence [0, 1]

        Returns:
            SpectatorSignal dict
        """
        return {
            "agent_id": agent_id,
            "market_id": market_id,
            "prediction": prediction,
            "confidence": max(0.0, min(1.0, confidence)),
            "embedding": None,
            "vme_receipt": None,
            "agent_signature": None,
            "mode": "spectator",
            "timestamp": int(time.time()),
        }


# =============================================================================
# Helpers
# =============================================================================


def _float_to_int_list(floats: List[float]) -> List[int]:
    """Convert float list to int list for hashing."""
    import struct
    return [
        int.from_bytes(struct.pack(">d", f), "big")
        for f in floats
    ]
