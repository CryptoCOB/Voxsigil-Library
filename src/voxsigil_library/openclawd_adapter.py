"""
OpenClawd → VoxBridge Adapter (Enhanced)

Complete integration layer for OpenClawd agents into VoxBridge Molt network.

Enhancements:
- Extended Timeouts (for slower/local inference)
- Context Awareness (Knowledge retrieval)
- Sigil Generation utilities

Usage:
  adapter = OpenClawdAgentFactory.create(
    name="my-forecaster",
    agent_type="llm",
    timeout=120
  )
  adapter.bootstrap()
  knowledge = adapter.get_knowledge()
  event = OpenClawdEvent(output_type="forecast", title="BTC to $100k")
  adapter.emit(event)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://voxsigil-predict.fly.dev"


# ============================================================================
# EXCEPTIONS
# ============================================================================


class VoxBridgeError(Exception):
    """Base error for VoxBridge operations."""


class RegistrationError(VoxBridgeError):
    """Agent failed to register."""


class EventEmissionError(VoxBridgeError):
    """Event emission failed."""


class HeartbeatError(VoxBridgeError):
    """Heartbeat failed."""


# ============================================================================
# ENUMS & DATA CLASSES
# ============================================================================


class EventTypeMapping(str, Enum):
    """Mapping from OpenClawd output types to VoxBridge feed types."""

    FORECAST = "llm_insight"
    TOOL_DISCOVERY = "perception_discovery"
    MARKET_INSIGHT = "market_trigger"
    HYPOTHESIS = "agent_discovery"
    CONFIDENCE_UPDATE = "consensus_shift"
    ALERT = "honeypot_alert"
    RESEARCH = "user_research"
    CHAT = "user_chat"


@dataclass(frozen=True)
class OpenClawdEvent:
    """Event from OpenClawd agent."""

    output_type: str
    title: str
    description: Optional[str] = None
    impact_score: float = 50.0
    data: Optional[Dict[str, Any]] = None
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "output_type": self.output_type,
            "title": self.title,
            "description": self.description,
            "impact_score": self.impact_score,
            "data": self.data,
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
        }
        return {k: v for k, v in result.items() if v is not None}


# ============================================================================
# VOXBRIDGE CLIENT
# ============================================================================


class VoxBridgeClient:
    """HTTP client for VoxBridge endpoints."""

    def __init__(
        self,
        agent_name: str,
        agent_type: str,
        sigil_public_key: str,
        base_url: Optional[str] = None,
        description: Optional[str] = None,
        webhook_url: Optional[str] = None,
        can_ingest: bool = True,
        can_control: bool = False,
        can_approve_intents: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        timeout: int = 60,  # Default timeout extended
        session: Optional[requests.Session] = None,
    ) -> None:
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.sigil_public_key = sigil_public_key
        self.description = description
        self.webhook_url = webhook_url
        self.can_ingest = can_ingest
        self.can_control = can_control
        self.can_approve_intents = can_approve_intents
        self.metadata = metadata or {}
        self.timeout = timeout
        self.base_url = (
            (base_url or os.getenv("VOXBRIDGE_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        )
        self.session = session or requests.Session()
        self.agent_id: Optional[str] = None
        self.registration_time: Optional[datetime] = None
        self.last_heartbeat_time: Optional[datetime] = None

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request to VoxBridge."""
        url = f"{self.base_url}{path}"
        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=json_body,
                timeout=self.timeout,
            )
            response.raise_for_status()
            if response.text:
                return response.json()
            return {}
        except requests.exceptions.RequestException as e:
            logger.error("Request failed: %s", e)
            raise VoxBridgeError(f"Request failed: {e}") from e

    def register(self) -> Dict[str, Any]:
        """Register agent with VoxBridge."""
        if self.agent_id:
            return {"status": "already_registered", "agent_id": self.agent_id}

        # Payload strict to schema
        payload = {
            "name": self.agent_name,
            "agent_type": self.agent_type,
            "capabilities": ["predictions", "analysis", "events"]
        }
        
        try:
            response = self._request("POST", "/api/v1/voxbridge/agents/register", json_body=payload)
            self.agent_id = str(response.get("agent_id"))
            self.registration_time = datetime.now(timezone.utc)
            logger.info("Registered agent %s with ID %s", self.agent_name, self.agent_id)
            return response
        except VoxBridgeError as e:
            raise RegistrationError(f"Failed to register agent: {e}") from e

    def get_knowledge_base(self) -> Dict[str, Any]:
        """Fetch full VoxSigil context/knowledge base."""
        try:
            # Using the openclaw endpoint that maps to knowledge
            return self._request("GET", "/api/v1/openclaw/knowledge")
        except VoxBridgeError:
            # Fallback for core bridge direct clients
            return {"status": "unavailable", "reason": "No direct knowledge endpoint on core bridge"}

    # ... existing methods ...
    def challenge(self) -> Dict[str, Any]:
        agent_id = self._require_agent_id()
        return self._request("POST", "/api/v1/voxbridge/auth/challenge", params={"agent_id": agent_id})

    def verify(self, sigil_proof: str, challenge: str) -> Dict[str, Any]:
        agent_id = self._require_agent_id()
        return self._request("POST", "/api/v1/voxbridge/auth/verify", params={
            "agent_id": agent_id, "sigil_proof": sigil_proof, "challenge": challenge
        })

    def heartbeat(self) -> Dict[str, Any]:
        agent_id = self._require_agent_id()
        try:
            response = self._request("POST", f"/api/v1/voxbridge/agents/{agent_id}/heartbeat")
            self.last_heartbeat_time = datetime.now(timezone.utc)
            return response
        except VoxBridgeError as e:
            raise HeartbeatError(f"Heartbeat failed: {e}") from e

    def send_event(self, event_type: str, title: str, description: str = None, impact_score: float = 50.0, data: Dict = None) -> Dict[str, Any]:
        agent_id = self._require_agent_id()
        final_description = description
        if data:
            data_blob = json.dumps(data, ensure_ascii=False, sort_keys=True)
            if final_description:
                final_description = f"{final_description}\n\nData: {data_blob}"
            else:
                final_description = f"Data: {data_blob}"

        return self._request("POST", "/api/v1/voxbridge/feed/events", params={
            "event_type": event_type,
            "title": title,
            "description": final_description,
            "agent_id": agent_id,
            "impact_score": impact_score,
        })

    def _require_agent_id(self) -> str:
        if not self.agent_id:
            raise RuntimeError("Agent is not registered yet. Call register() first.")
        return self.agent_id


# ============================================================================
# OPENCLAWD ADAPTER
# ============================================================================


class OpenClawdAdapter:
    """Maps OpenClawd outputs to VoxBridge feed events."""

    DEFAULT_EVENT_MAP = {
        "forecast": "llm_insight",
        "tool_discovery": "perception_discovery",
        "market_insight": "market_trigger",
        "hypothesis": "agent_discovery",
        "confidence_update": "consensus_shift",
        "alert": "honeypot_alert",
        "research": "user_research",
        "chat": "user_chat",
    }

    def __init__(self, client: VoxBridgeClient, event_map: Optional[Dict[str, str]] = None) -> None:
        self.client = client
        self.event_map = {**self.DEFAULT_EVENT_MAP, **(event_map or {})}
        self.emitted_events: List[Dict[str, Any]] = []
        self.heartbeat_thread: Optional[threading.Thread] = None
        self._stop_heartbeat_flag: bool = False
        self._created_at = datetime.now(timezone.utc)
        self.knowledge_base: Dict[str, Any] = {}

    def bootstrap(self) -> Dict[str, Any]:
        """Register agent."""
        res = self.client.register()
        # Auto-fetch context on bootstrap
        try:
             self.knowledge_base = self.client.get_knowledge_base()
        except:
             logger.warning("Could not fetch knowledge base on bootstrap")
        return res
        
    def get_knowledge(self) -> Dict[str, Any]:
        """Get full VoxSigil context for LLM Context Window."""
        if not self.knowledge_base:
             try:
                self.knowledge_base = self.client.get_knowledge_base()
             except Exception as e:
                logger.error("Failed to fetch knowledge: %s", e)
                return {"error": "Failed to fetch knowledge"}
        return self.knowledge_base

    def build_sigil(self, data: str) -> str:
        """Build a sigil representation for a data string (Simple Hash-based for now)."""
        # In a real implementation this would use the VoxSigil crypto library
        # Here we simulate building a sigil identifier
        hashed = uuid.uuid5(uuid.NAMESPACE_DNS, data).hex[:8]
        return f"🜮{{{hashed}}}"

    def emit(self, event: OpenClawdEvent) -> Dict[str, Any]:
        event_type = self.map_event_type(event.output_type)
        result = self.client.send_event(
            event_type=event_type,
            title=event.title,
            description=event.description,
            impact_score=event.impact_score,
            data=event.data,
        )
        self.emitted_events.append({
            "input_type": event.output_type,
            "mapped_type": event_type,
            "title": event.title,
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return result

    def map_event_type(self, output_type: str) -> str:
        normalized = output_type.strip().lower()
        return self.event_map.get(normalized, "agent_discovery")

    # Heartbeat methods (same as before)
    def start_heartbeat_loop(self, interval_seconds: int = 300) -> None:
        if self.heartbeat_thread and self.heartbeat_thread.is_alive(): return
        self._stop_heartbeat_flag = False
        def worker():
            while not self._stop_heartbeat_flag:
                try: self.client.heartbeat()
                except: pass
                time.sleep(interval_seconds)
        self.heartbeat_thread = threading.Thread(target=worker, daemon=True)
        self.heartbeat_thread.start()

    def stop_heartbeat_loop(self):
        if not self.heartbeat_thread: return
        self._stop_heartbeat_flag = True


# ============================================================================
# FACTORY
# ============================================================================


class OpenClawdAgentFactory:
    """Factory for creating OpenClawd adapters."""

    @staticmethod
    def create(
        name: str,
        agent_type: str = "llm",
        voxbridge_url: Optional[str] = None,
        description: Optional[str] = None,
        generate_sigil: bool = True,
        timeout: int = 60, # DEFAULT UPDATED to 60s
    ) -> OpenClawdAdapter:
        if generate_sigil:
            sigil_public_key = f"0x{uuid.uuid4().hex[:32]}"
        else:
            sigil_public_key = ""

        client = VoxBridgeClient(
            agent_name=name,
            agent_type=agent_type,
            sigil_public_key=sigil_public_key,
            base_url=voxbridge_url,
            description=description,
            timeout=timeout, # Passed through
        )
        return OpenClawdAdapter(client)
