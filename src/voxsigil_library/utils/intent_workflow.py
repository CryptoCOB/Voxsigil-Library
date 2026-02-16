"""
Intent Approval Workflow - Backend Routes

Handles intent submission, approval, denial, and execution orchestration.
Integrates with VoxBridge for agent identity verification.
"""

from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# ============================================================================
# MODELS
# ============================================================================


class IntentStatus(str, Enum):
    """Intent lifecycle status."""
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXECUTED = "executed"
    FAILED = "failed"
    EXPIRED = "expired"


class IntentType(str, Enum):
    """Types of intents agents can submit."""
    EXECUTE_TRADE = "execute_trade"
    UPDATE_MODEL = "update_model"
    SEND_ALERT = "send_alert"
    MODIFY_THRESHOLD = "modify_threshold"
    REQUEST_DATA = "request_data"


class Intent(BaseModel):
    """Intent submission from agent."""
    intent_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str
    agent_name: str
    intent_type: str
    intent_params: Dict[str, Any]
    sigil_proof: Optional[str] = None  # EIP-191 signature
    rationale: Optional[str] = None
    impact_assessment: Optional[Dict[str, Any]] = None
    
    status: IntentStatus = IntentStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime
    
    # Approval tracking
    approved_by: List[str] = Field(default_factory=list)
    denied_by: List[str] = Field(default_factory=list)
    approval_threshold: int = 1  # How many approvals needed
    
    # Execution tracking
    executed_at: Optional[datetime] = None
    execution_result: Optional[Dict[str, Any]] = None
    execution_error: Optional[str] = None
    
    # Dependency graph
    depends_on: List[str] = Field(default_factory=list)
    blocking: bool = False  # If true, blocks dependent intents on failure


class IntentApprovalRequest(BaseModel):
    """Request to approve an intent."""
    intent_id: str
    approver_id: str  # User or agent ID
    approver_signature: Optional[str] = None  # For agent approvals
    notes: Optional[str] = None


class IntentDenialRequest(BaseModel):
    """Request to deny an intent."""
    intent_id: str
    denier_id: str
    reason: str
    denier_signature: Optional[str] = None


class IntentExecutionConfig(BaseModel):
    """Configuration for intent execution."""
    execute_immediately: bool = True
    timeout_seconds: int = 300
    retry_on_failure: bool = True
    max_retries: int = 3
    notify_on_completion: bool = True


# ============================================================================
# IN-MEMORY STORE (Replace with database in production)
# ============================================================================


class IntentStore:
    """In-memory intent storage. Replace with database."""
    
    def __init__(self):
        self.intents: Dict[str, Intent] = {}
        self.audit_log: List[Dict[str, Any]] = []
    
    def create_intent(self, intent: Intent) -> Intent:
        """Store new intent."""
        self.intents[intent.intent_id] = intent
        self._log_event("intent_created", intent.intent_id, {"agent_id": intent.agent_id})
        return intent
    
    def get_intent(self, intent_id: str) -> Optional[Intent]:
        """Retrieve intent by ID."""
        return self.intents.get(intent_id)
    
    def list_intents(
        self,
        status: Optional[IntentStatus] = None,
        agent_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Intent]:
        """List intents with filters."""
        filtered = []
        for intent in self.intents.values():
            if status and intent.status != status:
                continue
            if agent_id and intent.agent_id != agent_id:
                continue
            filtered.append(intent)
        
        # Sort by created_at descending
        filtered.sort(key=lambda x: x.created_at, reverse=True)
        return filtered[:limit]
    
    def update_intent(self, intent_id: str, **updates) -> Optional[Intent]:
        """Update intent fields."""
        intent = self.intents.get(intent_id)
        if not intent:
            return None
        
        for key, value in updates.items():
            if hasattr(intent, key):
                setattr(intent, key, value)
        
        self._log_event("intent_updated", intent_id, updates)
        return intent
    
    def _log_event(self, event_type: str, intent_id: str, data: Dict[str, Any]):
        """Log audit event."""
        self.audit_log.append({
            "event_type": event_type,
            "intent_id": intent_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data
        })


# Global store (replace with DI in production)
intent_store = IntentStore()


# ============================================================================
# BUSINESS LOGIC
# ============================================================================


def submit_intent(
    agent_id: str,
    agent_name: str,
    intent_type: str,
    intent_params: Dict[str, Any],
    sigil_proof: Optional[str] = None,
    expires_in_hours: int = 24,
    rationale: Optional[str] = None,
    approval_threshold: int = 1,
    depends_on: Optional[List[str]] = None,
) -> Intent:
    """
    Submit new intent from agent.
    
    Args:
        agent_id: Agent's VoxBridge ID
        agent_name: Agent display name
        intent_type: Type of intent (must be valid IntentType)
        intent_params: Parameters for intent execution
        sigil_proof: Optional EIP-191 signature from Cronos wallet
        expires_in_hours: Intent expiration (default 24h)
        rationale: Why agent is submitting this intent
        approval_threshold: Number of approvals needed
        depends_on: List of intent IDs this depends on
    
    Returns:
        Created intent
    
    Raises:
        ValueError: If signature verification fails (when required)
    """
    # TODO: Verify sigil_proof if provided
    # if sigil_proof:
    #     verify_eip191_signature(agent_id, intent_params, sigil_proof)
    
    expires_at = datetime.now(timezone.utc) + timedelta(hours=expires_in_hours)
    
    intent = Intent(
        agent_id=agent_id,
        agent_name=agent_name,
        intent_type=intent_type,
        intent_params=intent_params,
        sigil_proof=sigil_proof,
        rationale=rationale,
        expires_at=expires_at,
        approval_threshold=approval_threshold,
        depends_on=depends_on or [],
    )
    
    return intent_store.create_intent(intent)


def approve_intent(
    intent_id: str,
    approver_id: str,
    approver_signature: Optional[str] = None,
    notes: Optional[str] = None
) -> Intent:
    """
    Approve pending intent.
    
    Args:
        intent_id: Intent to approve
        approver_id: User or agent ID approving
        approver_signature: Optional EIP-191 signature for agent approvers
        notes: Optional approval notes
    
    Returns:
        Updated intent
    
    Raises:
        ValueError: If intent not found or not in pending status
    """
    intent = intent_store.get_intent(intent_id)
    if not intent:
        raise ValueError(f"Intent {intent_id} not found")
    
    if intent.status != IntentStatus.PENDING:
        raise ValueError(f"Intent {intent_id} not in pending status")
    
    # Check expiration
    if datetime.now(timezone.utc) > intent.expires_at:
        intent_store.update_intent(intent_id, status=IntentStatus.EXPIRED)
        raise ValueError(f"Intent {intent_id} has expired")
    
    # TODO: Verify approver_signature if provided
    
    # Add approver
    if approver_id not in intent.approved_by:
        intent.approved_by.append(approver_id)
    
    # Check if threshold met
    if len(intent.approved_by) >= intent.approval_threshold:
        intent.status = IntentStatus.APPROVED
        # Queue for execution
        # TODO: queue_intent_for_execution(intent)
    
    return intent_store.update_intent(
        intent_id,
        status=intent.status,
        approved_by=intent.approved_by
    )


def deny_intent(
    intent_id: str,
    denier_id: str,
    reason: str,
    denier_signature: Optional[str] = None
) -> Intent:
    """
    Deny pending intent.
    
    Args:
        intent_id: Intent to deny
        denier_id: User or agent ID denying
        reason: Reason for denial
        denier_signature: Optional EIP-191 signature
    
    Returns:
        Updated intent
    
    Raises:
        ValueError: If intent not found or not in pending status
    """
    intent = intent_store.get_intent(intent_id)
    if not intent:
        raise ValueError(f"Intent {intent_id} not found")
    
    if intent.status != IntentStatus.PENDING:
        raise ValueError(f"Intent {intent_id} not in pending status")
    
    # TODO: Verify denier_signature if provided
    
    # Deny intent (single denial is enough)
    intent.denied_by.append(denier_id)
    intent.status = IntentStatus.DENIED
    
    return intent_store.update_intent(
        intent_id,
        status=IntentStatus.DENIED,
        denied_by=intent.denied_by,
        execution_error=reason
    )


def execute_intent(intent_id: str, config: Optional[IntentExecutionConfig] = None) -> Intent:
    """
    Execute approved intent.
    
    Args:
        intent_id: Intent to execute
        config: Execution configuration
    
    Returns:
        Updated intent with execution result
    
    Raises:
        ValueError: If intent not approved or dependencies not met
    """
    intent = intent_store.get_intent(intent_id)
    if not intent:
        raise ValueError(f"Intent {intent_id} not found")
    
    if intent.status != IntentStatus.APPROVED:
        raise ValueError(f"Intent {intent_id} not approved for execution")
    
    # Check dependencies
    if intent.depends_on:
        for dep_id in intent.depends_on:
            dep = intent_store.get_intent(dep_id)
            if not dep or dep.status != IntentStatus.EXECUTED:
                raise ValueError(f"Dependency {dep_id} not executed")
    
    config = config or IntentExecutionConfig()
    
    try:
        # TODO: Execute intent based on intent_type
        # This is where you'd dispatch to actual execution handlers
        result = _execute_intent_handler(intent)
        
        intent_store.update_intent(
            intent_id,
            status=IntentStatus.EXECUTED,
            executed_at=datetime.now(timezone.utc),
            execution_result=result
        )
        
    except Exception as e:
        intent_store.update_intent(
            intent_id,
            status=IntentStatus.FAILED,
            execution_error=str(e)
        )
    
    return intent_store.get_intent(intent_id)


def _execute_intent_handler(intent: Intent) -> Dict[str, Any]:
    """Placeholder for actual intent execution logic."""
    # TODO: Implement actual execution based on intent_type
    return {
        "status": "simulated_success",
        "intent_type": intent.intent_type,
        "params": intent.intent_params
    }


def get_intent_status(intent_id: str) -> Dict[str, Any]:
    """Get intent status with full context."""
    intent = intent_store.get_intent(intent_id)
    if not intent:
        raise ValueError(f"Intent {intent_id} not found")
    
    return {
        "intent_id": intent.intent_id,
        "status": intent.status.value,
        "agent_id": intent.agent_id,
        "agent_name": intent.agent_name,
        "intent_type": intent.intent_type,
        "created_at": intent.created_at.isoformat(),
        "expires_at": intent.expires_at.isoformat(),
        "approvals": len(intent.approved_by),
        "approvals_needed": intent.approval_threshold,
        "approved_by": intent.approved_by,
        "denied_by": intent.denied_by,
        "executed_at": intent.executed_at.isoformat() if intent.executed_at else None,
        "can_approve": intent.status == IntentStatus.PENDING and len(intent.approved_by) < intent.approval_threshold,
        "can_execute": intent.status == IntentStatus.APPROVED,
    }


# ============================================================================
# Example Usage
# ============================================================================


if __name__ == "__main__":
    # Example workflow
    print("Intent Approval Workflow Example\n")
    
    # 1. Agent submits intent
    intent = submit_intent(
        agent_id="agent-123",
        agent_name="btc-forecaster",
        intent_type="execute_trade",
        intent_params={
            "symbol": "BTC/USD",
            "action": "buy",
            "quantity": 1.0,
            "price_limit": 100000
        },
        rationale="Model predicts 95% probability of BTC reaching $100k",
        approval_threshold=2  # Needs 2 approvals
    )
    print(f"1. Intent submitted: {intent.intent_id}")
    print(f"   Status: {intent.status.value}")
    
    # 2. First approval
    intent = approve_intent(intent.intent_id, approver_id="user-alice")
    print(f"\n2. First approval by user-alice")
    print(f"   Status: {intent.status.value}")
    print(f"   Approvals: {len(intent.approved_by)}/{intent.approval_threshold}")
    
    # 3. Second approval (reaches threshold)
    intent = approve_intent(intent.intent_id, approver_id="user-bob")
    print(f"\n3. Second approval by user-bob")
    print(f"   Status: {intent.status.value}")
    print(f"   Approvals: {len(intent.approved_by)}/{intent.approval_threshold}")
    
    # 4. Execute intent
    intent = execute_intent(intent.intent_id)
    print(f"\n4. Intent executed")
    print(f"   Status: {intent.status.value}")
    print(f"   Executed at: {intent.executed_at}")
    print(f"   Result: {intent.execution_result}")
    
    # 5. Get final status
    status = get_intent_status(intent.intent_id)
    print(f"\n5. Final status:")
    for key, value in status.items():
        print(f"   {key}: {value}")
