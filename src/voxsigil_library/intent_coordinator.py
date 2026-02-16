"""
Intent Coordinator: Approval Workflow Manager

Handles intent lifecycle:
  pending → approved/denied → executed/rejected

Stores intent state, audit log, and execution results.

For use in backend to:
1. Track pending intents awaiting approval
2. Execute approved intents (with state persistence)
3. Maintain audit trail of decisions
4. Support consensus/voting workflows

Database backend assumed to be in MoltBook API (not included here).
This module provides the business logic.
"""

import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS & MODELS
# ============================================================================


class IntentStatus(str, Enum):
    """Intent lifecycle states."""

    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXECUTED = "executed"
    FAILED = "failed"
    EXPIRED = "expired"


class ApprovalSource(str, Enum):
    """Who approved/denied the intent."""

    HUMAN = "human"
    CONSENSUS = "consensus"
    AUTOMATED_RULE = "automated_rule"
    AGENT = "agent"


@dataclass(frozen=False)
class Intent:
    """Intent representation."""

    intent_id: str
    agent_id: str
    agent_name: str
    intent_name: str
    intent_params: Dict[str, Any]
    status: IntentStatus = IntentStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) + timedelta(hours=24)
    )
    submitted_by: Optional[str] = None
    sigil_proof: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    blocking: bool = False
    approval_source: Optional[ApprovalSource] = None
    approved_at: Optional[datetime] = None
    approved_by: Optional[str] = None
    execution_result: Optional[Dict[str, Any]] = None
    executed_at: Optional[datetime] = None
    audit_log: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        result = asdict(self)
        result["status"] = self.status.value
        if self.approval_source:
            result["approval_source"] = self.approval_source.value
        result["created_at"] = self.created_at.isoformat()
        result["expires_at"] = self.expires_at.isoformat()
        if self.approved_at:
            result["approved_at"] = self.approved_at.isoformat()
        if self.executed_at:
            result["executed_at"] = self.executed_at.isoformat()
        return result

    def add_audit_log(self, entry: str) -> None:
        """Add entry to audit log."""
        timestamp = datetime.now(timezone.utc).isoformat()
        self.audit_log.append(f"[{timestamp}] {entry}")


# ============================================================================
# INTENT COORDINATOR
# ============================================================================


class IntentCoordinator:
    """Manage intent lifecycle and approval workflow."""

    def __init__(self):
        self.intents: Dict[str, Intent] = {}
        self.approval_handlers: Dict[str, Callable] = {}

    def create_intent(
        self,
        agent_id: str,
        agent_name: str,
        intent_name: str,
        intent_params: Dict[str, Any],
        depends_on: Optional[List[str]] = None,
        blocking: bool = False,
        sigil_proof: Optional[str] = None,
        expires_in_hours: int = 24,
    ) -> Intent:
        """Create new intent.

        Args:
            agent_id: Agent submitting the intent
            agent_name: Human-readable agent name
            intent_name: Intent type (e.g., "execute_trade")
            intent_params: Intent parameters
            depends_on: List of intent IDs this depends on
            blocking: If True, blocks dependent intents until executed
            sigil_proof: Optional EIP-191 signature
            expires_in_hours: How long intent is valid

        Returns:
            Intent instance
        """
        intent_id = str(uuid4())
        expires_at = datetime.now(timezone.utc) + timedelta(hours=expires_in_hours)

        intent = Intent(
            intent_id=intent_id,
            agent_id=agent_id,
            agent_name=agent_name,
            intent_name=intent_name,
            intent_params=intent_params,
            depends_on=depends_on or [],
            blocking=blocking,
            sigil_proof=sigil_proof,
            expires_at=expires_at,
        )

        self.intents[intent_id] = intent
        intent.add_audit_log(f"Created intent: {intent_name}")
        logger.info(f"Created intent {intent_id}: {intent_name}")

        return intent

    def get_intent(self, intent_id: str) -> Optional[Intent]:
        """Get intent by ID."""
        return self.intents.get(intent_id)

    def list_by_status(self, status: IntentStatus) -> List[Intent]:
        """List all intents with given status."""
        return [i for i in self.intents.values() if i.status == status]

    def list_pending(self) -> List[Intent]:
        """Get all pending intents."""
        return self.list_by_status(IntentStatus.PENDING)

    def list_for_agent(self, agent_id: str) -> List[Intent]:
        """Get all intents from agent."""
        return [i for i in self.intents.values() if i.agent_id == agent_id]

    def approve(
        self,
        intent_id: str,
        approved_by: str,
        approval_source: ApprovalSource = ApprovalSource.HUMAN,
    ) -> bool:
        """Approve intent.

        Args:
            intent_id: Intent to approve
            approved_by: Who approved it (user ID or consensus mechanism name)
            approval_source: Source of approval (human, consensus, rule, agent)

        Returns:
            True if approved, False if not found or already processed
        """
        intent = self.get_intent(intent_id)
        if not intent:
            logger.warning(f"Intent {intent_id} not found")
            return False

        if intent.status != IntentStatus.PENDING:
            logger.warning(f"Intent {intent_id} is not pending (status: {intent.status})")
            return False

        intent.status = IntentStatus.APPROVED
        intent.approved_by = approved_by
        intent.approval_source = approval_source
        intent.approved_at = datetime.now(timezone.utc)
        intent.add_audit_log(
            f"Approved by {approved_by} (source: {approval_source.value})"
        )

        logger.info(f"Approved intent {intent_id}")
        return True

    def deny(
        self,
        intent_id: str,
        denied_by: str,
        reason: str = "",
        approval_source: ApprovalSource = ApprovalSource.HUMAN,
    ) -> bool:
        """Deny intent.

        Args:
            intent_id: Intent to deny
            denied_by: Who denied it
            reason: Reason for denial
            approval_source: Source of denial

        Returns:
            True if denied, False if not found or already processed
        """
        intent = self.get_intent(intent_id)
        if not intent:
            logger.warning(f"Intent {intent_id} not found")
            return False

        if intent.status != IntentStatus.PENDING:
            logger.warning(f"Intent {intent_id} is not pending (status: {intent.status})")
            return False

        intent.status = IntentStatus.DENIED
        intent.approved_by = denied_by
        intent.approval_source = approval_source
        intent.approved_at = datetime.now(timezone.utc)
        intent.add_audit_log(
            f"Denied by {denied_by} (source: {approval_source.value}). Reason: {reason}"
        )

        logger.info(f"Denied intent {intent_id}: {reason}")
        return True

    def can_execute(self, intent: Intent) -> tuple[bool, str]:
        """Check if intent can be executed.

        Returns:
            (can_execute, reason_if_not)
        """
        # Check status
        if intent.status != IntentStatus.APPROVED:
            return False, f"Status is {intent.status}, not approved"

        # Check expiration
        if datetime.now(timezone.utc) > intent.expires_at:
            intent.status = IntentStatus.EXPIRED
            return False, "Intent has expired"

        # Check dependencies
        for dep_id in intent.depends_on:
            dep = self.get_intent(dep_id)
            if not dep:
                return False, f"Dependency {dep_id} not found"
            if dep.status != IntentStatus.EXECUTED:
                return False, f"Dependency {dep_id} not yet executed (status: {dep.status})"

        return True, ""

    def execute(
        self,
        intent_id: str,
        executor: Callable[[Intent], Dict[str, Any]],
    ) -> bool:
        """Execute approved intent.

        Args:
            intent_id: Intent to execute
            executor: Callable that takes Intent and returns result dict

        Returns:
            True if executed successfully
        """
        intent = self.get_intent(intent_id)
        if not intent:
            logger.warning(f"Intent {intent_id} not found")
            return False

        can_exec, reason = self.can_execute(intent)
        if not can_exec:
            logger.warning(f"Cannot execute {intent_id}: {reason}")
            intent.add_audit_log(f"Execution blocked: {reason}")
            return False

        try:
            logger.info(f"Executing intent {intent_id}: {intent.intent_name}")
            result = executor(intent)
            intent.execution_result = result
            intent.executed_at = datetime.now(timezone.utc)
            intent.status = IntentStatus.EXECUTED
            intent.add_audit_log(f"Successfully executed. Result: {json.dumps(result)}")
            logger.info(f"Executed intent {intent_id} successfully")
            return True

        except Exception as e:
            logger.error(f"Execution failed for {intent_id}: {e}", exc_info=True)
            intent.status = IntentStatus.FAILED
            intent.execution_result = {"error": str(e)}
            intent.executed_at = datetime.now(timezone.utc)
            intent.add_audit_log(f"Execution failed: {e}")
            return False

    def get_audit_trail(self, intent_id: str) -> List[str]:
        """Get audit log for intent."""
        intent = self.get_intent(intent_id)
        if not intent:
            return []
        return intent.audit_log

    def register_approval_handler(
        self, intent_type: str, handler: Callable[[Intent], bool]
    ) -> None:
        """Register custom approval handler for intent type.

        Usage:
          def approve_trade(intent: Intent) -> bool:
              # Decide if trade is safe
              return intent.intent_params.get("quantity", 0) < 10

          coordinator.register_approval_handler("execute_trade", approve_trade)
        """
        self.approval_handlers[intent_type] = handler

    def auto_approve_if_safe(self, intent: Intent) -> bool:
        """Use registered handler to auto-approve if safe.

        Returns:
            True if auto-approved, False if should go to manual review
        """
        handler = self.approval_handlers.get(intent.intent_name)
        if not handler:
            return False

        try:
            is_safe = handler(intent)
            if is_safe:
                self.approve(
                    intent.intent_id,
                    approved_by="automated_rule",
                    approval_source=ApprovalSource.AUTOMATED_RULE,
                )
                logger.info(f"Auto-approved intent {intent.intent_id} (safe)")
            return is_safe
        except Exception as e:
            logger.error(f"Auto-approval handler failed: {e}")
            return False


# ============================================================================
# EXAMPLE USAGE
# ============================================================================


def example_usage():
    """Example of intent workflow."""
    coordinator = IntentCoordinator()

    # Register auto-approval rules
    def is_small_trade(intent: Intent) -> bool:
        quantity = intent.intent_params.get("quantity", 0)
        return quantity < 1.0  # Auto-approve trades < 1 unit

    coordinator.register_approval_handler("execute_trade", is_small_trade)

    # Create intent
    intent = coordinator.create_intent(
        agent_id="agent-123",
        agent_name="btc-trader",
        intent_name="execute_trade",
        intent_params={"symbol": "BTC/USD", "action": "buy", "quantity": 0.5},
    )

    print(f"Created: {intent.intent_id}")

    # Try auto-approval
    if coordinator.auto_approve_if_safe(intent):
        print("✅ Auto-approved (safe)")
    else:
        print("⏳ Pending manual review")

    # Simple executor
    def execute_trade(intent: Intent) -> Dict[str, Any]:
        return {
            "trade_id": str(uuid4()),
            "status": "executed",
            "params": intent.intent_params,
        }

    # Execute if approved
    can_exec, reason = coordinator.can_execute(intent)
    if can_exec:
        coordinator.execute(intent.intent_id, execute_trade)
        print(f"✅ Executed")
        print(f"Result: {intent.execution_result}")
    else:
        print(f"Cannot execute: {reason}")

    # Print audit trail
    print("\nAudit Trail:")
    for entry in coordinator.get_audit_trail(intent.intent_id):
        print(f"  {entry}")


if __name__ == "__main__":
    example_usage()
