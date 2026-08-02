"""Bounded prompt-free projection for independent proof verifiers."""

from __future__ import annotations

import datetime
import hashlib
import json
from typing import Any

from agent_harness.errors import HarnessError
from agent_harness.goals import goal_contract_digest
from agent_harness.ids import require_uuid, utc_now
from agent_harness.models import Evidence, Goal, Milestone
from agent_harness.orchestration import command_envelope_digest
from agent_harness.storage import StateStore

PROOF_SCHEMA = "p13i/agent-harness/proof-snapshot/v1"
MAX_PROOF_RECORDS = 1_200
MAX_PROOF_EVENTS = 1_000
MAX_PROOF_TOTAL_EVENTS = 50_000
MAX_TRANSITION_LEDGER_RECORDS = 1_200
DIGEST_ALGORITHM = "sha256-python-canonical-json-v1"

_EVENT_METADATA_KEYS = frozenset(
    {
        "action",
        "adoption_id",
        "approval_id",
        "attempt_id",
        "checkpoint_id",
        "code",
        "command_id",
        "control_command_id",
        "context_digest",
        "child_gate_remaining",
        "child_permission_mode",
        "instruction_digest",
        "workspace_instruction_digest",
        "plan_digest",
        "promotion_id",
        "previous_goal_id",
        "next_goal_id",
        "previous_goal_digest",
        "next_goal_digest",
        "authorization_digest",
        "skill_digest",
        "step_digest",
        "excluded_provider",
        "execution_profile",
        "epoch_id",
        "fingerprint",
        "generation_digest",
        "goal_id",
        "id",
        "lease_id",
        "invalidation_id",
        "next_command_digest",
        "next_turn_ref",
        "pair_digest",
        "policy_sha256",
        "prior_checkpoint_id",
        "prior_command_id",
        "prior_command_type",
        "prior_anchor_kind",
        "prior_reconciliation_id",
        "prior_reconciliation_resolution",
        "prior_generation_digest",
        "prior_material_digest",
        "provider",
        "parent_permission_mode",
        "reconciliation_id",
        "recovery_stage",
        "stage",
        "tool_use_id",
        "transition_sequence",
        "target_command_id",
        "workspace_material_digest",
        "child_id",
        "receiver_thread_ids",
        "receiverThreadIds",
        "usage",
    }
)


def proof_snapshot(
    store: StateStore,
    session_id: str,
    *,
    after_sequence: int = 0,
    event_limit: int = MAX_PROOF_EVENTS,
    through_sequence: int | None = None,
    snapshot_id: str = "",
) -> dict[str, Any]:
    require_uuid(session_id, "session_id")
    if after_sequence < 0:
        raise ValueError("after_sequence must not be negative")
    bounded_event_limit = max(1, min(event_limit, MAX_PROOF_EVENTS))
    if snapshot_id:
        require_uuid(snapshot_id, "snapshot_id")
        retained = store.proof_snapshot(snapshot_id)
        if str(retained.get("session_id", "")) != session_id:
            raise ValueError("proof snapshot belongs to another session")
        retained_through = _integer(retained.get("through_sequence"))
        if through_sequence is not None:
            if through_sequence != retained_through:
                raise ValueError("through_sequence does not match proof snapshot")
        return _proof_page(
            retained,
            after_sequence=after_sequence,
            event_limit=bounded_event_limit,
        )

    source = store.proof_source(
        session_id,
        through_sequence,
        MAX_PROOF_TOTAL_EVENTS,
    )
    through_sequence = _integer(source.get("through_sequence"))
    event_count = _integer(source.get("event_count"))
    if event_count > MAX_PROOF_TOTAL_EVENTS:
        raise HarnessError(
            "E_PROOF_EVENT_LIMIT",
            "proof snapshot exceeds the 50000-event retention limit",
            status=413,
        )
    session = source["session"]
    portable = _object(source.get("portable_session"))
    tables = _object(portable.get("tables"))
    truncated: list[str] = []

    commands = _bounded_rows(tables, "commands", truncated)
    attempts = _bounded_rows(tables, "provider_attempts", truncated)
    turns = _bounded_rows(tables, "turns", truncated)
    approvals = _bounded_rows(tables, "approvals", truncated)
    checkpoints = _bounded_rows(tables, "checkpoints", truncated)
    reconciliations = _bounded_rows(
        tables,
        "reconciliations",
        truncated,
    )
    context_deliveries = _bounded_rows(
        tables,
        "context_deliveries",
        truncated,
    )
    child_launch_gates = _bounded_rows(
        tables,
        "child_launch_gates",
        truncated,
    )
    child_launch_admissions = _bounded_rows(
        tables,
        "child_launch_admissions",
        truncated,
    )
    dispatches = _bounded_rows(
        tables,
        "command_dispatches",
        truncated,
    )
    routing = _bounded_rows(
        tables,
        "routing_decisions",
        truncated,
    )
    goal_promotions = _bounded_rows(
        tables,
        "goal_promotions",
        truncated,
    )
    goal_promotion_evidence = _bounded_rows(
        tables,
        "goal_promotion_evidence",
        truncated,
    )
    goal_contract_adoptions = _bounded_rows(
        tables,
        "goal_contract_adoptions",
        truncated,
    )
    transition_ledger_rows = _rows(tables, "dispatch_transition_ledger")
    if len(transition_ledger_rows) > MAX_TRANSITION_LEDGER_RECORDS:
        truncated.append("dispatch_transition_ledger")
    transition_ledger_rows = transition_ledger_rows[:MAX_TRANSITION_LEDGER_RECORDS]
    transition_invalidation_ids = {
        str(row.get("invalidation_id", "")) for row in transition_ledger_rows
    }
    all_authorization_receipts = _rows(tables, "authorization_receipts")
    transition_authorization_receipts = [
        row
        for row in all_authorization_receipts
        if str(row.get("schema", ""))
        == "p13i/agent-harness/dispatch-generation-transition-authorization/v1"
    ]
    if len(transition_authorization_receipts) > MAX_TRANSITION_LEDGER_RECORDS:
        truncated.append("dispatch_transition_ledger")
    transition_authorization_receipts = transition_authorization_receipts[
        :MAX_TRANSITION_LEDGER_RECORDS
    ]
    authorization_receipts = [
        row
        for row in all_authorization_receipts
        if str(row.get("schema", ""))
        != "p13i/agent-harness/dispatch-generation-transition-authorization/v1"
    ]
    if len(authorization_receipts) > MAX_PROOF_RECORDS:
        truncated.append("authorization_receipts")
    authorization_receipts = authorization_receipts[:MAX_PROOF_RECORDS]
    transition_policies = _bounded_rows(
        tables,
        "dispatch_transition_policies",
        truncated,
    )
    dispatch_invalidations = [
        row
        for row in _rows(tables, "dispatch_invalidations")
        if str(row.get("invalidation_id", "")) not in transition_invalidation_ids
    ]
    if len(dispatch_invalidations) > MAX_PROOF_RECORDS:
        truncated.append("dispatch_invalidations")
    dispatch_invalidations = dispatch_invalidations[:MAX_PROOF_RECORDS]
    all_event_rows = _rows(source, "event_rows")
    turn_commands = _turn_commands(commands, dispatches)
    command_turns = {
        command_id: turn_id
        for turn_id, command_id in turn_commands.items()
        if command_id
    }
    turn_attempts = {
        str(row.get("turn_id", "")): str(row.get("attempt_id", ""))
        for row in turns
        if str(row.get("turn_id", ""))
    }
    safe_snapshot = [_proof_event(row, turn_commands) for row in all_event_rows]
    transition_policy_proofs = [
        _proof_dispatch_transition_policy(row) for row in transition_policies
    ]
    transition_policy_payloads = {
        str(policy.get("policy_sha256", "")): _object(policy.get("policy"))
        for policy in transition_policy_proofs
    }
    authorization_proofs = [
        _proof_authorization_receipt(row, transition_policy_payloads)
        for row in authorization_receipts
    ]
    transition_authorization_proofs = [
        _proof_authorization_receipt(row, transition_policy_payloads)
        for row in transition_authorization_receipts
    ]
    transition_ledger_proof = _proof_dispatch_transition_ledger(
        transition_ledger_rows,
        transition_authorization_proofs,
        transition_policy_proofs,
    )
    if not bool(transition_ledger_proof.get("complete", False)):
        truncated.append("dispatch_transition_ledger")
    children = _proof_children(
        safe_snapshot,
        turns,
        attempts,
        turn_commands,
    )
    if len(children) > MAX_PROOF_RECORDS:
        truncated.append("children")
        children = children[:MAX_PROOF_RECORDS]

    goal = source.get("goal")
    goal_value: dict[str, Any] | None = None
    if isinstance(goal, Goal):
        evidence = [
            item for item in source.get("evidence", []) if isinstance(item, Evidence)
        ]
        goal_value = _proof_goal(goal, evidence)
    referenced_usage = {
        str(_json_object(row.get("payload_json")).get("usage_sample_id", ""))
        for row in routing
        if str(_json_object(row.get("payload_json")).get("usage_sample_id", ""))
    }
    captured_at = utc_now()
    usage = _usage_proof(
        _object(source.get("portable_global")),
        referenced_usage,
        truncated,
        captured_at,
    )
    usage_by_id = {
        str(item.get("sample_id", "")): item
        for item in usage
        if str(item.get("sample_id", ""))
    }
    attempt_started_at = {
        str(row.get("attempt_id", "")): str(row.get("started_at", ""))
        for row in attempts
        if str(row.get("attempt_id", ""))
    }
    leases = _rows(source, "leases")
    workers = _rows(source, "workers")
    safety = _object(source.get("safety"))
    command_envelope_profiles = {
        str(row.get("command_id", "")): str(row.get("profile", ""))
        for row in _rows(source, "envelopes")
        if str(row.get("command_id", ""))
    }
    if len(leases) > MAX_PROOF_RECORDS:
        truncated.append("leases")
    if len(workers) > MAX_PROOF_RECORDS:
        truncated.append("workers")
    retained_payload = {
        "schema": PROOF_SCHEMA,
        "captured_at": captured_at,
        "digest_algorithm": DIGEST_ALGORITHM,
        "complete": not truncated,
        "truncated": sorted(set(truncated)),
        "state_stable": True,
        "session": {
            "session_id": session.session_id,
            "lifecycle": session.lifecycle,
            "attention": session.attention,
            "permission_mode": session.permission_mode,
            "active_provider": session.active_provider,
            "model": session.model,
            "effort": session.effort,
            "goal_id": session.goal_id,
            "owner_host": session.owner_host,
            "owner_epoch": session.owner_epoch,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "external_ref": session.external_ref,
        },
        "goal": goal_value,
        "goal_history": _proof_goal_history(tables),
        "goal_promotions": [_proof_goal_promotion(row) for row in goal_promotions],
        "goal_promotion_evidence": _proof_goal_promotion_evidence(
            goal_promotion_evidence,
            _rows(tables, "evidence"),
        ),
        "goal_contract_adoptions": [
            _proof_goal_contract_adoption(row) for row in goal_contract_adoptions
        ],
        "authorization_receipts": authorization_proofs,
        "dispatch_transition_policies": transition_policy_proofs,
        "dispatch_transition_ledger": transition_ledger_proof,
        "dispatch_invalidations": [
            {
                "invalidation_id": str(row.get("invalidation_id", "")),
                "session_id": str(row.get("session_id", "")),
                "reason_digest": _text_digest(str(row.get("reason", ""))),
                "authorization_digest": str(row.get("authorization_digest", "")),
                "request_digest": str(row.get("request_digest", "")),
                "idempotency_key_digest": _text_digest(
                    str(row.get("idempotency_key", ""))
                ),
                "created_at": str(row.get("created_at", "")),
            }
            for row in dispatch_invalidations
        ],
        "transition_anchor": _object(source.get("transition_anchor")),
        "safety": {
            "profile": str(safety.get("profile", "")),
            "xhigh_authorizations": _integer(safety.get("xhigh_authorizations")),
            "envelopes": _rows(source, "envelopes"),
            "child_launch_gates": _proof_child_launch_gates(child_launch_gates),
            "child_launch_admissions": _proof_child_launch_admissions(
                child_launch_admissions
            ),
            "incidents": _rows(source, "incidents"),
        },
        "commands": [
            _proof_command(row, command_turns, command_envelope_profiles)
            for row in commands
        ],
        "turns": [_proof_turn(row, turn_commands) for row in turns],
        "attempts": [_proof_attempt(row) for row in attempts],
        "dispatches": [
            _selected(
                row,
                {
                    "attempt_id",
                    "command_id",
                    "session_id",
                    "turn_id",
                    "checkpoint_id",
                    "crossed_boundary",
                    "state",
                    "created_at",
                    "updated_at",
                },
            )
            for row in dispatches
        ],
        "routing": [
            _proof_routing(
                row,
                turn_commands,
                turn_attempts,
                usage_by_id,
                attempt_started_at,
            )
            for row in routing
        ],
        "children": children,
        "events": safe_snapshot,
        "usage": usage,
        "approvals": [_proof_approval(row) for row in approvals],
        "reconciliations": [
            _proof_reconciliation(
                row,
                {
                    str(checkpoint.get("checkpoint_id", "")): checkpoint
                    for checkpoint in checkpoints
                },
            )
            for row in reconciliations
        ],
        "leases": leases[:MAX_PROOF_RECORDS],
        "workers": workers[:MAX_PROOF_RECORDS],
        "checkpoints": [
            _selected(
                row,
                {
                    "checkpoint_id",
                    "session_id",
                    "sequence",
                    "provider",
                    "native_session_id",
                    "base_commit",
                    "patch_digest",
                    "untracked_digest",
                    "context_digest",
                    "created_at",
                },
            )
            for row in checkpoints
        ],
        "context_deliveries": _proof_context_deliveries(
            context_deliveries,
            dispatches,
        ),
    }
    retained_digest = _digest(retained_payload)
    retained = store.create_proof_snapshot(
        session_id,
        through_sequence,
        retained_payload,
        retained_digest,
    )
    return _proof_page(
        retained,
        after_sequence=after_sequence,
        event_limit=bounded_event_limit,
    )


def _proof_page(
    retained: dict[str, Any],
    *,
    after_sequence: int,
    event_limit: int,
) -> dict[str, Any]:
    payload = _object(retained.get("payload"))
    retained_digest = str(retained.get("digest", ""))
    if not retained_digest or _digest(payload) != retained_digest:
        raise HarnessError(
            "E_PROOF_INTEGRITY",
            "retained proof snapshot failed its integrity check",
            status=500,
        )
    all_events = [item for item in payload.get("events", []) if isinstance(item, dict)]
    retained_through = _integer(retained.get("through_sequence"))
    if after_sequence > retained_through:
        raise ValueError("after_sequence exceeds proof snapshot")
    if len(all_events) != retained_through:
        raise ValueError("proof event sequence is not contiguous")
    for expected_sequence, event in enumerate(all_events, start=1):
        if _integer(event.get("sequence")) != expected_sequence:
            raise ValueError("proof event sequence is not contiguous")
    remaining = [
        item for item in all_events if _integer(item.get("sequence")) > after_sequence
    ]
    events = remaining[:event_limit]
    complete = len(remaining) <= event_limit
    next_after_sequence = after_sequence
    if events:
        next_after_sequence = _integer(events[-1].get("sequence"))
    result = dict(payload)
    truncated = [
        str(item) for item in payload.get("truncated", []) if isinstance(item, str)
    ]
    if not complete:
        truncated.append("events")
    result.update(
        {
            "snapshot_id": str(retained.get("snapshot_id", "")),
            "snapshot_digest": retained_digest,
            "complete": not truncated and complete,
            "truncated": sorted(set(truncated)),
            "state_stable": True,
            "events": events,
            "event_range": {
                "after_sequence": after_sequence,
                "next_after_sequence": next_after_sequence,
                "last_sequence": retained_through,
                "through_sequence": retained_through,
                "complete": complete,
                "digest": _digest(events),
                "snapshot_digest": _digest(all_events),
            },
        }
    )
    return result


def _proof_predicate(
    index: int,
    predicate: dict[str, Any],
    evidence: list[Evidence],
) -> dict[str, Any]:
    matching = [
        item.evidence_id for item in evidence if _evidence_matches(predicate, item)
    ]
    definition = {
        "type": str(predicate.get("type", "")),
        "subject": _formal_value(str(predicate.get("subject", ""))),
        "outcome": str(predicate.get("outcome", "passed")),
    }
    if "equals" in predicate:
        definition["field"] = str(predicate.get("field", "value"))
        definition["equals"] = _formal_value(predicate.get("equals"))
    return {
        "index": index,
        "definition": definition,
        "definition_digest": _digest(predicate),
        "satisfied": bool(matching),
        "evidence_ids": matching,
    }


def _proof_goal(goal: Goal, evidence: list[Evidence]) -> dict[str, Any]:
    proof_predicates = list(goal.predicates)
    for milestone in goal.milestones:
        proof_predicates.extend(milestone.predicates)
    fields_by_type = _predicate_fields(tuple(proof_predicates))
    return {
        "goal_id": goal.goal_id,
        "kind": goal.kind,
        "status": goal.status,
        "objective_digest": _text_digest(goal.objective),
        "contract_digest": goal_contract_digest(goal),
        "budgets": goal.budgets,
        "permitted_providers": list(goal.permitted_providers),
        "permitted_efforts": list(goal.permitted_efforts),
        "max_concurrency": goal.max_concurrency,
        "completion_policy": goal.completion_policy,
        "incident_policy": goal.incident_policy,
        "constraints": [
            {
                "index": index,
                "type": "string",
                "digest": _text_digest(value),
            }
            for index, value in enumerate(goal.constraints)
        ],
        "predicates": [
            _proof_predicate(index, predicate, evidence)
            for index, predicate in enumerate(goal.predicates)
        ],
        "milestones": [
            {
                "milestone_id": milestone.milestone_id,
                "title_digest": _text_digest(milestone.title),
                "status": milestone.status,
                "dependencies": list(milestone.dependencies),
                "position": milestone.position,
                "predicates": [
                    _proof_predicate(index, predicate, evidence)
                    for index, predicate in enumerate(milestone.predicates)
                ],
            }
            for milestone in goal.milestones
        ],
        "evidence": [_proof_evidence(item, fields_by_type) for item in evidence],
        "created_at": goal.created_at,
        "updated_at": goal.updated_at,
    }


def _proof_goal_history(tables: dict[str, Any]) -> list[dict[str, Any]]:
    milestones_by_goal: dict[str, list[dict[str, Any]]] = {}
    for row in _rows(tables, "goal_milestones"):
        goal_id = str(row.get("goal_id", ""))
        milestones_by_goal.setdefault(goal_id, []).append(row)
    goals = [
        _goal_from_row(
            row,
            milestones_by_goal.get(str(row.get("goal_id", "")), []),
        )
        for row in _rows(tables, "goals")
    ]
    evidence_by_goal: dict[str, list[Evidence]] = {}
    for row in _rows(tables, "evidence"):
        evidence = _evidence_from_row(row)
        evidence_by_goal.setdefault(evidence.goal_id, []).append(evidence)
    result = [
        _proof_goal(goal, evidence_by_goal.get(goal.goal_id, [])) for goal in goals
    ]
    return sorted(
        result,
        key=lambda item: (
            str(item.get("created_at", "")),
            str(item.get("goal_id", "")),
        ),
    )


def _goal_from_row(
    row: dict[str, Any],
    milestone_rows: list[dict[str, Any]],
) -> Goal:
    constraints = tuple(str(item) for item in _json_list(row.get("constraints_json")))
    predicates = tuple(
        item
        for item in _json_list(row.get("predicates_json"))
        if isinstance(item, dict)
    )
    return Goal(
        goal_id=str(row.get("goal_id", "")),
        session_id=str(row.get("session_id", "")),
        kind=str(row.get("kind", "")),
        objective=str(row.get("objective", "")),
        status=str(row.get("status", "")),
        constraints=constraints,
        predicates=predicates,
        budgets=_json_object(row.get("budgets_json")),
        created_at=str(row.get("created_at", "")),
        updated_at=str(row.get("updated_at", "")),
        milestones=tuple(
            _milestone_from_row(item)
            for item in sorted(
                milestone_rows,
                key=lambda value: (
                    _integer(value.get("position")),
                    str(value.get("milestone_id", "")),
                ),
            )
        ),
        permitted_providers=tuple(
            str(item) for item in _json_list(row.get("permitted_providers_json"))
        ),
        permitted_efforts=tuple(
            str(item) for item in _json_list(row.get("permitted_efforts_json"))
        ),
        max_concurrency=_integer(row.get("max_concurrency", 1)),
        completion_policy=str(row.get("completion_policy", "evidence-all")),
        incident_policy=str(row.get("incident_policy", "recover-then-pause")),
    )


def _milestone_from_row(row: dict[str, Any]) -> Milestone:
    return Milestone(
        milestone_id=str(row.get("milestone_id", "")),
        title=str(row.get("title", "")),
        status=str(row.get("status", "")),
        dependencies=tuple(
            str(item) for item in _json_list(row.get("dependencies_json"))
        ),
        predicates=tuple(
            item
            for item in _json_list(row.get("predicates_json"))
            if isinstance(item, dict)
        ),
        position=_integer(row.get("position")),
    )


def _evidence_from_row(row: dict[str, Any]) -> Evidence:
    return Evidence(
        evidence_id=str(row.get("evidence_id", "")),
        goal_id=str(row.get("goal_id", "")),
        evidence_type=str(row.get("evidence_type", "")),
        subject=str(row.get("subject", "")),
        outcome=str(row.get("outcome", "")),
        value=_json_object(row.get("value_json")),
        created_at=str(row.get("created_at", "")),
    )


def _proof_goal_promotion(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "promotion_id": str(row.get("promotion_id", "")),
        "session_id": str(row.get("session_id", "")),
        "previous_goal_id": str(row.get("previous_goal_id", "")),
        "next_goal_id": str(row.get("next_goal_id", "")),
        "stage": str(row.get("stage", "")),
        "authorization_digest": str(row.get("authorization_digest", "")),
        "request_digest": str(row.get("request_digest", "")),
        "idempotency_key_digest": _text_digest(str(row.get("idempotency_key", ""))),
        "previous_goal_digest": str(row.get("previous_goal_digest", "")),
        "next_goal_digest": str(row.get("next_goal_digest", "")),
        "created_at": str(row.get("created_at", "")),
    }


def _proof_goal_contract_adoption(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "adoption_id": str(row.get("adoption_id", "")),
        "session_id": str(row.get("session_id", "")),
        "previous_goal_id": str(row.get("previous_goal_id", "")),
        "next_goal_id": str(row.get("next_goal_id", "")),
        "external_ref": {
            "orchestrator": str(row.get("external_orchestrator", "")),
            "job_id": str(row.get("external_job_id", "")),
        },
        "authorization_digest": str(row.get("authorization_digest", "")),
        "request_digest": str(row.get("request_digest", "")),
        "creation_digest": str(row.get("creation_digest", "")),
        "previous_goal_digest": str(row.get("previous_goal_digest", "")),
        "next_goal_digest": str(row.get("next_goal_digest", "")),
        "idempotency_key_digest": _text_digest(str(row.get("idempotency_key", ""))),
        "created_at": str(row.get("created_at", "")),
    }


def _proof_authorization_receipt(
    row: dict[str, Any],
    transition_policies: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    payload = _json_object(row.get("payload_json"))
    receipt = payload.get("receipt")
    if not isinstance(receipt, dict):
        receipt = {}
    authorization_digest = str(row.get("authorization_digest", ""))
    receipt_sha256 = str(row.get("receipt_sha256", ""))
    binding = {
        name: payload[name]
        for name in (
            "session_id",
            "from_goal_id",
            "stage",
            "budget_increases",
            "next_goal_contract_digest",
            "external_ref",
            "previous_goal_digest",
            "goal_envelope_digest",
            "goal_id",
            "prior_command_id",
            "prior_command_type",
            "prior_anchor_kind",
            "prior_reconciliation_id",
            "prior_reconciliation_resolution",
            "prior_checkpoint_id",
            "prior_generation_digest",
            "prior_material_digest",
            "next_turn_ref",
            "transition_sequence",
            "epoch_id",
            "external_orchestrator",
            "external_job_id",
            "policy_sha256",
            "policy_ref",
            "next_command_digest",
        )
        if name in payload
    }
    if "reason" in payload:
        binding["reason_digest"] = _text_digest(str(payload.get("reason", "")))
    authorization_material = payload
    transition_schema = (
        "p13i/agent-harness/dispatch-generation-transition-authorization/v1"
    )
    if (
        str(row.get("schema", "")) == transition_schema
        and _integer(payload.get("transition_sequence")) == 1
    ):
        policy_sha256 = str(payload.get("policy_sha256", ""))
        policy = transition_policies.get(policy_sha256)
        if policy is not None:
            authorization_material = dict(payload)
            authorization_material.pop("policy_ref", None)
            authorization_material["policy"] = policy
    return {
        "authorization_digest": authorization_digest,
        "session_id": str(row.get("session_id", "")),
        "operation": str(row.get("operation", "")),
        "operation_id": str(row.get("operation_id", "")),
        "schema": str(row.get("schema", "")),
        "receipt_sha256": receipt_sha256,
        "binding": binding,
        "binding_digest": _digest(binding),
        "receipt_digest": _digest(receipt),
        "authorization_digest_valid": (
            _digest(authorization_material) == authorization_digest
        ),
        "receipt_digest_valid": _digest(receipt) == receipt_sha256,
        "created_at": str(row.get("created_at", "")),
    }


def _proof_dispatch_transition_policy(row: dict[str, Any]) -> dict[str, Any]:
    payload = _json_object(row.get("payload_json"))
    policy_sha256 = str(row.get("policy_sha256", ""))
    transitions = payload.get("transitions")
    max_transitions = payload.get("max_transitions")
    allowed_roles = payload.get("allowed_agent_roles")
    allowed_prefixes = payload.get("allowed_step_prefixes")
    ordered = isinstance(transitions, list)
    stages_valid = ordered
    if ordered:
        for index, transition in enumerate(transitions, start=1):
            if not isinstance(transition, dict):
                ordered = False
                stages_valid = False
                break
            if transition.get("sequence") != index:
                ordered = False
                stages_valid = False
                break
            transition_ref = _object(transition.get("next_turn_ref"))
            step_id = str(transition_ref.get("step_id", ""))
            agent_role = str(transition_ref.get("agent_role", ""))
            role_valid = isinstance(allowed_roles, list) and (
                agent_role in allowed_roles
            )
            prefix_valid = False
            if isinstance(allowed_prefixes, list):
                prefix_valid = any(
                    isinstance(prefix, str)
                    and bool(prefix)
                    and step_id.startswith(prefix)
                    for prefix in allowed_prefixes
                )
            command_digest = str(transition.get("next_command_digest", ""))
            digest_valid = len(command_digest) == 64 and not any(
                character not in "0123456789abcdef" for character in command_digest
            )
            if not step_id or not agent_role or not role_valid:
                stages_valid = False
            if not prefix_valid or not digest_valid:
                stages_valid = False
    contract_valid = (
        payload.get("schema") == str(row.get("schema", ""))
        and payload.get("session_id") == str(row.get("session_id", ""))
        and payload.get("epoch_id") == str(row.get("epoch_id", ""))
        and isinstance(payload.get("external_ref"), dict)
        and isinstance(allowed_roles, list)
        and bool(allowed_roles)
        and isinstance(allowed_prefixes, list)
        and bool(allowed_prefixes)
        and isinstance(max_transitions, int)
        and not isinstance(max_transitions, bool)
        and 0 < max_transitions <= 1_000
        and isinstance(transitions, list)
        and len(transitions) == max_transitions
        and ordered
        and stages_valid
    )
    return {
        "policy_sha256": policy_sha256,
        "session_id": str(row.get("session_id", "")),
        "goal_id": str(row.get("goal_id", "")),
        "epoch_id": str(row.get("epoch_id", "")),
        "schema": str(row.get("schema", "")),
        "policy": payload,
        "policy_digest_valid": _digest(payload) == policy_sha256,
        "policy_contract_valid": contract_valid,
        "created_at": str(row.get("created_at", "")),
    }


def _proof_dispatch_transition_ledger(
    ledger_rows: list[dict[str, Any]],
    authorization_receipts: list[dict[str, Any]],
    policies: list[dict[str, Any]],
) -> dict[str, Any]:
    policy_by_digest = {
        str(policy.get("policy_sha256", "")): policy
        for policy in policies
        if str(policy.get("policy_sha256", ""))
    }
    authorization_by_invalidation = {
        str(authorization.get("operation_id", "")): authorization
        for authorization in authorization_receipts
        if str(authorization.get("operation_id", ""))
    }
    receipts: list[dict[str, Any]] = []
    sequences: dict[tuple[str, str], list[int]] = {}
    complete = len(authorization_by_invalidation) == len(ledger_rows)
    ordered = sorted(
        ledger_rows,
        key=lambda row: (
            str(row.get("created_at", "")),
            str(row.get("invalidation_id", "")),
        ),
    )
    for row in ordered:
        invalidation_id = str(row.get("invalidation_id", ""))
        authorization = authorization_by_invalidation.get(invalidation_id, {})
        binding = _object(authorization.get("binding"))
        policy_ref = _object(binding.get("policy_ref"))
        policy_sha256 = str(row.get("policy_sha256", ""))
        goal_id = str(row.get("goal_id", ""))
        epoch_id = str(row.get("epoch_id", ""))
        transition_sequence = _integer(row.get("transition_sequence"))
        policy = policy_by_digest.get(policy_sha256)
        expected_policy_ref = {
            "policy_sha256": policy_sha256,
            "session_id": str(row.get("session_id", "")),
            "goal_id": goal_id,
            "epoch_id": epoch_id,
        }
        next_turn_ref = _json_object(row.get("next_turn_ref_json"))
        state = str(row.get("state", ""))
        reserved_command_id = str(row.get("reserved_command_id", ""))
        consumed_command_id = str(row.get("consumed_command_id", ""))
        state_valid = state in {
            "authorized",
            "reserved",
            "released",
            "consumed",
        }
        if state in {"authorized", "released"}:
            state_valid = state_valid and not reserved_command_id
            state_valid = state_valid and not consumed_command_id
        if state == "reserved":
            state_valid = state_valid and bool(reserved_command_id)
            state_valid = state_valid and not consumed_command_id
        if state == "consumed":
            state_valid = state_valid and bool(reserved_command_id)
            state_valid = state_valid and consumed_command_id == reserved_command_id
        binding_matches = (
            binding.get("goal_id") == goal_id
            and binding.get("epoch_id") == epoch_id
            and binding.get("transition_sequence") == transition_sequence
            and binding.get("policy_sha256") == policy_sha256
            and binding.get("prior_command_id") == str(row.get("prior_command_id", ""))
            and binding.get("prior_command_type")
            == str(row.get("prior_command_type", ""))
            and binding.get("prior_anchor_kind")
            == str(row.get("prior_anchor_kind", ""))
            and binding.get("prior_reconciliation_id")
            == str(row.get("prior_reconciliation_id", ""))
            and binding.get("prior_reconciliation_resolution")
            == str(row.get("prior_reconciliation_resolution", ""))
            and binding.get("prior_checkpoint_id")
            == str(row.get("prior_checkpoint_id", ""))
            and binding.get("prior_generation_digest")
            == str(row.get("prior_generation_digest", ""))
            and binding.get("prior_material_digest")
            == str(row.get("prior_material_digest", ""))
            and _object(binding.get("next_turn_ref")) == next_turn_ref
            and binding.get("next_command_digest")
            == str(row.get("next_command_digest", ""))
        )
        policy_identity_valid = False
        policy_stage_valid = False
        if policy is not None:
            policy_payload = _object(policy.get("policy"))
            policy_identity_valid = (
                policy.get("session_id") == str(row.get("session_id", ""))
                and policy.get("goal_id") == goal_id
                and policy.get("epoch_id") == epoch_id
                and policy_payload.get("session_id") == str(row.get("session_id", ""))
                and policy_payload.get("epoch_id") == epoch_id
                and policy_payload.get("external_ref")
                == {
                    "orchestrator": binding.get("external_orchestrator"),
                    "job_id": binding.get("external_job_id"),
                }
            )
            policy_transitions = policy_payload.get("transitions")
            if (
                isinstance(policy_transitions, list)
                and transition_sequence > 0
                and transition_sequence <= len(policy_transitions)
            ):
                policy_stage = policy_transitions[transition_sequence - 1]
                if isinstance(policy_stage, dict):
                    allowed_roles = policy_payload.get("allowed_agent_roles")
                    allowed_prefixes = policy_payload.get("allowed_step_prefixes")
                    agent_role = str(next_turn_ref.get("agent_role", ""))
                    step_id = str(next_turn_ref.get("step_id", ""))
                    role_allowed = (
                        isinstance(allowed_roles, list) and agent_role in allowed_roles
                    )
                    prefix_allowed = False
                    if isinstance(allowed_prefixes, list):
                        prefix_allowed = any(
                            isinstance(prefix, str)
                            and bool(prefix)
                            and step_id.startswith(prefix)
                            for prefix in allowed_prefixes
                        )
                    policy_stage_valid = (
                        policy_stage.get("sequence") == transition_sequence
                        and _object(policy_stage.get("next_turn_ref")) == next_turn_ref
                        and policy_stage.get("next_command_digest")
                        == str(row.get("next_command_digest", ""))
                        and role_allowed
                        and prefix_allowed
                    )
        valid = (
            bool(authorization.get("authorization_digest_valid", False))
            and bool(authorization.get("receipt_digest_valid", False))
            and authorization.get("authorization_digest")
            == str(row.get("authorization_digest", ""))
            and authorization.get("receipt_sha256")
            == str(row.get("receipt_sha256", ""))
            and policy is not None
            and bool(policy.get("policy_digest_valid", False))
            and bool(policy.get("policy_contract_valid", False))
            and policy_identity_valid
            and policy_stage_valid
            and policy_ref == expected_policy_ref
            and binding_matches
            and state_valid
            and transition_sequence > 0
        )
        if not valid:
            complete = False
        group = (goal_id, epoch_id)
        sequences.setdefault(group, []).append(transition_sequence)
        safe_request_binding = {
            "reason_digest": str(binding.get("reason_digest", "")),
            "prior_command_id": str(row.get("prior_command_id", "")),
            "prior_command_type": str(row.get("prior_command_type", "")),
            "prior_anchor_kind": str(row.get("prior_anchor_kind", "")),
            "prior_reconciliation_id": str(row.get("prior_reconciliation_id", "")),
            "prior_reconciliation_resolution": str(
                row.get("prior_reconciliation_resolution", "")
            ),
            "prior_checkpoint_id": str(row.get("prior_checkpoint_id", "")),
            "prior_generation_digest": str(row.get("prior_generation_digest", "")),
            "prior_material_digest": str(row.get("prior_material_digest", "")),
            "next_turn_ref": next_turn_ref,
            "transition_sequence": transition_sequence,
            "next_command_digest": str(row.get("next_command_digest", "")),
            "authorization_digest": str(row.get("authorization_digest", "")),
        }
        receipts.append(
            {
                "authorization_digest": str(row.get("authorization_digest", "")),
                "invalidation_id": invalidation_id,
                "request_digest": str(row.get("request_digest", "")),
                "goal_id": goal_id,
                "epoch_id": epoch_id,
                "transition_sequence": transition_sequence,
                "policy_sha256": policy_sha256,
                "policy_ref": policy_ref,
                "prior_command_id": str(row.get("prior_command_id", "")),
                "prior_command_type": str(row.get("prior_command_type", "")),
                "prior_anchor_kind": str(row.get("prior_anchor_kind", "")),
                "prior_reconciliation_id": str(row.get("prior_reconciliation_id", "")),
                "prior_reconciliation_resolution": str(
                    row.get("prior_reconciliation_resolution", "")
                ),
                "prior_checkpoint_id": str(row.get("prior_checkpoint_id", "")),
                "prior_generation_digest": str(row.get("prior_generation_digest", "")),
                "prior_material_digest": str(row.get("prior_material_digest", "")),
                "authorization": authorization,
                "safe_request_binding": safe_request_binding,
                "safe_request_binding_digest": _digest(safe_request_binding),
                "next_turn_ref": next_turn_ref,
                "next_command_digest": str(row.get("next_command_digest", "")),
                "state": state,
                "reserved_command_id": reserved_command_id,
                "consumed": state == "consumed",
                "consumed_command_id": consumed_command_id,
                "valid": valid,
                "created_at": str(row.get("created_at", "")),
                "updated_at": str(row.get("updated_at", "")),
            }
        )
    for values in sequences.values():
        if sorted(values) != list(range(1, len(values) + 1)):
            complete = False
    ledger_material = {
        "policies": policies,
        "receipts": receipts,
    }
    return {
        "schema": "p13i/agent-harness/dispatch-transition-ledger/v1",
        "bounded_limit": MAX_TRANSITION_LEDGER_RECORDS,
        "policy_count": len(policies),
        "receipt_count": len(receipts),
        "complete": complete,
        "ledger_digest": _digest(ledger_material),
        **ledger_material,
    }


def _proof_goal_promotion_evidence(
    rows: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence_by_id = {str(row.get("evidence_id", "")): row for row in evidence_rows}
    result: list[dict[str, Any]] = []
    for row in rows:
        source_id = str(row.get("source_evidence_id", ""))
        copied_id = str(row.get("copied_evidence_id", ""))
        source = evidence_by_id.get(source_id, {})
        copied = evidence_by_id.get(copied_id, {})
        source_contract = _lineage_evidence_contract(source)
        source_contract["source_evidence_id"] = source_id
        stored_digest = str(row.get("value_digest", ""))
        result.append(
            {
                "promotion_id": str(row.get("promotion_id", "")),
                "source": _proof_lineage_evidence(source),
                "copied": _proof_lineage_evidence(copied),
                "source_contract_digest": _digest(source_contract),
                "stored_contract_digest": stored_digest,
                "digest_valid": _digest(source_contract) == stored_digest,
                "copied_matches_source": (
                    _lineage_evidence_contract(source)
                    == _lineage_evidence_contract(copied)
                ),
                "created_at": str(row.get("created_at", "")),
            }
        )
    return result


def _lineage_evidence_contract(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_type": str(row.get("evidence_type", "")),
        "subject": str(row.get("subject", "")),
        "outcome": str(row.get("outcome", "")),
        "value": _json_object(row.get("value_json")),
    }


def _proof_lineage_evidence(row: dict[str, Any]) -> dict[str, Any]:
    contract = _lineage_evidence_contract(row)
    return {
        "evidence_id": str(row.get("evidence_id", "")),
        "goal_id": str(row.get("goal_id", "")),
        "evidence_type": str(contract["evidence_type"]),
        "subject": _formal_value(contract["subject"]),
        "outcome": str(contract["outcome"]),
        "value_digest": _digest(contract["value"]),
        "created_at": str(row.get("created_at", "")),
    }


def _proof_children(
    events: list[dict[str, Any]],
    turns: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    turn_commands: dict[str, str],
) -> list[dict[str, Any]]:
    turn_policy: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("event_type") != "routing.selected":
            continue
        turn_id = str(event.get("turn_id", ""))
        turn_policy[turn_id] = _object(event.get("metadata"))
    turn_attempts = {
        str(row.get("turn_id", "")): str(row.get("attempt_id", "")) for row in turns
    }
    attempt_providers = {
        str(row.get("attempt_id", "")): str(row.get("provider", "")) for row in attempts
    }
    children: dict[tuple[str, str], dict[str, Any]] = {}
    for event in events:
        event_type = str(event.get("event_type", ""))
        if not event_type.startswith("agent.child."):
            continue
        metadata = _object(event.get("metadata"))
        identities = _child_identities(metadata)
        turn_id = str(event.get("turn_id", ""))
        attempt_id = turn_attempts.get(turn_id, "")
        provider = attempt_providers.get(attempt_id, "")
        policy = turn_policy.get(turn_id, {})
        for child_id, native_thread_ids in identities:
            key = (turn_id, child_id)
            child = children.setdefault(
                key,
                {
                    "child_id": child_id,
                    "child_call_id": str(
                        metadata.get("id", metadata.get("tool_use_id", ""))
                    ),
                    "native_thread_ids": native_thread_ids,
                    "provider": provider,
                    "parent_command_id": turn_commands.get(turn_id, ""),
                    "parent_turn_id": turn_id,
                    "parent_attempt_id": attempt_id,
                    "execution_profile": str(policy.get("execution_profile", "")),
                    "parent_permission_mode": str(
                        policy.get("parent_permission_mode", "")
                    ),
                    "child_permission_mode": str(
                        policy.get("child_permission_mode", "")
                    ),
                    "child_gate_remaining_at_admission": _integer(
                        policy.get("child_gate_remaining")
                    ),
                    "status": "unknown",
                    "started_at": "",
                    "terminal_at": "",
                    "usage": {},
                },
            )
            suffix = event_type.removeprefix("agent.child.")
            if suffix == "started":
                child["status"] = "running"
                child["started_at"] = str(event.get("created_at", ""))
            if suffix in {"completed", "failed", "cancelled"}:
                child["status"] = suffix
                child["terminal_at"] = str(event.get("created_at", ""))
            usage = metadata.get("usage")
            if isinstance(usage, dict):
                child["usage"] = _numeric_tree(usage)
    return sorted(
        children.values(),
        key=lambda item: (
            str(item.get("started_at", "")),
            str(item.get("child_id", "")),
        ),
    )


def _proof_child_launch_gates(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "command_id": str(row.get("command_id", "")),
            "session_id": str(row.get("session_id", "")),
            "permit_limit": _integer(row.get("permit_limit")),
            "consumed": _integer(row.get("consumed")),
            "created_at": str(row.get("created_at", "")),
            "updated_at": str(row.get("updated_at", "")),
        }
        for row in rows
    ]


def _proof_child_launch_admissions(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "command_id": str(row.get("command_id", "")),
            "admission_key_digest": _text_digest(str(row.get("admission_key", ""))),
            "admitted": bool(_integer(row.get("admitted"))),
            "created_at": str(row.get("created_at", "")),
        }
        for row in rows
    ]


def _child_identities(
    metadata: dict[str, Any],
) -> list[tuple[str, list[str]]]:
    native_thread_ids: list[str] = []
    for name in ("receiver_thread_ids", "receiverThreadIds"):
        value = metadata.get(name)
        if isinstance(value, list):
            native_thread_ids = [str(item) for item in value if isinstance(item, str)]
            if native_thread_ids:
                break
    for name in ("child_id", "id", "tool_use_id"):
        value = metadata.get(name)
        if isinstance(value, str) and value:
            return [(value, native_thread_ids)]
    return [(item, [item]) for item in native_thread_ids]


def _proof_evidence(
    item: Evidence,
    fields_by_type: dict[str, set[str]],
) -> dict[str, Any]:
    fields = fields_by_type.get(item.evidence_type, set())
    return {
        "evidence_id": item.evidence_id,
        "type": item.evidence_type,
        "subject": _formal_value(item.subject),
        "outcome": item.outcome,
        "fields": {
            name: _formal_value(item.value.get(name)) for name in sorted(fields)
        },
        "value_digest": _digest(item.value),
        "created_at": item.created_at,
    }


def _proof_routing(
    row: dict[str, Any],
    turn_commands: dict[str, str],
    turn_attempts: dict[str, str],
    usage_by_id: dict[str, dict[str, Any]],
    attempt_started_at: dict[str, str],
) -> dict[str, Any]:
    turn_id = str(row.get("turn_id", ""))
    payload = _json_object(row.get("payload_json"))
    attempt_id = turn_attempts.get(turn_id, "")
    usage_sample_id = str(payload.get("usage_sample_id", ""))
    usage = usage_by_id.get(usage_sample_id, {})
    route_recorded_at = str(row.get("created_at", ""))
    attempt_admitted_at = attempt_started_at.get(attempt_id, "")
    usage_observed_at = str(payload.get("usage_observed_at", ""))
    usage_sample_bound = _usage_sample_bound(row, payload, usage)
    fresh_at_route = False
    fresh_at_attempt_admission = False
    if usage_sample_bound:
        fresh_at_route = _usage_is_fresh(
            usage_observed_at,
            route_recorded_at,
        )
        fresh_at_attempt_admission = _usage_is_fresh(
            usage_observed_at,
            attempt_admitted_at,
        )
    admissible_at_route = _usage_admissible_at_route(
        payload,
        usage,
        usage_sample_bound,
        fresh_at_route,
    )
    return {
        "decision_id": str(row.get("decision_id", "")),
        "session_id": str(row.get("session_id", "")),
        "command_id": turn_commands.get(turn_id, ""),
        "turn_id": turn_id,
        "attempt_id": attempt_id,
        "provider": str(row.get("provider", "")),
        "model": str(row.get("model", "")),
        "effort": str(row.get("effort", "")),
        "usage_sample_id": usage_sample_id,
        "usage_observed_at": usage_observed_at,
        "usage_sample_bound": usage_sample_bound,
        "route_recorded_at": route_recorded_at,
        "attempt_admitted_at": attempt_admitted_at,
        "usage_age_seconds_at_route": _usage_age_seconds(
            usage_observed_at,
            route_recorded_at,
        ),
        "usage_age_seconds_at_attempt_admission": _usage_age_seconds(
            usage_observed_at,
            attempt_admitted_at,
        ),
        "fresh_at_route": fresh_at_route,
        "fresh_at_attempt_admission": fresh_at_attempt_admission,
        "admissible_at_route": admissible_at_route,
        "binding_percent": payload.get("binding_percent"),
        "credits_engaged": _optional_boolean(payload.get("credits_engaged")),
        "required_capabilities": _strings(payload.get("required_capabilities")),
        "metered_budget": payload.get("metered_budget"),
        "binding_ceiling": payload.get("binding_ceiling"),
        "execution_profile": str(payload.get("execution_profile", "")),
        "workload": str(payload.get("workload", "")),
        "payload_digest": _digest(payload),
        "created_at": route_recorded_at,
    }


def _usage_sample_bound(
    routing_row: dict[str, Any],
    payload: dict[str, Any],
    usage: dict[str, Any],
) -> bool:
    if not usage:
        return False
    return bool(
        str(usage.get("provider", "")) == str(routing_row.get("provider", ""))
        and str(usage.get("observed_at", ""))
        == str(payload.get("usage_observed_at", ""))
        and usage.get("binding_percent") == payload.get("binding_percent")
        and usage.get("credits_engaged")
        == _optional_boolean(payload.get("credits_engaged"))
    )


def _usage_admissible_at_route(
    payload: dict[str, Any],
    usage: dict[str, Any],
    usage_sample_bound: bool,
    fresh_at_route: bool,
) -> bool:
    if not usage_sample_bound or not fresh_at_route:
        return False
    if bool(usage.get("error_present", False)):
        return False
    binding = payload.get("binding_percent")
    if not isinstance(binding, (int, float)) or isinstance(binding, bool):
        return False
    ceiling = payload.get("binding_ceiling")
    if not isinstance(ceiling, (int, float)) or isinstance(ceiling, bool):
        ceiling = 90.0
    if float(binding) >= float(ceiling):
        return False
    return _optional_boolean(payload.get("credits_engaged")) is False


def _proof_command(
    row: dict[str, Any],
    command_turns: dict[str, str],
    command_envelope_profiles: dict[str, str],
) -> dict[str, Any]:
    result = _json_object(row.get("result_json"))
    command_id = str(row.get("command_id", ""))
    turn_id = str(result.get("turn_id", ""))
    if not turn_id:
        turn_id = command_turns.get(command_id, "")
    profile = command_envelope_profiles.get(command_id, "")
    envelope_digest = ""
    if profile:
        envelope_digest = command_envelope_digest(
            str(row.get("command_type", "")),
            _json_object(row.get("payload_json")),
            profile,
        )
    return {
        "command_id": command_id,
        "session_id": str(row.get("session_id", "")),
        "command_envelope_digest": envelope_digest,
        "idempotency_key_digest": _text_digest(str(row.get("idempotency_key", ""))),
        "command_type": str(row.get("command_type", "")),
        "status": str(row.get("status", "")),
        "turn_id": turn_id,
        "turn_ref": {
            "step_id": str(row.get("turn_step_id", "")),
            "agent_role": str(row.get("turn_agent_role", "")),
        },
        "result": _proof_result(result),
        "created_at": str(row.get("created_at", "")),
        "updated_at": str(row.get("updated_at", "")),
    }


def _proof_result(value: dict[str, Any]) -> dict[str, Any]:
    result = _selected(
        value,
        {
            "checkpoint_id",
            "code",
            "effort",
            "model",
            "native_session_id",
            "provider",
            "retryable",
            "status",
            "target_command_id",
            "turn_id",
            "workspace_material_digest",
        },
    )
    usage = value.get("usage")
    if isinstance(usage, dict):
        result["usage"] = _numeric_tree(usage)
    safety = value.get("safety")
    if isinstance(safety, dict):
        result["safety_digest"] = _digest(safety)
    return result


def _proof_turn(
    row: dict[str, Any],
    turn_commands: dict[str, str],
) -> dict[str, Any]:
    turn_id = str(row.get("turn_id", ""))
    return {
        "turn_id": turn_id,
        "command_id": turn_commands.get(turn_id, ""),
        "attempt_id": str(row.get("attempt_id", "")),
        "status": str(row.get("status", "")),
        "replay_of": str(row.get("replay_of", "")),
        "turn_ref": {
            "step_id": str(row.get("turn_step_id", "")),
            "agent_role": str(row.get("turn_agent_role", "")),
        },
        "started_at": str(row.get("started_at", "")),
        "completed_at": str(row.get("completed_at", "")),
    }


def _proof_attempt(row: dict[str, Any]) -> dict[str, Any]:
    return _selected(
        row,
        {
            "attempt_id",
            "session_id",
            "provider",
            "native_session_id",
            "model",
            "effort",
            "auth_mode",
            "status",
            "started_at",
            "ended_at",
        },
    )


def _proof_event(
    row: dict[str, Any],
    turn_commands: dict[str, str],
) -> dict[str, Any]:
    turn_id = str(row.get("turn_id", ""))
    text_value = str(row.get("text", ""))
    metadata = _json_object(row.get("metadata_json"))
    safe_metadata = _selected(metadata, _EVENT_METADATA_KEYS)
    if "usage" in safe_metadata:
        safe_metadata["usage"] = _numeric_tree(safe_metadata["usage"])
    reason = str(metadata.get("reason", ""))
    if reason:
        safe_metadata["reason_digest"] = _text_digest(reason)
    return {
        "sequence": _integer(row.get("sequence")),
        "event_id": str(row.get("event_id", "")),
        "event_type": str(row.get("event_type", "")),
        "role": str(row.get("role", "")),
        "status": str(row.get("status", "")),
        "turn_id": turn_id,
        "command_id": turn_commands.get(
            turn_id,
            str(metadata.get("command_id", "")),
        ),
        "metadata": safe_metadata,
        "metadata_digest": _digest(metadata),
        "text_digest": _text_digest(text_value),
        "blob_digest": str(row.get("blob_digest", "")),
        "created_at": str(row.get("created_at", "")),
    }


def _proof_approval(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "approval_id": str(row.get("approval_id", "")),
        "session_id": str(row.get("session_id", "")),
        "turn_id": str(row.get("turn_id", "")),
        "provider_request_id": str(row.get("provider_request_id", "")),
        "kind": str(row.get("kind", "")),
        "status": str(row.get("status", "")),
        "decision_digest": _text_digest(str(row.get("decision_json", ""))),
        "created_at": str(row.get("created_at", "")),
        "resolved_at": str(row.get("resolved_at", "")),
    }


def _proof_reconciliation(
    row: dict[str, Any],
    checkpoints: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    audit = _json_object(row.get("audit_json"))
    pre_dispatch_checkpoint_id = str(row.get("pre_dispatch_checkpoint_id", ""))
    discovery_checkpoint_id = str(audit.get("discovery_checkpoint_id", ""))
    resolution_checkpoint_id = str(
        audit.get(
            "resolution_checkpoint_id",
            audit.get("checkpoint_id", ""),
        )
    )
    discovery = checkpoints.get(discovery_checkpoint_id)
    resolution = checkpoints.get(resolution_checkpoint_id)
    resolution_workspace_digest = str(audit.get("resolution_workspace_digest", ""))
    resolution_workspace_digest_valid = len(resolution_workspace_digest) == 64 and all(
        character in "0123456789abcdef" for character in resolution_workspace_digest
    )
    material_fields = (
        "base_commit",
        "patch_digest",
        "untracked_digest",
        "context_digest",
    )
    material_equal = False
    if discovery is not None and resolution is not None:
        material_equal = all(
            str(discovery.get(name, "")) == str(resolution.get(name, ""))
            for name in material_fields
        )
    return {
        "reconciliation_id": str(row.get("reconciliation_id", "")),
        "session_id": str(row.get("session_id", "")),
        "command_id": str(row.get("command_id", "")),
        "pre_dispatch_checkpoint_id": pre_dispatch_checkpoint_id,
        "pre_dispatch_checkpoint_bound": (pre_dispatch_checkpoint_id in checkpoints),
        "discovery_checkpoint_id": discovery_checkpoint_id,
        "discovery_checkpoint_bound": discovery is not None,
        "resolution_checkpoint_id": resolution_checkpoint_id,
        "resolution_checkpoint_bound": resolution is not None,
        "resolution_workspace_digest": resolution_workspace_digest,
        "resolution_workspace_digest_valid": resolution_workspace_digest_valid,
        "resolution_material_certified": bool(
            resolution is not None and resolution_workspace_digest_valid
        ),
        "discovery_resolution_material_equal": material_equal,
        "recovery_material_certified": bool(
            str(row.get("resolution", "")) == "accept-current"
            and material_equal
            and resolution_workspace_digest_valid
        ),
        "current_workspace_digest": str(row.get("current_workspace_digest", "")),
        "provider_attempts": _json_list(row.get("provider_attempts_json")),
        "safety_consumption": _json_object(row.get("safety_consumption_json")),
        "status": str(row.get("status", "")),
        "resolution": str(row.get("resolution", "")),
        "audit_digest": _text_digest(str(row.get("audit_json", ""))),
        "created_at": str(row.get("created_at", "")),
        "resolved_at": str(row.get("resolved_at", "")),
    }


def _usage_proof(
    portable: dict[str, Any],
    referenced_ids: set[str],
    truncated: list[str],
    captured_at: str,
) -> list[dict[str, Any]]:
    tables = _object(portable.get("tables"))
    latest: dict[str, dict[str, Any]] = {}
    rows = _rows(tables, "usage_samples")
    for row in rows:
        provider = str(row.get("provider", ""))
        previous = latest.get(provider)
        if previous is None:
            latest[provider] = row
            continue
        if str(row.get("observed_at", "")) > str(previous.get("observed_at", "")):
            latest[provider] = row
    latest_ids = {str(row.get("sample_id", "")) for row in latest.values()}
    selected = [
        row
        for row in rows
        if str(row.get("sample_id", "")) in referenced_ids | latest_ids
    ]
    selected.sort(
        key=lambda row: (
            str(row.get("observed_at", "")),
            str(row.get("sample_id", "")),
        )
    )
    if len(selected) > MAX_PROOF_RECORDS:
        truncated.append("usage")
    result: list[dict[str, Any]] = []
    for row in selected[:MAX_PROOF_RECORDS]:
        payload = _json_object(row.get("payload_json"))
        error_present = bool(str(payload.get("error", "")))
        observed_at = str(row.get("observed_at", ""))
        fresh = _usage_is_fresh(observed_at, captured_at)
        binding = row.get("binding_percent")
        admissible = fresh and not error_present
        if not isinstance(binding, (int, float)) or isinstance(binding, bool):
            admissible = False
        elif float(binding) >= 90.0:
            admissible = False
        credits_engaged = _optional_boolean(row.get("credits_engaged"))
        if credits_engaged is not False:
            admissible = False
        result.append(
            {
                "sample_id": str(row.get("sample_id", "")),
                "provider": str(row.get("provider", "")),
                "observed_at": observed_at,
                "binding_percent": binding,
                "credits_engaged": credits_engaged,
                "error_present": error_present,
                "fresh_at_capture": fresh,
                "admissible_at_90_percent": admissible,
                "latest": str(row.get("sample_id", "")) in latest_ids,
                "source_digest": _text_digest(str(row.get("payload_json", ""))),
            }
        )
    return result


def _usage_is_fresh(observed_at: str, captured_at: str) -> bool:
    age_seconds = _usage_age_seconds(observed_at, captured_at)
    if age_seconds is None:
        return False
    return 0.0 <= age_seconds <= 90.0


def _usage_age_seconds(observed_at: str, measured_at: str) -> float | None:
    try:
        observed = datetime.datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
        measured = datetime.datetime.fromisoformat(measured_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=datetime.UTC)
    if measured.tzinfo is None:
        measured = measured.replace(tzinfo=datetime.UTC)
    age = measured.astimezone(datetime.UTC) - observed.astimezone(datetime.UTC)
    return age.total_seconds()


def _proof_context_deliveries(
    rows: list[dict[str, Any]],
    dispatches: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_attempt = {
        str(item.get("attempt_id", "")): item
        for item in dispatches
        if str(item.get("attempt_id", ""))
    }
    result: list[dict[str, Any]] = []
    for row in rows:
        checkpoint_id = str(row.get("checkpoint_id", ""))
        attempt_id = str(row.get("attempt_id", ""))
        dispatch = by_attempt.get(attempt_id, {})
        binding = {
            "session_id": str(row.get("session_id", "")),
            "provider": str(row.get("provider", "")),
            "context_digest": str(row.get("context_digest", "")),
            "checkpoint_id": checkpoint_id,
            "command_id": str(row.get("command_id", "")),
            "turn_id": str(dispatch.get("turn_id", "")),
            "attempt_id": attempt_id,
            "payload_digest": str(row.get("payload_digest", "")),
        }
        delivery = {
            **binding,
            "idempotency_digest": _digest(binding),
            "state": str(row.get("state", "")),
            "accepted_at": str(row.get("accepted_at", "")),
            "delivered_at": str(row.get("delivered_at", "")),
        }
        delivery["context_delivery_digest"] = _digest(delivery)
        result.append(delivery)
    return result


def _turn_commands(
    commands: list[dict[str, Any]],
    dispatches: list[dict[str, Any]],
) -> dict[str, str]:
    attempts = {
        str(row.get("turn_id", "")): str(row.get("command_id", ""))
        for row in dispatches
        if str(row.get("turn_id", ""))
    }
    for row in commands:
        result = _json_object(row.get("result_json"))
        turn_id = str(result.get("turn_id", ""))
        if turn_id:
            attempts[turn_id] = str(row.get("command_id", ""))
    return attempts


def _bounded_rows(
    tables: dict[str, Any],
    name: str,
    truncated: list[str],
) -> list[dict[str, Any]]:
    values = _rows(tables, name)
    if len(values) > MAX_PROOF_RECORDS:
        truncated.append(name)
    return values[:MAX_PROOF_RECORDS]


def _rows(tables: dict[str, Any], name: str) -> list[dict[str, Any]]:
    value = tables.get(name, [])
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _object(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return value


def _json_object(value: object) -> dict[str, Any]:
    parsed = _json_value(value)
    if not isinstance(parsed, dict):
        return {}
    return parsed


def _json_list(value: object) -> list[Any]:
    parsed = _json_value(value)
    if not isinstance(parsed, list):
        return []
    return parsed


def _json_value(value: object) -> object:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _predicate_fields(
    predicates: tuple[dict[str, Any], ...],
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for predicate in predicates:
        if "equals" not in predicate:
            continue
        predicate_type = str(predicate.get("type", ""))
        field = str(predicate.get("field", "value"))
        result.setdefault(predicate_type, set()).add(field)
    return result


def _evidence_matches(
    predicate: dict[str, Any],
    item: Evidence,
) -> bool:
    if item.evidence_type != str(predicate.get("type", "")):
        return False
    subject = str(predicate.get("subject", ""))
    if subject and item.subject != subject:
        return False
    if item.outcome != str(predicate.get("outcome", "passed")):
        return False
    if "equals" not in predicate:
        return True
    field = str(predicate.get("field", "value"))
    return item.value.get(field) == predicate.get("equals")


def _formal_value(value: object) -> dict[str, Any]:
    if value is None:
        return {"type": "null", "value": None}
    if isinstance(value, bool):
        return {"type": "boolean", "value": value}
    if isinstance(value, int):
        return {"type": "integer", "value": value}
    if isinstance(value, float):
        return {"type": "number", "value": value}
    if isinstance(value, str):
        return {"type": "string", "digest": _text_digest(value)}
    return {"type": "json", "digest": _digest(value)}


def _strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted(str(item) for item in value if isinstance(item, str))


def _selected(
    value: dict[str, Any],
    names: set[str] | frozenset[str],
) -> dict[str, Any]:
    return {name: value[name] for name in names if name in value}


def _numeric_tree(value: object) -> object:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, list):
        return [_numeric_tree(item) for item in value]
    if isinstance(value, dict):
        return {
            str(name): _numeric_tree(item)
            for name, item in value.items()
            if isinstance(item, (dict, list, int, float, bool))
        }
    return None


def _integer(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0


def _optional_boolean(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    return None


def _text_digest(value: str) -> str:
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
