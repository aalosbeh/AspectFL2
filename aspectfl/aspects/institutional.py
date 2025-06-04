"""
Institutional Policy Aspects for AspectFL
Implements policy management, conflict resolution, and compliance enforcement
"""

import time
import uuid
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import logging

from ..core.base import Advice, JoinPoint, ExecutionPhase

logger = logging.getLogger(__name__)

class PolicyType(Enum):
    """Types of institutional policies"""
    DATA_GOVERNANCE = "data_governance"
    PRIVACY_PROTECTION = "privacy_protection"
    SECURITY_REQUIREMENT = "security_requirement"
    COMPLIANCE_MANDATE = "compliance_mandate"
    OPERATIONAL_CONSTRAINT = "operational_constraint"
    ETHICAL_GUIDELINE = "ethical_guideline"

class PolicyScope(Enum):
    """Scope of policy application"""
    GLOBAL = "global"
    INSTITUTIONAL = "institutional"
    PROJECT_SPECIFIC = "project_specific"
    CLIENT_SPECIFIC = "client_specific"

@dataclass
class Policy:
    """Represents an institutional policy"""
    id: str
    name: str
    type: PolicyType
    scope: PolicyScope
    institution_id: str
    priority: int  # Higher number = higher priority
    conditions: Dict[str, Any]  # When policy applies
    requirements: Dict[str, Any]  # What policy requires
    penalties: Dict[str, float]  # Penalty weights for violations
    metadata: Dict[str, Any]
    created_at: float
    expires_at: Optional[float] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = time.time()

@dataclass
class PolicyViolation:
    """Represents a policy violation"""
    id: str
    policy_id: str
    violation_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    timestamp: float
    context: str
    details: Dict[str, Any]
    resolved: bool = False
    resolution_actions: List[str] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if self.resolution_actions is None:
            self.resolution_actions = []

@dataclass
class PolicyConflict:
    """Represents a conflict between policies"""
    id: str
    policy_ids: List[str]
    conflict_type: str
    severity: str
    description: str
    resolution_strategy: str
    resolved: bool = False
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

class PolicyEvaluationAdvice(Advice):
    """Advice for evaluating policy compliance"""
    
    def __init__(self, policies: List[Policy] = None):
        self.policies = policies or []
        self.policy_index = self._build_policy_index()
    
    def _build_policy_index(self) -> Dict[str, List[Policy]]:
        """Build index for efficient policy lookup"""
        index = {}
        for policy in self.policies:
            # Index by type
            if policy.type.value not in index:
                index[policy.type.value] = []
            index[policy.type.value].append(policy)
            
            # Index by scope
            if policy.scope.value not in index:
                index[policy.scope.value] = []
            index[policy.scope.value].append(policy)
        
        return index
    
    def execute(self, join_point: JoinPoint) -> JoinPoint:
        """Evaluate policy compliance at join point"""
        applicable_policies = self._find_applicable_policies(join_point)
        violations = []
        compliance_scores = {}
        
        for policy in applicable_policies:
            violation = self._evaluate_policy_compliance(policy, join_point)
            if violation:
                violations.append(violation)
            
            # Compute compliance score for this policy
            compliance_score = self._compute_policy_compliance_score(policy, join_point)
            compliance_scores[policy.id] = compliance_score
        
        # Compute overall compliance score
        overall_compliance = self._compute_overall_compliance(compliance_scores)
        
        # Update join point state
        join_point.state['applicable_policies'] = [p.id for p in applicable_policies]
        join_point.state['policy_violations'] = violations
        join_point.state['policy_compliance_scores'] = compliance_scores
        join_point.state['overall_policy_compliance'] = overall_compliance
        
        if violations:
            logger.warning(f"Policy violations detected: {len(violations)} violations")
        
        return join_point
    
    def _find_applicable_policies(self, join_point: JoinPoint) -> List[Policy]:
        """Find policies applicable to the current join point"""
        applicable = []
        
        for policy in self.policies:
            if self._policy_applies(policy, join_point):
                applicable.append(policy)
        
        # Sort by priority (higher priority first)
        applicable.sort(key=lambda p: p.priority, reverse=True)
        
        return applicable
    
    def _policy_applies(self, policy: Policy, join_point: JoinPoint) -> bool:
        """Check if a policy applies to the current context"""
        # Check expiration
        if policy.expires_at and time.time() > policy.expires_at:
            return False
        
        # Check scope
        if policy.scope == PolicyScope.CLIENT_SPECIFIC:
            required_client = policy.conditions.get('client_id')
            if required_client and required_client != join_point.context:
                return False
        
        # Check phase conditions
        if 'phases' in policy.conditions:
            required_phases = policy.conditions['phases']
            if join_point.phase.value not in required_phases:
                return False
        
        # Check data type conditions
        if 'data_types' in policy.conditions:
            required_types = policy.conditions['data_types']
            data_type = join_point.state.get('data_type')
            if data_type and data_type not in required_types:
                return False
        
        # Check custom conditions
        if 'custom_condition' in policy.conditions:
            condition_func = policy.conditions['custom_condition']
            if callable(condition_func) and not condition_func(join_point):
                return False
        
        return True
    
    def _evaluate_policy_compliance(self, policy: Policy, 
                                  join_point: JoinPoint) -> Optional[PolicyViolation]:
        """Evaluate compliance with a specific policy"""
        requirements = policy.requirements
        
        # Check different types of requirements
        for req_type, req_value in requirements.items():
            violation = self._check_requirement(req_type, req_value, policy, join_point)
            if violation:
                return violation
        
        return None
    
    def _check_requirement(self, req_type: str, req_value: Any, 
                          policy: Policy, join_point: JoinPoint) -> Optional[PolicyViolation]:
        """Check a specific requirement"""
        if req_type == 'encryption_required':
            if req_value and 'encrypted_data' not in join_point.state:
                return PolicyViolation(
                    id=str(uuid.uuid4()),
                    policy_id=policy.id,
                    violation_type='encryption_required',
                    severity='HIGH',
                    timestamp=join_point.timestamp,
                    context=join_point.context,
                    details={'requirement': 'Data must be encrypted', 'policy': policy.name}
                )
        
        elif req_type == 'min_participants':
            num_participants = join_point.state.get('num_participants', 0)
            if num_participants < req_value:
                return PolicyViolation(
                    id=str(uuid.uuid4()),
                    policy_id=policy.id,
                    violation_type='insufficient_participants',
                    severity='MEDIUM',
                    timestamp=join_point.timestamp,
                    context=join_point.context,
                    details={
                        'required': req_value,
                        'actual': num_participants,
                        'policy': policy.name
                    }
                )
        
        elif req_type == 'max_data_age_hours':
            data_timestamp = join_point.state.get('data_timestamp', join_point.timestamp)
            age_hours = (join_point.timestamp - data_timestamp) / 3600
            if age_hours > req_value:
                return PolicyViolation(
                    id=str(uuid.uuid4()),
                    policy_id=policy.id,
                    violation_type='data_too_old',
                    severity='MEDIUM',
                    timestamp=join_point.timestamp,
                    context=join_point.context,
                    details={
                        'max_age_hours': req_value,
                        'actual_age_hours': age_hours,
                        'policy': policy.name
                    }
                )
        
        elif req_type == 'required_metadata':
            metadata = join_point.state.get('metadata', {})
            missing_fields = [field for field in req_value if field not in metadata]
            if missing_fields:
                return PolicyViolation(
                    id=str(uuid.uuid4()),
                    policy_id=policy.id,
                    violation_type='missing_metadata',
                    severity='LOW',
                    timestamp=join_point.timestamp,
                    context=join_point.context,
                    details={
                        'missing_fields': missing_fields,
                        'policy': policy.name
                    }
                )
        
        elif req_type == 'min_security_score':
            security_score = join_point.state.get('security_score', 0.0)
            if security_score < req_value:
                return PolicyViolation(
                    id=str(uuid.uuid4()),
                    policy_id=policy.id,
                    violation_type='insufficient_security',
                    severity='HIGH',
                    timestamp=join_point.timestamp,
                    context=join_point.context,
                    details={
                        'required_score': req_value,
                        'actual_score': security_score,
                        'policy': policy.name
                    }
                )
        
        return None
    
    def _compute_policy_compliance_score(self, policy: Policy, join_point: JoinPoint) -> float:
        """Compute compliance score for a specific policy"""
        requirements = policy.requirements
        scores = []
        
        for req_type, req_value in requirements.items():
            score = self._compute_requirement_score(req_type, req_value, join_point)
            scores.append(score)
        
        return np.mean(scores) if scores else 1.0
    
    def _compute_requirement_score(self, req_type: str, req_value: Any, 
                                 join_point: JoinPoint) -> float:
        """Compute compliance score for a specific requirement"""
        if req_type == 'encryption_required':
            return 1.0 if 'encrypted_data' in join_point.state else 0.0
        
        elif req_type == 'min_participants':
            num_participants = join_point.state.get('num_participants', 0)
            return min(num_participants / req_value, 1.0)
        
        elif req_type == 'max_data_age_hours':
            data_timestamp = join_point.state.get('data_timestamp', join_point.timestamp)
            age_hours = (join_point.timestamp - data_timestamp) / 3600
            return max(0.0, 1.0 - age_hours / req_value)
        
        elif req_type == 'required_metadata':
            metadata = join_point.state.get('metadata', {})
            present_fields = sum(1 for field in req_value if field in metadata)
            return present_fields / len(req_value) if req_value else 1.0
        
        elif req_type == 'min_security_score':
            security_score = join_point.state.get('security_score', 0.0)
            return min(security_score / req_value, 1.0)
        
        return 1.0  # Default to compliant for unknown requirements
    
    def _compute_overall_compliance(self, compliance_scores: Dict[str, float]) -> float:
        """Compute overall policy compliance score"""
        if not compliance_scores:
            return 1.0
        
        # Weight by policy priority
        weighted_scores = []
        total_weight = 0
        
        for policy_id, score in compliance_scores.items():
            policy = next((p for p in self.policies if p.id == policy_id), None)
            if policy:
                weight = policy.priority
                weighted_scores.append(score * weight)
                total_weight += weight
        
        if total_weight > 0:
            return sum(weighted_scores) / total_weight
        else:
            return np.mean(list(compliance_scores.values()))

class PolicyConflictResolutionAdvice(Advice):
    """Advice for detecting and resolving policy conflicts"""
    
    def __init__(self):
        self.detected_conflicts: List[PolicyConflict] = []
    
    def execute(self, join_point: JoinPoint) -> JoinPoint:
        """Detect and resolve policy conflicts"""
        applicable_policies = join_point.state.get('applicable_policies', [])
        
        if len(applicable_policies) < 2:
            return join_point  # No conflicts possible with fewer than 2 policies
        
        # Detect conflicts
        conflicts = self._detect_conflicts(applicable_policies, join_point)
        
        # Resolve conflicts
        resolved_policies = self._resolve_conflicts(conflicts, applicable_policies, join_point)
        
        # Update join point state
        join_point.state['policy_conflicts'] = conflicts
        join_point.state['resolved_policies'] = resolved_policies
        join_point.state['conflict_resolution_applied'] = len(conflicts) > 0
        
        if conflicts:
            logger.warning(f"Policy conflicts detected and resolved: {len(conflicts)} conflicts")
        
        return join_point
    
    def _detect_conflicts(self, policy_ids: List[str], 
                         join_point: JoinPoint) -> List[PolicyConflict]:
        """Detect conflicts between policies"""
        conflicts = []
        
        # Get policy objects (this would need access to policy registry)
        policies = self._get_policies_by_ids(policy_ids)
        
        # Check for requirement conflicts
        for i, policy1 in enumerate(policies):
            for j, policy2 in enumerate(policies[i+1:], i+1):
                conflict = self._check_policy_pair_conflict(policy1, policy2)
                if conflict:
                    conflicts.append(conflict)
        
        return conflicts
    
    def _get_policies_by_ids(self, policy_ids: List[str]) -> List[Policy]:
        """Get policy objects by their IDs (placeholder implementation)"""
        # In a real implementation, this would query a policy registry
        return []
    
    def _check_policy_pair_conflict(self, policy1: Policy, policy2: Policy) -> Optional[PolicyConflict]:
        """Check if two policies conflict"""
        # Check for direct requirement conflicts
        req1 = policy1.requirements
        req2 = policy2.requirements
        
        # Example: conflicting encryption requirements
        if ('encryption_required' in req1 and 'encryption_required' in req2 and
            req1['encryption_required'] != req2['encryption_required']):
            return PolicyConflict(
                id=str(uuid.uuid4()),
                policy_ids=[policy1.id, policy2.id],
                conflict_type='encryption_requirement_conflict',
                severity='HIGH',
                description=f"Conflicting encryption requirements between {policy1.name} and {policy2.name}",
                resolution_strategy='priority_based'
            )
        
        # Example: conflicting participant requirements
        if ('min_participants' in req1 and 'max_participants' in req2 and
            req1['min_participants'] > req2['max_participants']):
            return PolicyConflict(
                id=str(uuid.uuid4()),
                policy_ids=[policy1.id, policy2.id],
                conflict_type='participant_count_conflict',
                severity='MEDIUM',
                description=f"Conflicting participant requirements between {policy1.name} and {policy2.name}",
                resolution_strategy='negotiation'
            )
        
        return None
    
    def _resolve_conflicts(self, conflicts: List[PolicyConflict], 
                          policy_ids: List[str], join_point: JoinPoint) -> List[str]:
        """Resolve policy conflicts and return final policy set"""
        if not conflicts:
            return policy_ids
        
        resolved_policies = set(policy_ids)
        
        for conflict in conflicts:
            if conflict.resolution_strategy == 'priority_based':
                resolved_policies = self._resolve_by_priority(conflict, resolved_policies)
            elif conflict.resolution_strategy == 'negotiation':
                resolved_policies = self._resolve_by_negotiation(conflict, resolved_policies)
            elif conflict.resolution_strategy == 'union':
                resolved_policies = self._resolve_by_union(conflict, resolved_policies)
            
            conflict.resolved = True
        
        return list(resolved_policies)
    
    def _resolve_by_priority(self, conflict: PolicyConflict, 
                           policy_set: Set[str]) -> Set[str]:
        """Resolve conflict by keeping highest priority policy"""
        # Get policies involved in conflict
        conflicting_policies = self._get_policies_by_ids(conflict.policy_ids)
        
        if not conflicting_policies:
            return policy_set
        
        # Keep only the highest priority policy
        highest_priority_policy = max(conflicting_policies, key=lambda p: p.priority)
        
        # Remove other conflicting policies
        for policy in conflicting_policies:
            if policy.id != highest_priority_policy.id:
                policy_set.discard(policy.id)
        
        return policy_set
    
    def _resolve_by_negotiation(self, conflict: PolicyConflict, 
                              policy_set: Set[str]) -> Set[str]:
        """Resolve conflict by finding middle ground"""
        # Simplified negotiation: average conflicting requirements
        # In practice, this would involve more sophisticated negotiation algorithms
        return policy_set
    
    def _resolve_by_union(self, conflict: PolicyConflict, 
                         policy_set: Set[str]) -> Set[str]:
        """Resolve conflict by taking union of requirements"""
        # Take the most restrictive requirement from conflicting policies
        return policy_set

class PolicyEnforcementAdvice(Advice):
    """Advice for enforcing resolved policies"""
    
    def execute(self, join_point: JoinPoint) -> JoinPoint:
        """Enforce resolved policies"""
        resolved_policies = join_point.state.get('resolved_policies', [])
        enforcement_actions = []
        
        for policy_id in resolved_policies:
            actions = self._enforce_policy(policy_id, join_point)
            enforcement_actions.extend(actions)
        
        # Update join point state
        join_point.state['policy_enforcement_actions'] = enforcement_actions
        join_point.state['policies_enforced'] = len(resolved_policies)
        
        if enforcement_actions:
            logger.info(f"Policy enforcement actions applied: {len(enforcement_actions)} actions")
        
        return join_point
    
    def _enforce_policy(self, policy_id: str, join_point: JoinPoint) -> List[str]:
        """Enforce a specific policy"""
        actions = []
        
        # Get policy object (placeholder)
        policy = self._get_policy_by_id(policy_id)
        if not policy:
            return actions
        
        # Apply enforcement based on policy requirements
        for req_type, req_value in policy.requirements.items():
            action = self._apply_enforcement_action(req_type, req_value, join_point)
            if action:
                actions.append(action)
        
        return actions
    
    def _get_policy_by_id(self, policy_id: str) -> Optional[Policy]:
        """Get policy by ID (placeholder implementation)"""
        return None
    
    def _apply_enforcement_action(self, req_type: str, req_value: Any, 
                                join_point: JoinPoint) -> Optional[str]:
        """Apply specific enforcement action"""
        if req_type == 'encryption_required' and req_value:
            if 'sensitive_data' in join_point.state and 'encrypted_data' not in join_point.state:
                # Trigger encryption
                join_point.state['enforce_encryption'] = True
                return 'encryption_enforced'
        
        elif req_type == 'audit_logging_required' and req_value:
            if 'audit_log' not in join_point.state:
                # Initialize audit log
                join_point.state['audit_log'] = []
                return 'audit_logging_enabled'
        
        elif req_type == 'data_minimization_required' and req_value:
            if 'raw_data' in join_point.state:
                # Trigger data minimization
                join_point.state['apply_data_minimization'] = True
                return 'data_minimization_applied'
        
        return None

class InstitutionalPolicyAspect:
    """Main institutional policy aspect that coordinates policy management"""
    
    def __init__(self, policies: List[Policy] = None):
        self.policies = policies or []
        
        # Initialize policy advice components
        self.evaluation_advice = PolicyEvaluationAdvice(self.policies)
        self.conflict_resolution_advice = PolicyConflictResolutionAdvice()
        self.enforcement_advice = PolicyEnforcementAdvice()
        
        self.policy_violations: List[PolicyViolation] = []
        self.policy_conflicts: List[PolicyConflict] = []
    
    def execute(self, join_point: JoinPoint) -> JoinPoint:
        """Execute comprehensive policy management"""
        # Evaluate policy compliance
        join_point = self.evaluation_advice.execute(join_point)
        
        # Detect and resolve conflicts
        join_point = self.conflict_resolution_advice.execute(join_point)
        
        # Enforce resolved policies
        join_point = self.enforcement_advice.execute(join_point)
        
        # Generate policy summary
        self._generate_policy_summary(join_point)
        
        # Log violations and conflicts
        self._log_policy_events(join_point)
        
        return join_point
    
    def add_policy(self, policy: Policy):
        """Add a new policy to the system"""
        self.policies.append(policy)
        self.evaluation_advice.policies.append(policy)
        self.evaluation_advice.policy_index = self.evaluation_advice._build_policy_index()
        logger.info(f"Added policy: {policy.name} ({policy.id})")
    
    def remove_policy(self, policy_id: str):
        """Remove a policy from the system"""
        self.policies = [p for p in self.policies if p.id != policy_id]
        self.evaluation_advice.policies = [p for p in self.evaluation_advice.policies if p.id != policy_id]
        self.evaluation_advice.policy_index = self.evaluation_advice._build_policy_index()
        logger.info(f"Removed policy: {policy_id}")
    
    def _generate_policy_summary(self, join_point: JoinPoint):
        """Generate summary of policy management results"""
        summary = {
            'timestamp': join_point.timestamp,
            'phase': join_point.phase.value,
            'context': join_point.context,
            'total_policies': len(self.policies),
            'applicable_policies': len(join_point.state.get('applicable_policies', [])),
            'violations_detected': len(join_point.state.get('policy_violations', [])),
            'conflicts_detected': len(join_point.state.get('policy_conflicts', [])),
            'overall_compliance': join_point.state.get('overall_policy_compliance', 1.0),
            'enforcement_actions': len(join_point.state.get('policy_enforcement_actions', []))
        }
        
        join_point.state['policy_summary'] = summary
    
    def _log_policy_events(self, join_point: JoinPoint):
        """Log policy violations and conflicts"""
        violations = join_point.state.get('policy_violations', [])
        conflicts = join_point.state.get('policy_conflicts', [])
        
        for violation in violations:
            self.policy_violations.append(violation)
            logger.warning(f"Policy violation: {violation.violation_type} "
                          f"(Policy: {violation.policy_id}, Severity: {violation.severity})")
        
        for conflict in conflicts:
            self.policy_conflicts.append(conflict)
            logger.warning(f"Policy conflict: {conflict.conflict_type} "
                          f"(Policies: {conflict.policy_ids}, Severity: {conflict.severity})")

# Utility functions for creating common policies

def create_data_governance_policy(institution_id: str, **kwargs) -> Policy:
    """Create a standard data governance policy"""
    return Policy(
        id=str(uuid.uuid4()),
        name=f"Data Governance Policy - {institution_id}",
        type=PolicyType.DATA_GOVERNANCE,
        scope=PolicyScope.INSTITUTIONAL,
        institution_id=institution_id,
        priority=kwargs.get('priority', 5),
        conditions={
            'phases': ['data_loading', 'local_training'],
            'data_types': kwargs.get('data_types', ['sensitive', 'personal'])
        },
        requirements={
            'encryption_required': True,
            'required_metadata': ['source', 'classification', 'owner'],
            'max_data_age_hours': kwargs.get('max_data_age_hours', 168)  # 1 week
        },
        penalties={'violation_weight': 0.8},
        metadata={'created_by': 'system', 'category': 'data_governance'},
        created_at=time.time()
    )

def create_privacy_policy(institution_id: str, **kwargs) -> Policy:
    """Create a standard privacy protection policy"""
    return Policy(
        id=str(uuid.uuid4()),
        name=f"Privacy Protection Policy - {institution_id}",
        type=PolicyType.PRIVACY_PROTECTION,
        scope=PolicyScope.INSTITUTIONAL,
        institution_id=institution_id,
        priority=kwargs.get('priority', 8),
        conditions={
            'phases': ['local_training', 'model_update_submission'],
            'data_types': ['personal', 'sensitive']
        },
        requirements={
            'differential_privacy_required': True,
            'min_epsilon': kwargs.get('min_epsilon', 1.0),
            'data_minimization_required': True
        },
        penalties={'violation_weight': 0.9},
        metadata={'created_by': 'system', 'category': 'privacy'},
        created_at=time.time()
    )

def create_security_policy(institution_id: str, **kwargs) -> Policy:
    """Create a standard security policy"""
    return Policy(
        id=str(uuid.uuid4()),
        name=f"Security Policy - {institution_id}",
        type=PolicyType.SECURITY_REQUIREMENT,
        scope=PolicyScope.INSTITUTIONAL,
        institution_id=institution_id,
        priority=kwargs.get('priority', 7),
        conditions={
            'phases': ['model_update_submission', 'aggregation']
        },
        requirements={
            'min_security_score': kwargs.get('min_security_score', 0.8),
            'anomaly_detection_required': True,
            'integrity_verification_required': True
        },
        penalties={'violation_weight': 0.85},
        metadata={'created_by': 'system', 'category': 'security'},
        created_at=time.time()
    )

