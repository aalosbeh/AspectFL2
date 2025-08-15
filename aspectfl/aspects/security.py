"""
Security Aspects for AspectFL
Implements security monitoring, threat detection, and policy enforcement
"""

import time
import numpy as np
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import logging

from ..core.base import Advice, JoinPoint, ExecutionPhase

logger = logging.getLogger(__name__)

@dataclass
class SecurityEvent:
    """Represents a security event or violation"""
    event_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    timestamp: float
    context: str
    details: Dict[str, Any]
    resolved: bool = False

@dataclass
class ThreatAssessment:
    """Comprehensive threat assessment results"""
    overall_threat_level: str
    anomaly_score: float
    integrity_score: float
    privacy_score: float
    policy_compliance_score: float
    detected_threats: List[str]
    recommendations: List[str]

class AnomalyDetectionAdvice(Advice):
    """Advice for detecting anomalous behavior in federated learning"""
    
    def __init__(self, sensitivity: float = 2.0, window_size: int = 10):
        self.sensitivity = sensitivity  # Standard deviations for anomaly threshold
        self.window_size = window_size
        self.historical_updates = []
        self.baseline_stats = None
    
    def execute(self, join_point: JoinPoint) -> JoinPoint:
        """Detect anomalies in model updates or data"""
        anomaly_score = 0.0
        detected_anomalies = []
        
        if join_point.phase == ExecutionPhase.MODEL_UPDATE_SUBMISSION:
            anomaly_score, anomalies = self._detect_model_anomalies(join_point)
            detected_anomalies.extend(anomalies)
        
        elif join_point.phase == ExecutionPhase.DATA_LOADING:
            anomaly_score, anomalies = self._detect_data_anomalies(join_point)
            detected_anomalies.extend(anomalies)
        
        # Update join point state
        join_point.state['anomaly_score'] = anomaly_score
        join_point.state['detected_anomalies'] = detected_anomalies
        join_point.state['is_anomalous'] = anomaly_score > self.sensitivity
        
        if anomaly_score > self.sensitivity:
            logger.warning(f"Anomaly detected: score={anomaly_score:.3f}, "
                          f"anomalies={detected_anomalies}")
        
        return join_point
    
    def _detect_model_anomalies(self, join_point: JoinPoint) -> Tuple[float, List[str]]:
        """Detect anomalies in model updates"""
        model_update = join_point.state.get('model_update')
        if model_update is None:
            return 0.0, []
        
        anomalies = []
        scores = []
        
        # Convert model update to numpy array if needed
        if not isinstance(model_update, np.ndarray):
            try:
                model_update = np.array(model_update)
            except:
                return 0.0, ['invalid_model_format']
        
        # Check for NaN or infinite values
        if np.any(np.isnan(model_update)) or np.any(np.isinf(model_update)):
            anomalies.append('invalid_values')
            scores.append(10.0)  # Very high anomaly score
        
        # Check magnitude anomalies
        magnitude = np.linalg.norm(model_update)
        if self.baseline_stats is not None:
            expected_magnitude = self.baseline_stats.get('mean_magnitude', 1.0)
            magnitude_std = self.baseline_stats.get('std_magnitude', 0.1)
            
            magnitude_z_score = abs(magnitude - expected_magnitude) / max(magnitude_std, 1e-6)
            if magnitude_z_score > self.sensitivity:
                anomalies.append('magnitude_anomaly')
                scores.append(magnitude_z_score)
        
        # Check for gradient explosion
        if magnitude > 100.0:  # Arbitrary large threshold
            anomalies.append('gradient_explosion')
            scores.append(magnitude / 10.0)
        
        # Check for suspicious patterns (e.g., all zeros, all same value)
        if np.all(model_update == 0):
            anomalies.append('zero_update')
            scores.append(5.0)
        elif np.all(model_update == model_update.flat[0]):
            anomalies.append('constant_update')
            scores.append(3.0)
        
        # Update historical data
        self._update_baseline_stats(model_update)
        
        overall_score = max(scores) if scores else 0.0
        return overall_score, anomalies
    
    def _detect_data_anomalies(self, join_point: JoinPoint) -> Tuple[float, List[str]]:
        """Detect anomalies in data"""
        data_batch = join_point.state.get('data_batch')
        if data_batch is None:
            return 0.0, []
        
        anomalies = []
        scores = []
        
        # Check data size anomalies
        batch_size = len(data_batch) if hasattr(data_batch, '__len__') else 0
        if batch_size == 0:
            anomalies.append('empty_batch')
            scores.append(5.0)
        elif batch_size > 10000:  # Arbitrary large threshold
            anomalies.append('oversized_batch')
            scores.append(2.0)
        
        # Check for data quality issues
        if hasattr(data_batch, 'isnull') and data_batch.isnull().any().any():
            anomalies.append('missing_values')
            scores.append(1.0)
        
        overall_score = max(scores) if scores else 0.0
        return overall_score, anomalies
    
    def _update_baseline_stats(self, model_update: np.ndarray):
        """Update baseline statistics for anomaly detection"""
        magnitude = np.linalg.norm(model_update)
        self.historical_updates.append(magnitude)
        
        # Keep only recent updates
        if len(self.historical_updates) > self.window_size:
            self.historical_updates = self.historical_updates[-self.window_size:]
        
        # Update baseline statistics
        if len(self.historical_updates) >= 3:
            self.baseline_stats = {
                'mean_magnitude': np.mean(self.historical_updates),
                'std_magnitude': np.std(self.historical_updates),
                'median_magnitude': np.median(self.historical_updates)
            }

class IntegrityVerificationAdvice(Advice):
    """Advice for verifying data and model integrity"""
    
    def execute(self, join_point: JoinPoint) -> JoinPoint:
        """Verify integrity of data and models"""
        integrity_score = 1.0
        integrity_issues = []
        
        # Verify checksums if available
        if 'expected_checksum' in join_point.state:
            actual_checksum = self._compute_checksum(join_point.state.get('data'))
            expected_checksum = join_point.state['expected_checksum']
            
            if actual_checksum != expected_checksum:
                integrity_issues.append('checksum_mismatch')
                integrity_score *= 0.0
        
        # Verify digital signatures if available
        if 'signature' in join_point.state:
            signature_valid = self._verify_signature(
                join_point.state.get('data'),
                join_point.state['signature'],
                join_point.state.get('public_key')
            )
            
            if not signature_valid:
                integrity_issues.append('invalid_signature')
                integrity_score *= 0.0
        
        # Check for tampering indicators
        tampering_score = self._detect_tampering(join_point.state.get('data'))
        integrity_score *= tampering_score
        
        if tampering_score < 1.0:
            integrity_issues.append('potential_tampering')
        
        # Update join point state
        join_point.state['integrity_score'] = integrity_score
        join_point.state['integrity_issues'] = integrity_issues
        join_point.state['integrity_verified'] = integrity_score > 0.9
        
        if integrity_score < 0.9:
            logger.warning(f"Integrity issues detected: score={integrity_score:.3f}, "
                          f"issues={integrity_issues}")
        
        return join_point
    
    def _compute_checksum(self, data: Any) -> str:
        """Compute SHA-256 checksum of data"""
        if data is None:
            return ""
        
        if isinstance(data, np.ndarray):
            data_bytes = data.tobytes()
        elif isinstance(data, (str, bytes)):
            data_bytes = data.encode('utf-8') if isinstance(data, str) else data
        else:
            data_bytes = str(data).encode('utf-8')
        
        return hashlib.sha256(data_bytes).hexdigest()
    
    def _verify_signature(self, data: Any, signature: str, public_key: str) -> bool:
        """Verify digital signature (simplified implementation)"""
        # In a real implementation, this would use proper cryptographic verification
        # For demonstration, we'll use a simple hash-based verification
        if not all([data, signature, public_key]):
            return False
        
        expected_signature = hashlib.sha256(
            (str(data) + public_key).encode('utf-8')
        ).hexdigest()
        
        return signature == expected_signature
    
    def _detect_tampering(self, data: Any) -> float:
        """Detect potential tampering in data"""
        if data is None:
            return 0.5
        
        # Simple heuristics for tampering detection
        tampering_score = 1.0
        
        if isinstance(data, np.ndarray):
            # Check for unusual patterns that might indicate tampering
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                tampering_score *= 0.5
            
            # Check for suspicious statistical properties
            if data.size > 0:
                mean_val = np.mean(data)
                std_val = np.std(data)
                
                # Extremely high or low variance might indicate tampering
                if std_val < 1e-10 or std_val > 1e10:
                    tampering_score *= 0.7
        
        return tampering_score

class PrivacyPreservationAdvice(Advice):
    """Advice for implementing privacy-preserving mechanisms"""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon  # Privacy budget
        self.delta = delta      # Privacy parameter
        self.noise_scale = None
        self._compute_noise_scale()
    
    def _compute_noise_scale(self):
        """Compute noise scale for differential privacy"""
        # Simplified noise scale computation
        # In practice, this would depend on sensitivity analysis
        self.noise_scale = 2.0 / self.epsilon
    
    def execute(self, join_point: JoinPoint) -> JoinPoint:
        """Apply privacy-preserving mechanisms"""
        privacy_score = 1.0
        privacy_mechanisms = []
        
        # Apply differential privacy to model updates
        if join_point.phase == ExecutionPhase.MODEL_UPDATE_SUBMISSION:
            model_update = join_point.state.get('model_update')
            if model_update is not None:
                noisy_update = self._add_differential_privacy_noise(model_update)
                join_point.state['model_update'] = noisy_update
                privacy_mechanisms.append('differential_privacy')
        
        # Encrypt sensitive data
        if 'sensitive_data' in join_point.state:
            encrypted_data = self._encrypt_data(join_point.state['sensitive_data'])
            join_point.state['encrypted_data'] = encrypted_data
            privacy_mechanisms.append('encryption')
        
        # Apply data minimization
        if 'raw_data' in join_point.state:
            minimized_data = self._minimize_data(join_point.state['raw_data'])
            join_point.state['minimized_data'] = minimized_data
            privacy_mechanisms.append('data_minimization')
        
        # Update join point state
        join_point.state['privacy_score'] = privacy_score
        join_point.state['privacy_mechanisms'] = privacy_mechanisms
        join_point.state['epsilon_used'] = self.epsilon
        
        logger.info(f"Privacy mechanisms applied: {privacy_mechanisms}")
        
        return join_point
    
    def _add_differential_privacy_noise(self, data: np.ndarray) -> np.ndarray:
        """Add Gaussian noise for differential privacy"""
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        noise = np.random.normal(0, self.noise_scale, data.shape)
        return data + noise
    
    def _encrypt_data(self, data: Any) -> str:
        """Encrypt sensitive data"""
        # Generate a key for encryption
        key = Fernet.generate_key()
        cipher_suite = Fernet(key)
        
        # Convert data to bytes
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = str(data).encode('utf-8')
        
        # Encrypt the data
        encrypted_data = cipher_suite.encrypt(data_bytes)
        
        # Return base64 encoded encrypted data
        return base64.b64encode(encrypted_data).decode('utf-8')
    
    def _minimize_data(self, data: Any) -> Any:
        """Apply data minimization principles"""
        # Simple data minimization: remove unnecessary fields
        if isinstance(data, dict):
            # Keep only essential fields
            essential_fields = ['id', 'features', 'label', 'timestamp']
            minimized = {k: v for k, v in data.items() if k in essential_fields}
            return minimized
        
        return data

class PolicyEnforcementAdvice(Advice):
    """Advice for enforcing institutional and regulatory policies"""
    
    def __init__(self, policies: List[Dict[str, Any]] = None):
        self.policies = policies or []
        self.policy_violations = []
    
    def execute(self, join_point: JoinPoint) -> JoinPoint:
        """Enforce policies at join point"""
        violations = []
        compliance_score = 1.0
        
        for policy in self.policies:
            violation = self._check_policy_compliance(policy, join_point)
            if violation:
                violations.append(violation)
                compliance_score *= (1.0 - policy.get('severity_weight', 0.1))
        
        # Update join point state
        join_point.state['policy_violations'] = violations
        join_point.state['policy_compliance_score'] = compliance_score
        join_point.state['policy_compliant'] = compliance_score > 0.8
        
        if violations:
            logger.warning(f"Policy violations detected: {violations}")
        
        return join_point
    
    def _check_policy_compliance(self, policy: Dict[str, Any], 
                                join_point: JoinPoint) -> Optional[Dict[str, Any]]:
        """Check compliance with a specific policy"""
        policy_type = policy.get('type')
        
        if policy_type == 'data_retention':
            return self._check_data_retention_policy(policy, join_point)
        elif policy_type == 'access_control':
            return self._check_access_control_policy(policy, join_point)
        elif policy_type == 'encryption_required':
            return self._check_encryption_policy(policy, join_point)
        elif policy_type == 'audit_logging':
            return self._check_audit_logging_policy(policy, join_point)
        
        return None
    
    def _check_data_retention_policy(self, policy: Dict[str, Any], 
                                   join_point: JoinPoint) -> Optional[Dict[str, Any]]:
        """Check data retention policy compliance"""
        max_retention_days = policy.get('max_retention_days', 365)
        
        data_timestamp = join_point.state.get('data_timestamp')
        if data_timestamp:
            age_days = (time.time() - data_timestamp) / (24 * 3600)
            if age_days > max_retention_days:
                return {
                    'policy_id': policy.get('id'),
                    'violation_type': 'data_retention_exceeded',
                    'details': f'Data age {age_days:.1f} days exceeds limit {max_retention_days}'
                }
        
        return None
    
    def _check_access_control_policy(self, policy: Dict[str, Any], 
                                   join_point: JoinPoint) -> Optional[Dict[str, Any]]:
        """Check access control policy compliance"""
        required_permissions = policy.get('required_permissions', [])
        user_permissions = join_point.state.get('user_permissions', [])
        
        missing_permissions = set(required_permissions) - set(user_permissions)
        if missing_permissions:
            return {
                'policy_id': policy.get('id'),
                'violation_type': 'insufficient_permissions',
                'details': f'Missing permissions: {list(missing_permissions)}'
            }
        
        return None
    
    def _check_encryption_policy(self, policy: Dict[str, Any], 
                               join_point: JoinPoint) -> Optional[Dict[str, Any]]:
        """Check encryption policy compliance"""
        if policy.get('encryption_required', False):
            if 'encrypted_data' not in join_point.state and 'sensitive_data' in join_point.state:
                return {
                    'policy_id': policy.get('id'),
                    'violation_type': 'encryption_required',
                    'details': 'Sensitive data must be encrypted'
                }
        
        return None
    
    def _check_audit_logging_policy(self, policy: Dict[str, Any], 
                                  join_point: JoinPoint) -> Optional[Dict[str, Any]]:
        """Check audit logging policy compliance"""
        if policy.get('audit_required', False):
            if 'audit_log' not in join_point.state:
                return {
                    'policy_id': policy.get('id'),
                    'violation_type': 'audit_logging_required',
                    'details': 'Audit logging is required for this operation'
                }
        
        return None

class SecurityAspect:
    """Main security aspect that coordinates all security checks"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize security advice components
        self.anomaly_detection = AnomalyDetectionAdvice(
            sensitivity=self.config.get('anomaly_sensitivity', 2.0)
        )
        
        self.integrity_verification = IntegrityVerificationAdvice()
        
        self.privacy_preservation = PrivacyPreservationAdvice(
            epsilon=self.config.get('epsilon', 1.0),
            delta=self.config.get('delta', 1e-5)
        )
        
        self.policy_enforcement = PolicyEnforcementAdvice(
            policies=self.config.get('policies', [])
        )
        
        self.security_events = []
    
    def execute(self, join_point: JoinPoint) -> JoinPoint:
        """Execute comprehensive security assessment"""
        # Execute individual security components
        join_point = self.anomaly_detection.execute(join_point)
        join_point = self.integrity_verification.execute(join_point)
        join_point = self.privacy_preservation.execute(join_point)
        join_point = self.policy_enforcement.execute(join_point)
        
        # Compute overall threat assessment
        threat_assessment = self._compute_threat_assessment(join_point)
        
        # Update join point state
        join_point.state['threat_assessment'] = threat_assessment
        join_point.state['security_score'] = self._compute_security_score(threat_assessment)
        
        # Log security events
        self._log_security_events(join_point, threat_assessment)
        
        # Trigger incident response if needed
        if threat_assessment.overall_threat_level in ['HIGH', 'CRITICAL']:
            self._trigger_incident_response(join_point, threat_assessment)
        
        return join_point
    
    def _compute_threat_assessment(self, join_point: JoinPoint) -> ThreatAssessment:
        """Compute comprehensive threat assessment"""
        anomaly_score = join_point.state.get('anomaly_score', 0.0)
        integrity_score = join_point.state.get('integrity_score', 1.0)
        privacy_score = join_point.state.get('privacy_score', 1.0)
        policy_compliance_score = join_point.state.get('policy_compliance_score', 1.0)
        
        # Determine overall threat level
        if anomaly_score > 5.0 or integrity_score < 0.5:
            threat_level = 'CRITICAL'
        elif anomaly_score > 3.0 or integrity_score < 0.7:
            threat_level = 'HIGH'
        elif anomaly_score > 1.0 or policy_compliance_score < 0.8:
            threat_level = 'MEDIUM'
        else:
            threat_level = 'LOW'
        
        # Collect detected threats
        detected_threats = []
        if join_point.state.get('is_anomalous', False):
            detected_threats.extend(join_point.state.get('detected_anomalies', []))
        
        if not join_point.state.get('integrity_verified', True):
            detected_threats.extend(join_point.state.get('integrity_issues', []))
        
        if join_point.state.get('policy_violations'):
            detected_threats.append('policy_violations')
        
        # Generate recommendations
        recommendations = self._generate_security_recommendations(join_point)
        
        return ThreatAssessment(
            overall_threat_level=threat_level,
            anomaly_score=anomaly_score,
            integrity_score=integrity_score,
            privacy_score=privacy_score,
            policy_compliance_score=policy_compliance_score,
            detected_threats=detected_threats,
            recommendations=recommendations
        )
    
    def _compute_security_score(self, threat_assessment: ThreatAssessment) -> float:
        """Compute overall security score"""
        # Weighted combination of security metrics
        weights = {
            'anomaly': 0.3,
            'integrity': 0.3,
            'privacy': 0.2,
            'policy': 0.2
        }
        
        # Normalize anomaly score (lower is better)
        normalized_anomaly = max(0, 1.0 - threat_assessment.anomaly_score / 10.0)
        
        security_score = (
            weights['anomaly'] * normalized_anomaly +
            weights['integrity'] * threat_assessment.integrity_score +
            weights['privacy'] * threat_assessment.privacy_score +
            weights['policy'] * threat_assessment.policy_compliance_score
        )
        
        return max(0.0, min(1.0, security_score))
    
    def _generate_security_recommendations(self, join_point: JoinPoint) -> List[str]:
        """Generate security recommendations based on assessment"""
        recommendations = []
        
        if join_point.state.get('is_anomalous', False):
            recommendations.append("Investigate anomalous behavior and consider quarantine")
        
        if not join_point.state.get('integrity_verified', True):
            recommendations.append("Verify data integrity and re-authenticate sources")
        
        if join_point.state.get('policy_violations'):
            recommendations.append("Address policy violations before proceeding")
        
        if 'encrypted_data' not in join_point.state and 'sensitive_data' in join_point.state:
            recommendations.append("Encrypt sensitive data before transmission")
        
        return recommendations
    
    def _log_security_events(self, join_point: JoinPoint, threat_assessment: ThreatAssessment):
        """Log security events for audit trail"""
        if threat_assessment.detected_threats:
            event = SecurityEvent(
                event_type='threat_detected',
                severity=threat_assessment.overall_threat_level,
                timestamp=join_point.timestamp,
                context=join_point.context,
                details={
                    'threats': threat_assessment.detected_threats,
                    'phase': join_point.phase.value,
                    'security_score': self._compute_security_score(threat_assessment)
                }
            )
            self.security_events.append(event)
    
    def _trigger_incident_response(self, join_point: JoinPoint, threat_assessment: ThreatAssessment):
        """Trigger incident response for high-severity threats"""
        logger.critical(f"SECURITY INCIDENT: {threat_assessment.overall_threat_level} threat detected "
                       f"in {join_point.phase.value} at {join_point.context}")
        logger.critical(f"Threats: {threat_assessment.detected_threats}")
        logger.critical(f"Recommendations: {threat_assessment.recommendations}")
        
        # In a real system, this would trigger automated response actions
        # such as quarantine, notification, or system shutdown

