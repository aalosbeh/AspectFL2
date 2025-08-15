"""
FAIR Compliance Aspects for AspectFL
Implements aspects for monitoring and enforcing FAIR principles
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import requests
import json
import logging

from ..core.base import Advice, JoinPoint, ExecutionPhase

logger = logging.getLogger(__name__)

@dataclass
class FAIRMetrics:
    """Metrics for FAIR compliance assessment"""
    findability_score: float = 0.0
    accessibility_score: float = 0.0
    interoperability_score: float = 0.0
    reusability_score: float = 0.0
    overall_score: float = 0.0
    timestamp: float = 0.0
    
    def compute_overall_score(self, weights: Dict[str, float] = None):
        """Compute weighted overall FAIR score"""
        if weights is None:
            weights = {'F': 0.25, 'A': 0.25, 'I': 0.25, 'R': 0.25}
        
        self.overall_score = (
            weights['F'] * self.findability_score +
            weights['A'] * self.accessibility_score +
            weights['I'] * self.interoperability_score +
            weights['R'] * self.reusability_score
        )
        self.timestamp = time.time()

class FindabilityAdvice(Advice):
    """Advice for checking and improving findability"""
    
    def __init__(self, required_metadata: List[str] = None):
        self.required_metadata = required_metadata or [
            'title', 'description', 'creator', 'identifier', 
            'keywords', 'creation_date', 'format', 'size'
        ]
    
    def execute(self, join_point: JoinPoint) -> JoinPoint:
        """Check and improve findability at join point"""
        metadata = join_point.state.get('metadata', {})
        
        # Check metadata completeness
        present_fields = 0
        total_fields = len(self.required_metadata)
        quality_scores = []
        
        for field in self.required_metadata:
            if field in metadata and metadata[field] is not None:
                present_fields += 1
                # Assess quality of metadata field
                quality = self._assess_metadata_quality(field, metadata[field])
                quality_scores.append(quality)
            else:
                quality_scores.append(0.0)
                # Try to auto-generate missing metadata
                auto_value = self._auto_generate_metadata(field, join_point)
                if auto_value:
                    metadata[field] = auto_value
                    present_fields += 1
                    quality_scores.append(0.7)  # Auto-generated gets lower quality
        
        # Compute findability score
        completeness = present_fields / total_fields
        avg_quality = np.mean(quality_scores) if quality_scores else 0.0
        findability_score = 0.6 * completeness + 0.4 * avg_quality
        
        # Update join point state
        join_point.state['metadata'] = metadata
        join_point.state['findability_score'] = findability_score
        join_point.state['metadata_completeness'] = completeness
        
        logger.info(f"Findability assessment: {findability_score:.3f} "
                   f"(completeness: {completeness:.3f}, quality: {avg_quality:.3f})")
        
        return join_point
    
    def _assess_metadata_quality(self, field: str, value: Any) -> float:
        """Assess the quality of a metadata field"""
        if value is None or value == "":
            return 0.0
        
        quality = 1.0
        
        # Field-specific quality checks
        if field == 'description':
            # Longer descriptions are generally better
            desc_length = len(str(value))
            if desc_length < 10:
                quality *= 0.3
            elif desc_length < 50:
                quality *= 0.7
        
        elif field == 'keywords':
            # More keywords (up to a point) are better
            if isinstance(value, list):
                num_keywords = len(value)
            else:
                num_keywords = len(str(value).split(','))
            
            if num_keywords == 0:
                quality = 0.0
            elif num_keywords < 3:
                quality *= 0.5
            elif num_keywords > 10:
                quality *= 0.8
        
        elif field == 'identifier':
            # Check if identifier follows standard format (e.g., DOI, UUID)
            identifier_str = str(value)
            if 'doi:' in identifier_str.lower() or 'uuid:' in identifier_str.lower():
                quality = 1.0
            elif len(identifier_str) > 8:
                quality = 0.8
            else:
                quality = 0.5
        
        return min(quality, 1.0)
    
    def _auto_generate_metadata(self, field: str, join_point: JoinPoint) -> Optional[str]:
        """Auto-generate missing metadata fields"""
        if field == 'identifier':
            return f"aspectfl:{join_point.id}"
        elif field == 'creation_date':
            return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(join_point.timestamp))
        elif field == 'creator':
            return f"AspectFL-{join_point.context}"
        elif field == 'format':
            if join_point.phase == ExecutionPhase.DATA_LOADING:
                return "application/json"
            elif join_point.phase in [ExecutionPhase.LOCAL_TRAINING, ExecutionPhase.AGGREGATION]:
                return "application/octet-stream"
        
        return None

class AccessibilityAdvice(Advice):
    """Advice for checking and improving accessibility"""
    
    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout
    
    def execute(self, join_point: JoinPoint) -> JoinPoint:
        """Check and improve accessibility at join point"""
        endpoints = join_point.state.get('endpoints', [])
        protocols = join_point.state.get('access_protocols', [])
        
        accessibility_scores = []
        
        # Check endpoint accessibility
        for endpoint in endpoints:
            score = self._check_endpoint_accessibility(endpoint)
            accessibility_scores.append(score)
        
        # Check protocol compliance
        for protocol in protocols:
            score = self._check_protocol_compliance(protocol)
            accessibility_scores.append(score)
        
        # Check authentication mechanisms
        auth_score = self._check_authentication(join_point.state.get('authentication', {}))
        accessibility_scores.append(auth_score)
        
        # Compute overall accessibility score
        accessibility_score = np.mean(accessibility_scores) if accessibility_scores else 0.0
        
        # Update join point state
        join_point.state['accessibility_score'] = accessibility_score
        join_point.state['endpoint_status'] = {
            ep: self._check_endpoint_accessibility(ep) for ep in endpoints
        }
        
        logger.info(f"Accessibility assessment: {accessibility_score:.3f}")
        
        return join_point
    
    def _check_endpoint_accessibility(self, endpoint: str) -> float:
        """Check if an endpoint is accessible"""
        try:
            if endpoint.startswith('http'):
                response = requests.head(endpoint, timeout=self.timeout)
                if response.status_code == 200:
                    return 1.0
                elif response.status_code < 500:
                    return 0.7  # Client error but server is responding
                else:
                    return 0.3  # Server error
            else:
                # For non-HTTP endpoints, assume accessible if properly formatted
                return 0.8
        except requests.RequestException:
            return 0.0
        except Exception:
            return 0.0
    
    def _check_protocol_compliance(self, protocol: str) -> float:
        """Check protocol compliance"""
        standard_protocols = ['HTTP', 'HTTPS', 'FTP', 'SFTP', 'REST', 'GraphQL']
        
        if protocol.upper() in standard_protocols:
            return 1.0
        elif protocol.upper().startswith('HTTP'):
            return 0.8
        else:
            return 0.5
    
    def _check_authentication(self, auth_config: Dict[str, Any]) -> float:
        """Check authentication mechanism quality"""
        if not auth_config:
            return 0.5  # No auth specified
        
        auth_type = auth_config.get('type', '').lower()
        
        if auth_type in ['oauth2', 'jwt', 'saml']:
            return 1.0
        elif auth_type in ['basic', 'digest']:
            return 0.7
        elif auth_type == 'api_key':
            return 0.8
        else:
            return 0.3

class InteroperabilityAdvice(Advice):
    """Advice for checking and improving interoperability"""
    
    def __init__(self):
        self.standard_formats = {
            'data': ['JSON', 'XML', 'CSV', 'HDF5', 'Parquet'],
            'model': ['ONNX', 'TensorFlow', 'PyTorch', 'Pickle'],
            'metadata': ['JSON-LD', 'RDF', 'Dublin Core', 'DataCite']
        }
        
        self.standard_vocabularies = [
            'Dublin Core', 'FOAF', 'SKOS', 'DCAT', 'PROV-O'
        ]
    
    def execute(self, join_point: JoinPoint) -> JoinPoint:
        """Check and improve interoperability at join point"""
        formats = join_point.state.get('formats', {})
        vocabularies = join_point.state.get('vocabularies', [])
        schemas = join_point.state.get('schemas', [])
        
        interoperability_scores = []
        
        # Check format standardization
        format_score = self._assess_format_standards(formats)
        interoperability_scores.append(format_score)
        
        # Check vocabulary usage
        vocab_score = self._assess_vocabulary_standards(vocabularies)
        interoperability_scores.append(vocab_score)
        
        # Check schema compliance
        schema_score = self._assess_schema_compliance(schemas)
        interoperability_scores.append(schema_score)
        
        # Check API compatibility
        api_score = self._assess_api_compatibility(join_point.state.get('api_spec', {}))
        interoperability_scores.append(api_score)
        
        # Compute overall interoperability score
        interoperability_score = np.mean(interoperability_scores)
        
        # Update join point state
        join_point.state['interoperability_score'] = interoperability_score
        join_point.state['format_compliance'] = format_score
        join_point.state['vocabulary_compliance'] = vocab_score
        
        logger.info(f"Interoperability assessment: {interoperability_score:.3f}")
        
        return join_point
    
    def _assess_format_standards(self, formats: Dict[str, str]) -> float:
        """Assess compliance with standard formats"""
        if not formats:
            return 0.5
        
        scores = []
        for category, format_name in formats.items():
            if category in self.standard_formats:
                if format_name in self.standard_formats[category]:
                    scores.append(1.0)
                else:
                    scores.append(0.3)
            else:
                scores.append(0.5)
        
        return np.mean(scores) if scores else 0.0
    
    def _assess_vocabulary_standards(self, vocabularies: List[str]) -> float:
        """Assess usage of standard vocabularies"""
        if not vocabularies:
            return 0.3
        
        standard_count = sum(1 for vocab in vocabularies if vocab in self.standard_vocabularies)
        return min(standard_count / len(vocabularies), 1.0)
    
    def _assess_schema_compliance(self, schemas: List[str]) -> float:
        """Assess schema compliance"""
        if not schemas:
            return 0.5
        
        # Simple heuristic: longer schema names suggest more detailed schemas
        avg_length = np.mean([len(schema) for schema in schemas])
        return min(avg_length / 50.0, 1.0)
    
    def _assess_api_compatibility(self, api_spec: Dict[str, Any]) -> float:
        """Assess API compatibility with standards"""
        if not api_spec:
            return 0.5
        
        score = 0.0
        
        # Check for OpenAPI/Swagger specification
        if 'openapi' in api_spec or 'swagger' in api_spec:
            score += 0.4
        
        # Check for standard HTTP methods
        methods = api_spec.get('methods', [])
        standard_methods = ['GET', 'POST', 'PUT', 'DELETE']
        if any(method in standard_methods for method in methods):
            score += 0.3
        
        # Check for content negotiation
        if 'content_types' in api_spec:
            score += 0.3
        
        return min(score, 1.0)

class ReusabilityAdvice(Advice):
    """Advice for checking and improving reusability"""
    
    def execute(self, join_point: JoinPoint) -> JoinPoint:
        """Check and improve reusability at join point"""
        license_info = join_point.state.get('license', {})
        documentation = join_point.state.get('documentation', {})
        provenance = join_point.state.get('provenance', {})
        
        reusability_scores = []
        
        # Check license clarity
        license_score = self._assess_license_clarity(license_info)
        reusability_scores.append(license_score)
        
        # Check documentation completeness
        doc_score = self._assess_documentation_completeness(documentation)
        reusability_scores.append(doc_score)
        
        # Check provenance information
        prov_score = self._assess_provenance_completeness(provenance)
        reusability_scores.append(prov_score)
        
        # Check usage examples
        example_score = self._assess_usage_examples(join_point.state.get('examples', []))
        reusability_scores.append(example_score)
        
        # Compute overall reusability score
        reusability_score = np.mean(reusability_scores)
        
        # Update join point state
        join_point.state['reusability_score'] = reusability_score
        join_point.state['license_clarity'] = license_score
        join_point.state['documentation_completeness'] = doc_score
        
        logger.info(f"Reusability assessment: {reusability_score:.3f}")
        
        return join_point
    
    def _assess_license_clarity(self, license_info: Dict[str, Any]) -> float:
        """Assess license clarity and permissiveness"""
        if not license_info:
            return 0.2
        
        license_type = license_info.get('type', '').lower()
        
        # Standard open licenses get high scores
        open_licenses = ['mit', 'apache', 'bsd', 'gpl', 'cc-by', 'cc0']
        if any(ol in license_type for ol in open_licenses):
            return 1.0
        
        # Proprietary but clear licenses get medium scores
        if 'proprietary' in license_type and 'terms' in license_info:
            return 0.6
        
        # Unclear or restrictive licenses get low scores
        return 0.3
    
    def _assess_documentation_completeness(self, documentation: Dict[str, Any]) -> float:
        """Assess documentation completeness"""
        if not documentation:
            return 0.1
        
        required_sections = ['description', 'usage', 'parameters', 'examples', 'contact']
        present_sections = sum(1 for section in required_sections if section in documentation)
        
        completeness = present_sections / len(required_sections)
        
        # Bonus for detailed documentation
        total_length = sum(len(str(doc)) for doc in documentation.values())
        detail_bonus = min(total_length / 1000.0, 0.2)
        
        return min(completeness + detail_bonus, 1.0)
    
    def _assess_provenance_completeness(self, provenance: Dict[str, Any]) -> float:
        """Assess provenance information completeness"""
        if not provenance:
            return 0.3
        
        required_fields = ['source', 'creation_method', 'dependencies', 'version']
        present_fields = sum(1 for field in required_fields if field in provenance)
        
        return present_fields / len(required_fields)
    
    def _assess_usage_examples(self, examples: List[Dict[str, Any]]) -> float:
        """Assess quality and quantity of usage examples"""
        if not examples:
            return 0.2
        
        # More examples are better (up to a point)
        quantity_score = min(len(examples) / 3.0, 1.0)
        
        # Assess example quality
        quality_scores = []
        for example in examples:
            quality = 0.0
            if 'code' in example:
                quality += 0.4
            if 'description' in example:
                quality += 0.3
            if 'expected_output' in example:
                quality += 0.3
            quality_scores.append(quality)
        
        avg_quality = np.mean(quality_scores) if quality_scores else 0.0
        
        return 0.6 * quantity_score + 0.4 * avg_quality

class FAIRComplianceAspect:
    """Main FAIR compliance aspect that coordinates all FAIR checks"""
    
    def __init__(self, weights: Dict[str, float] = None, threshold: float = 0.7):
        self.weights = weights or {'F': 0.25, 'A': 0.25, 'I': 0.25, 'R': 0.25}
        self.threshold = threshold
        
        # Initialize individual advice components
        self.findability_advice = FindabilityAdvice()
        self.accessibility_advice = AccessibilityAdvice()
        self.interoperability_advice = InteroperabilityAdvice()
        self.reusability_advice = ReusabilityAdvice()
    
    def execute(self, join_point: JoinPoint) -> JoinPoint:
        """Execute comprehensive FAIR compliance check"""
        # Execute individual FAIR components
        join_point = self.findability_advice.execute(join_point)
        join_point = self.accessibility_advice.execute(join_point)
        join_point = self.interoperability_advice.execute(join_point)
        join_point = self.reusability_advice.execute(join_point)
        
        # Compute overall FAIR metrics
        metrics = FAIRMetrics(
            findability_score=join_point.state.get('findability_score', 0.0),
            accessibility_score=join_point.state.get('accessibility_score', 0.0),
            interoperability_score=join_point.state.get('interoperability_score', 0.0),
            reusability_score=join_point.state.get('reusability_score', 0.0)
        )
        
        metrics.compute_overall_score(self.weights)
        
        # Update join point state
        join_point.state['fair_metrics'] = metrics
        join_point.state['fair_compliance_score'] = metrics.overall_score
        
        # Trigger improvement actions if below threshold
        if metrics.overall_score < self.threshold:
            self._trigger_improvement_actions(join_point, metrics)
        
        logger.info(f"FAIR compliance: {metrics.overall_score:.3f} "
                   f"(F:{metrics.findability_score:.2f}, "
                   f"A:{metrics.accessibility_score:.2f}, "
                   f"I:{metrics.interoperability_score:.2f}, "
                   f"R:{metrics.reusability_score:.2f})")
        
        return join_point
    
    def _trigger_improvement_actions(self, join_point: JoinPoint, metrics: FAIRMetrics):
        """Trigger actions to improve FAIR compliance"""
        improvements = []
        
        if metrics.findability_score < self.threshold:
            improvements.append("Improve metadata completeness and quality")
        
        if metrics.accessibility_score < self.threshold:
            improvements.append("Ensure endpoints are accessible and protocols are standard")
        
        if metrics.interoperability_score < self.threshold:
            improvements.append("Use standard formats and vocabularies")
        
        if metrics.reusability_score < self.threshold:
            improvements.append("Provide clear license and comprehensive documentation")
        
        join_point.state['fair_improvement_actions'] = improvements
        logger.warning(f"FAIR compliance below threshold. Suggested improvements: {improvements}")

