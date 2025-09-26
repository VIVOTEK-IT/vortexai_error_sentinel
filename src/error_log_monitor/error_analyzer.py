"""
Error Analyzer with OpenAI integration for impact analysis.
"""

import logging
import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any
from openai import OpenAI

from error_log_monitor.config import SystemConfig, ModelType, get_model_for_complexity, calculate_estimated_cost
from error_log_monitor.rag_engine import RAGEngine, MergedIssue
from error_log_monitor.llm_helper import llm_chat

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels based on system impact and data damage."""

    LEVEL_1 = "LEVEL_1"  # Single API/service broken, system working, no data damage, no action needed
    LEVEL_2 = "LEVEL_2"  # Part of services broken, no data damage, no action needed
    LEVEL_3 = "LEVEL_3"  # Data was damaged, human remedial action needed


@dataclass
class ErrorScope:
    """Analysis of error scope and impact."""

    affected_services: List[str]
    affected_users: Optional[int] = None
    technical_impact: str = "UNKNOWN"
    estimated_downtime: Optional[str] = None


@dataclass
class RemediationPlan:
    """Remediation plan for error."""

    human_action_needed: bool = False
    action_guidelines: List[str] = None
    damaged_modules: List[str] = None
    root_cause: Optional[str] = None
    urgency: str = "MEDIUM"

    def __post_init__(self):
        if self.action_guidelines is None:
            self.action_guidelines = []
        if self.damaged_modules is None:
            self.damaged_modules = []


@dataclass
class ErrorAnalysis:
    """Complete error analysis result."""

    error_id: str
    error_message: str
    severity: ErrorSeverity
    service: str
    confidence_score: float
    estimated_cost: float
    analysis_time: float
    scope: ErrorScope
    remediation_plan: RemediationPlan
    data_damage_assessment: Optional[Dict[str, Any]] = None
    analysis_model: str = "unknown"


class ErrorAnalyzer:
    """Analyzes errors using OpenAI API with intelligent model selection and RAG."""

    def __init__(self, config: SystemConfig, rag_engine: RAGEngine):
        """Initialize error analyzer."""
        self.config = config
        self.rag_engine = rag_engine
        self.openai_client = OpenAI(api_key=config.openai_api_key)

    def analyze_merged_issues(self, merged_issues: List[MergedIssue]) -> List[ErrorAnalysis]:
        """
        Analyze merged issues for impact and remediation needs.

        Args:
            merged_issues: List of merged similar issues

        Returns:
            List of error analysis results
        """
        analyses = []

        for issue in merged_issues:
            try:
                analysis = self.analyze_issue(issue)
                analyses.append(analysis)
            except Exception as e:
                logger.error(f"Error analyzing issue {issue.issue_id}: {e}", exc_info=True)
                # Create fallback analysis
                fallback_analysis = self._create_fallback_analysis(issue)
                analyses.append(fallback_analysis)

        return analyses

    def analyze_issue(self, merged_issue: MergedIssue) -> ErrorAnalysis:
        """
        Analyze a single merged issue.

        Args:
            merged_issue: Merged issue to analyze

        Returns:
            Error analysis result
        """
        start_time = time.time()

        try:
            # Calculate complexity score
            complexity_score = self._calculate_complexity_score(merged_issue)

            # Select appropriate model
            model_type = get_model_for_complexity(complexity_score, self.config)
            model_name = self.config.models[model_type].name

            # Retrieve context using RAG
            context = self.rag_engine.retrieve_context_for_analysis(merged_issue.representative_log)

            # Perform analysis
            analysis_result = self._perform_llm_analysis(merged_issue, context, model_name, complexity_score)

            # Calculate costs
            estimated_cost = self._calculate_analysis_cost(analysis_result, model_type)

            analysis_time = time.time() - start_time

            # Create analysis result
            analysis = ErrorAnalysis(
                error_id=merged_issue.issue_id,
                error_message=merged_issue.representative_log.error_message,
                severity=analysis_result['severity'],
                service=merged_issue.representative_log.service or "unknown",
                confidence_score=analysis_result.get('confidence_score', 0.8),
                estimated_cost=estimated_cost,
                analysis_time=analysis_time,
                scope=analysis_result['scope'],
                remediation_plan=analysis_result['remediation_plan'],
                data_damage_assessment=analysis_result.get('data_damage_assessment'),
                analysis_model=model_name,
            )

            logger.info(f"Analysis completed for issue {merged_issue.issue_id} using {model_name}")
            return analysis

        except Exception as e:
            logger.error(f"Error analyzing issue {merged_issue.issue_id}: {e}")
            return self._create_fallback_analysis(merged_issue)

    def _calculate_complexity_score(self, merged_issue: MergedIssue) -> float:
        """Calculate complexity score for model selection."""
        score = 0.0

        # Base score from occurrence count
        if merged_issue.occurrence_count > 10:
            score += 0.1
        elif merged_issue.occurrence_count > 5:
            score += 0.05

        # Score from affected services
        if len(merged_issue.affected_services) > 3:
            score += 0.1
        elif len(merged_issue.affected_services) > 1:
            score += 0.05

        # Score from error message complexity
        error_msg = merged_issue.representative_log.error_message
        if len(error_msg) > 1000:
            score += 0.5
        if any(keyword in error_msg.lower() for keyword in ['database', 'connection', 'timeout', 'critical']):
            score += 0.2

        # Score from traceback presence
        if merged_issue.representative_log.traceback and len(merged_issue.representative_log.traceback) > 1000:
            score += 0.4

        return min(score, 1.0)

    def _perform_llm_analysis(
        self, merged_issue: MergedIssue, context: str, model_name: str, complexity_score: float
    ) -> Dict[str, Any]:
        """Perform LLM analysis with round-trip conversation until valid JSON response."""
        try:
            # Create analysis prompt with RDS data access
            prompt = self._create_analysis_prompt(merged_issue, context, complexity_score)

            # Initialize conversation
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt},
            ]

            max_attempts = 5
            for attempt in range(max_attempts):
                logger.info(f"LLM analysis attempt {attempt + 1}/{max_attempts}")
                response_message = llm_chat(messages, model_name, self.config)

                # If no tool calls, try to parse the final analysis
                if response_message.content:
                    try:
                        analysis_result = self._parse_analysis_response(response_message.content[0].text)

                        # Check if we got a valid analysis (not fallback)
                        if self._is_valid_analysis(analysis_result):
                            logger.info(f"Valid analysis obtained on attempt {attempt + 1}")
                            return analysis_result
                        else:
                            logger.warning(f"Invalid analysis on attempt {attempt + 1}, retrying...")

                    except Exception as parse_error:
                        logger.warning(f"Parse error on attempt {attempt + 1}: {parse_error}")

                # If not the last attempt, ask for correction
                if attempt < max_attempts - 1:
                    correction_prompt = (
                        "Please provide your final analysis in the exact JSON format specified in the system prompt. "
                        "Make sure all required fields are present and the JSON is valid. "
                        "If you need to use tools, do so first, then provide the final analysis."
                    )
                    messages.append({"role": "user", "content": correction_prompt})

            # If all attempts failed, return default analysis
            logger.error("All LLM analysis attempts failed, using default analysis")
            return self._get_default_analysis()

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}", exc_info=True)
            return self._get_default_analysis()

    def _is_valid_analysis(self, analysis_result: Dict[str, Any]) -> bool:
        """Check if the analysis result is valid (not a fallback)."""
        try:
            # Check if it's a fallback analysis
            if analysis_result.get('confidence_score', 0) <= 0.1:
                return False

            # Check if technical impact indicates failure
            technical_impact = analysis_result.get('scope', {}).technical_impact
            if 'Analysis failed' in technical_impact or 'manual review required' in technical_impact:
                return False

            # Check if root cause indicates failure
            root_cause = analysis_result.get('remediation_plan', {}).root_cause
            if 'Analysis parsing failed' in root_cause or 'Analysis failed' in root_cause:
                return False

            # Check if we have meaningful affected services
            affected_services = analysis_result.get('scope', {}).affected_services
            if not affected_services and analysis_result.get('severity') != ErrorSeverity.LEVEL_1:
                return False

            return True

        except Exception as e:
            logger.warning(f"Error validating analysis result: {e}", exc_info=True)
            return False

    def _create_analysis_prompt(self, merged_issue: MergedIssue, context: str, complexity_score: float) -> str:
        """Create analysis prompt for OpenAI with RDS data access."""
        prompt = f"""Analyze this error for Vortexai service:

Error: {merged_issue.representative_log.error_message}
Type: {merged_issue.representative_log.error_type or 'Unknown'}
Service: {merged_issue.representative_log.service or 'Unknown'}
Count: {merged_issue.occurrence_count}
Affected: {', '.join(merged_issue.affected_services)}
Time: {merged_issue.time_span}
"""
        return prompt

    def _get_system_prompt(self) -> str:
        """Get system prompt for error analysis."""
        return """You are an expert error analysis error log for Vortexai service.
You are given an error log and you need to analyze the error log and determine if data is damaged, service is broken, and if human action is needed.
Analyze errors and respond with valid JSON only. No explanations, no markdown, just JSON.

Severity levels:
- LEVEL_1: Single service broken, no data damage, no action needed
- LEVEL_2: Multiple services broken, no data damage, no action needed
- LEVEL_3: Data was damaged, human action needed

You have following API for RDS access to check if the error log is about to insert, update or delete data of RDS database:
    get_camera_info, check_object_trace_partition_exists,get_partition_info,get_object_trace
    
Always respond with valid JSON in this exact format:
{
    "severity": "LEVEL_1|LEVEL_2|LEVEL_3",
    "scope": {
        "affected_services": ["service1"],
        "affected_users": "UNKNOWN",
        "technical_impact": "brief description",
        "estimated_downtime": "UNKNOWN"
    },
    "remediation_plan": {
        "human_action_needed": false,
        "action_guidelines": ["guideline1"],
        "damaged_modules": ["module1"],
        "root_cause": "brief description",
        "urgency": "MEDIUM"
    },
    "data_damage_assessment": {
        "data_damaged": false,
        "damaged_tables": [],
        "damage_description": "No damage detected",
        "affected_records": "UNKNOWN"
    },
    "confidence_score": 0.8
}"""

    def _parse_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """Parse OpenAI response into structured analysis."""
        try:
            # Clean the response text
            response_text = response_text.strip()

            # Try to find JSON in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx == -1 or end_idx == 0:
                logger.warning(f"No JSON found in response. Response: {response_text[:200]}...")
                return self._get_default_analysis()

            json_str = response_text[start_idx:end_idx]

            # Try to parse JSON
            try:
                result = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error: {e}. Attempting to fix JSON...", exc_info=True)
                # Try to fix common JSON issues
                json_str = self._fix_json_string(json_str)
                result = json.loads(json_str)

            # Convert to proper data types
            analysis = {
                'severity': ErrorSeverity(result.get('severity', 'LEVEL_2')),
                'scope': ErrorScope(
                    affected_services=result.get('scope', {}).get('affected_services', []),
                    affected_users=result.get('scope', {}).get('affected_users'),
                    technical_impact=result.get('scope', {}).get('technical_impact', 'UNKNOWN'),
                    estimated_downtime=result.get('scope', {}).get('estimated_downtime'),
                ),
                'remediation_plan': RemediationPlan(
                    human_action_needed=result.get('remediation_plan', {}).get('human_action_needed', False),
                    action_guidelines=result.get('remediation_plan', {}).get('action_guidelines', []),
                    damaged_modules=result.get('remediation_plan', {}).get('damaged_modules', []),
                    root_cause=result.get('remediation_plan', {}).get('root_cause'),
                    urgency=result.get('remediation_plan', {}).get('urgency', 'MEDIUM'),
                ),
                'data_damage_assessment': result.get(
                    'data_damage_assessment',
                    {
                        'data_damaged': False,
                        'damaged_tables': [],
                        'damage_description': 'No data damage detected',
                        'affected_records': 'UNKNOWN',
                    },
                ),
                'confidence_score': result.get('confidence_score', 0.8),
            }

            return analysis

        except Exception as e:
            logger.error(f"Failed to parse analysis response: {e}", exc_info=True)
            logger.error(f"Response text: {response_text[:500]}...")
            return self._get_default_analysis()

    def _fix_json_string(self, json_str: str) -> str:
        """Attempt to fix common JSON formatting issues."""
        # Remove any text before the first {
        start_idx = json_str.find('{')
        if start_idx > 0:
            json_str = json_str[start_idx:]

        # Remove any text after the last }
        end_idx = json_str.rfind('}')
        if end_idx < len(json_str) - 1:
            json_str = json_str[: end_idx + 1]

        # Fix common issues
        json_str = json_str.replace('true_or_false', 'false')
        json_str = json_str.replace('0.0_to_1.0', '0.8')

        return json_str

    def _get_default_analysis(self) -> Dict[str, Any]:
        """Get default analysis when parsing fails."""
        return {
            'severity': ErrorSeverity.LEVEL_1,
            'scope': ErrorScope(
                affected_services=[],
                technical_impact="Analysis failed - manual review required",
            ),
            'remediation_plan': RemediationPlan(
                human_action_needed=True,
                action_guidelines=["Manual review required due to analysis failure"],
                damaged_modules=[],
                root_cause="Analysis parsing failed",
                urgency="MEDIUM",
            ),
            'confidence_score': 0.1,
        }

    def _create_fallback_analysis(self, merged_issue: MergedIssue) -> ErrorAnalysis:
        """Create fallback analysis when main analysis fails."""
        return ErrorAnalysis(
            error_id=merged_issue.issue_id,
            error_message=merged_issue.representative_log.error_message,
            severity=ErrorSeverity.LEVEL_2,
            service=merged_issue.representative_log.service or "unknown",
            confidence_score=0.1,
            estimated_cost=0.0,
            analysis_time=0.0,
            scope=ErrorScope(
                affected_services=merged_issue.affected_services,
                technical_impact="Analysis failed - manual review required",
            ),
            remediation_plan=RemediationPlan(
                human_action_needed=True,
                action_guidelines=["Manual review required due to analysis failure"],
                damaged_modules=[],
                root_cause="Analysis failed",
                urgency="MEDIUM",
            ),
            analysis_model="fallback",
        )

    def _calculate_analysis_cost(self, analysis_result: Dict[str, Any], model_type: ModelType) -> float:
        """Calculate estimated cost for analysis."""
        # Estimate tokens based on analysis complexity
        input_tokens = 1000  # Base input tokens
        output_tokens = 500  # Base output tokens

        # Adjust based on complexity
        if analysis_result.get('confidence_score', 0.8) > 0.9:
            input_tokens += 200
            output_tokens += 100

        return calculate_estimated_cost(input_tokens, output_tokens, model_type, self.config)
