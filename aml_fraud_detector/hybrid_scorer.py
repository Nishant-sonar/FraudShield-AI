"""
Hybrid Fraud Detection Scorer
=============================

Combines ML-based predictions with rule-based detection for comprehensive
fraud scoring. Provides weighted hybrid scoring combining both approaches.

Architecture:
    ML Probability → Score (0-100)
    Rule Scores → Risk Boost (0-100)
    Combine with weights → Final Hybrid Score

Usage:
    from aml_fraud_detector.hybrid_scorer import HybridScorer
    
    scorer = HybridScorer()
    hybrid_score = scorer.calculate_hybrid_score(
        ml_probability=0.35,
        transaction=transaction_data
    )
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
from aml_fraud_detector.rule_engine import RuleEngine, RuleConfig
from aml_fraud_detector.risk_scoring import calculate_risk_score, RiskScore


# ═══════════════════════════════════════════════════════════════════════════
# HYBRID SCORER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HybridScorerConfig:
    """Configuration for hybrid scoring system."""
    
    # Weights for combining ML and rules (must sum to 1.0)
    ML_WEIGHT: float = 0.65          # ML predictions: 65%
    RULES_WEIGHT: float = 0.35       # Rule-based: 35%
    
    # Verification: weights should sum to 1.0
    def __post_init__(self):
        total_weight = self.ML_WEIGHT + self.RULES_WEIGHT
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(
                f"Weights must sum to 1.0, got {total_weight}. "
                f"ML_WEIGHT ({self.ML_WEIGHT}) + RULES_WEIGHT ({self.RULES_WEIGHT})"
            )
    
    # Rule boost application method
    # Options: 'additive' (simple add) or 'multiplicative' (multiply)
    BOOST_METHOD: str = 'multiplicative'  
    
    # If using multiplicative, this scales the boost effect
    BOOST_SCALING_FACTOR: float = 1.5  # 1.5 means rules can boost by up to 50%
    
    # Confidence weighting
    WEIGHT_BY_RULE_CONFIDENCE: bool = True  # Weight rules by their confidence


@dataclass
class HybridScoringResult:
    """Result from hybrid fraud scoring."""
    
    hybrid_score: int                    # Final score 0-100
    hybrid_level: str                    # Low/Medium/High Risk
    ml_score: int                        # ML-only score
    rule_risk_boost: int                 # Boost from rules
    ml_component: float                  # ML contribution to hybrid
    rules_component: float               # Rules contribution to hybrid
    triggered_rules: list[str]           # Which rules triggered
    description: str                     # Human-readable explanation
    confidence: float                    # Confidence in this assessment
    
    def __str__(self) -> str:
        return (
            f"HybridScore(score={self.hybrid_score}, "
            f"level={self.hybrid_level}, "
            f"ml={self.ml_score}+{self.rule_risk_boost})"
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'hybrid_score': self.hybrid_score,
            'hybrid_level': self.hybrid_level,
            'ml_score': self.ml_score,
            'rule_risk_boost': self.rule_risk_boost,
            'ml_component': round(self.ml_component, 2),
            'rules_component': round(self.rules_component, 2),
            'triggered_rules': self.triggered_rules,
            'description': self.description,
            'confidence': round(self.confidence, 2),
        }


# ═══════════════════════════════════════════════════════════════════════════
# HYBRID SCORER
# ═══════════════════════════════════════════════════════════════════════════

class HybridScorer:
    """
    Combines ML predictions with rule-based detection.
    
    Approach:
        1. ML model generates fraud probability (primary signal)
        2. Rule engine evaluates transaction (secondary signal)
        3. Combine both with weighted average
        4. Return single unified risk score
    
    Example:
        scorer = HybridScorer()
        result = scorer.calculate_hybrid_score(0.35, transaction_data)
        print(result.hybrid_score)  # 0-100
        print(result.hybrid_level)   # "Low/Medium/High Risk"
    """
    
    def __init__(
        self,
        config: HybridScorerConfig = None,
        rule_config: RuleConfig = None
    ):
        """
        Initialize hybrid scorer.
        
        Args:
            config (HybridScorerConfig): Hybrid scoring configuration
            rule_config (RuleConfig): Rule engine configuration
        """
        self.config = config or HybridScorerConfig()
        self.rule_engine = RuleEngine(rule_config or RuleConfig())
    
    def calculate_hybrid_score(
        self,
        ml_probability: float,
        transaction: Dict[str, Any]
    ) -> HybridScoringResult:
        """
        Calculate hybrid fraud score combining ML and rules.
        
        Args:
            ml_probability (float): ML model fraud probability (0.0-1.0)
            transaction (dict): Transaction data for rule evaluation
        
        Returns:
            HybridScoringResult: Detailed hybrid scoring result
        
        Example:
            result = scorer.calculate_hybrid_score(
                ml_probability=0.35,
                transaction={
                    'from_bank': 50,
                    'to_bank': 200,
                    'amount_received': 60000,
                    'receiving_currency': 'Bitcoin',
                    'payment_format': 'Wire'
                }
            )
            
            print(f"Score: {result.hybrid_score}/100")
            print(f"Level: {result.hybrid_level}")
        """
        
        # 1. Convert ML probability to 0-100 score
        ml_score = int(ml_probability * 100)
        
        # 2. Evaluate rules
        rule_evaluation = self.rule_engine.evaluate_transaction(transaction)
        rule_risk_boost = rule_evaluation['total_risk_boost']
        triggered_rules = rule_evaluation['triggered_rules']
        
        # 3. Combine scores
        hybrid_score, ml_component, rules_component = self._combine_scores(
            ml_score=ml_score,
            rule_boost=rule_risk_boost,
            rule_results=rule_evaluation['rule_results']
        )
        
        # 4. Get risk level
        risk_score = calculate_risk_score(hybrid_score / 100.0)
        
        # 5. Calculate confidence
        confidence = self._calculate_confidence(
            ml_probability=ml_probability,
            rule_results=rule_evaluation['rule_results'],
            triggered_count=rule_evaluation['triggered_count']
        )
        
        # 6. Create description
        description = self._create_description(
            ml_score=ml_score,
            rule_boost=rule_risk_boost,
            triggered_rules=triggered_rules,
            hybrid_score=hybrid_score
        )
        
        return HybridScoringResult(
            hybrid_score=hybrid_score,
            hybrid_level=risk_score.level,
            ml_score=ml_score,
            rule_risk_boost=rule_risk_boost,
            ml_component=ml_component,
            rules_component=rules_component,
            triggered_rules=triggered_rules,
            description=description,
            confidence=confidence
        )
    
    def _combine_scores(
        self,
        ml_score: int,
        rule_boost: int,
        rule_results: list
    ) -> Tuple[int, float, float]:
        """
        Combine ML score and rule boost using configured method.
        
        Returns:
            Tuple of (hybrid_score, ml_component, rules_component)
        """
        
        if self.config.BOOST_METHOD == 'additive':
            # Simple addition with weights
            hybrid = int(
                ml_score * self.config.ML_WEIGHT +
                rule_boost * self.config.RULES_WEIGHT
            )
            ml_component = ml_score * self.config.ML_WEIGHT
            rules_component = rule_boost * self.config.RULES_WEIGHT
        
        elif self.config.BOOST_METHOD == 'multiplicative':
            # ML score is base, rules amplify
            if self.config.WEIGHT_BY_RULE_CONFIDENCE:
                # Weight boost by average rule confidence
                avg_confidence = sum(
                    r.confidence for r in rule_results
                ) / len(rule_results) if rule_results else 1.0
                effective_boost = rule_boost * avg_confidence
            else:
                effective_boost = rule_boost
            
            # Apply boost with scaling factor
            boost_multiplier = 1.0 + (effective_boost / 100.0) * self.config.BOOST_SCALING_FACTOR
            ml_amplified = ml_score * boost_multiplier
            
            # Combine with weights
            hybrid = int(
                ml_amplified * self.config.ML_WEIGHT +
                rule_boost * self.config.RULES_WEIGHT
            )
            ml_component = ml_amplified * self.config.ML_WEIGHT
            rules_component = rule_boost * self.config.RULES_WEIGHT
        
        else:
            raise ValueError(f"Unknown boost method: {self.config.BOOST_METHOD}")
        
        # Cap at 0-100
        hybrid = max(0, min(100, hybrid))
        
        return hybrid, ml_component, rules_component
    
    def _calculate_confidence(
        self,
        ml_probability: float,
        rule_results: list,
        triggered_count: int
    ) -> float:
        """
        Calculate confidence in the hybrid assessment.
        
        Higher confidence when:
        - ML model is very certain (prob near 0 or 1)
        - Rules strongly agree with ML
        - Multiple rules agree
        """
        
        # ML confidence: high if probability is extreme (0-0.2 or 0.8-1.0)
        ml_confidence = 1.0 - abs(ml_probability - 0.5) * 2.0  # 0.0-1.0
        
        # Rules confidence: average of triggered rule confidences + count bonus
        if rule_results:
            avg_rule_confidence = sum(
                r.confidence for r in rule_results
            ) / len(rule_results)
            # Boost confidence if multiple rules agree
            rules_confidence = avg_rule_confidence * (1.0 + triggered_count * 0.1)
            rules_confidence = min(1.0, rules_confidence)
        else:
            rules_confidence = 0.5  # Neutral if no rules
        
        # Combined confidence (weighted average)
        combined_confidence = (
            ml_confidence * self.config.ML_WEIGHT +
            rules_confidence * self.config.RULES_WEIGHT
        )
        
        return min(1.0, max(0.0, combined_confidence))
    
    def _create_description(
        self,
        ml_score: int,
        rule_boost: int,
        triggered_rules: list,
        hybrid_score: int
    ) -> str:
        """Create human-readable description of hybrid score."""
        
        ml_part = f"ML model: {ml_score}/100"
        
        if triggered_rules:
            rule_part = f"({len(triggered_rules)} rule(s) triggered: +{rule_boost})"
        else:
            rule_part = "(no rules triggered)"
        
        if hybrid_score < 30:
            risk_assessment = "Transaction appears LEGITIMATE"
        elif hybrid_score < 70:
            risk_assessment = "Transaction requires REVIEW"
        else:
            risk_assessment = "Transaction is SUSPICIOUS"
        
        return f"{ml_part}, Rules {rule_part} → {hybrid_score}/100 ({risk_assessment})"
    
    def get_detailed_report(
        self,
        ml_probability: float,
        transaction: Dict[str, Any]
    ) -> str:
        """
        Get detailed text report of hybrid scoring.
        
        Args:
            ml_probability (float): ML fraud probability
            transaction (dict): Transaction data
        
        Returns:
            str: Formatted detailed report
        """
        
        result = self.calculate_hybrid_score(ml_probability, transaction)
        rule_eval = self.rule_engine.evaluate_transaction(transaction)
        
        report = "HYBRID FRAUD SCORING REPORT\n"
        report += "=" * 60 + "\n\n"
        
        # ML Score Section
        report += "1. ML MODEL PREDICTION\n"
        report += "-" * 60 + "\n"
        report += f"   Fraud Probability: {ml_probability:.2%}\n"
        report += f"   ML Score: {result.ml_score}/100\n"
        report += f"   Confidence: {result.ml_component:.1f}\n\n"
        
        # Rule Evaluation Section
        report += "2. RULE-BASED EVALUATION\n"
        report += "-" * 60 + "\n"
        report += f"   Rules Triggered: {len(triggered_rules := result.triggered_rules)}/{len(self.rule_engine.rules)}\n"
        
        if triggered_rules:
            for rule_name in triggered_rules:
                report += f"   • {rule_name}\n"
        else:
            report += "   • No rules triggered\n"
        
        report += f"   Risk Boost: +{result.rule_risk_boost}\n"
        report += f"   Confidence: {result.rules_component:.1f}\n\n"
        
        # Hybrid Score Section
        report += "3. HYBRID SCORE (COMBINED)\n"
        report += "-" * 60 + "\n"
        report += f"   Hybrid Score: {result.hybrid_score}/100\n"
        report += f"   Risk Level: {result.hybrid_level}\n"
        report += f"   Overall Confidence: {result.confidence:.1%}\n"
        report += f"   Recommendation: {self._get_recommendation(result.hybrid_score)}\n\n"
        
        # Summary
        report += "4. ASSESSMENT SUMMARY\n"
        report += "-" * 60 + "\n"
        report += f"   {result.description}\n"
        
        return report
    
    def _get_recommendation(self, hybrid_score: int) -> str:
        """Get action recommendation based on score."""
        if hybrid_score < 30:
            return "✅ Auto-Approve"
        elif hybrid_score < 70:
            return "⚠️ Manual Review Required"
        else:
            return "🚨 Block & Investigate"
    
    def batch_score(
        self,
        ml_probabilities: list,
        transactions: list
    ) -> list[HybridScoringResult]:
        """
        Score multiple transactions.
        
        Args:
            ml_probabilities (list): List of ML probabilities
            transactions (list): List of transaction data dicts
        
        Returns:
            list: List of HybridScoringResult objects
        """
        
        if len(ml_probabilities) != len(transactions):
            raise ValueError(
                f"Mismatched lengths: {len(ml_probabilities)} probabilities "
                f"vs {len(transactions)} transactions"
            )
        
        return [
            self.calculate_hybrid_score(prob, tx)
            for prob, tx in zip(ml_probabilities, transactions)
        ]
