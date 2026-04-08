"""
Rule Engine for Fraud Detection
================================

Modular rule-based fraud detection engine with configurable rules.
Designed to work alongside ML models for hybrid predictions.

Architecture:
    Rule (Base Class)
        ├─ Rule implementations
        └─ Rule Engine (orchestrator)

Usage:
    from aml_fraud_detector.rule_engine import RuleEngine, RuleConfig
    
    engine = RuleEngine()
    rule_scores = engine.evaluate_transaction(transaction_data)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Callable
from abc import ABC, abstractmethod
import pandas as pd
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════
# RULE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RuleConfig:
    """Configuration for rule thresholds and weights."""
    
    # Amount thresholds
    HIGH_AMOUNT_THRESHOLD: float = 50000.0
    VERY_HIGH_AMOUNT_THRESHOLD: float = 100000.0
    
    # Crypto flags
    CRYPTO_CURRENCIES: List[str] = field(default_factory=lambda: [
        'Bitcoin', 'Ethereum', 'Cryptocurrency'
    ])
    
    # Transfer flags
    CROSS_BORDER_PAIRS: List[tuple] = field(default_factory=lambda: [
        ('US Dollar', 'Ruble'),
        ('US Dollar', 'Yuan'),
        ('Euro', 'Bitcoin'),
    ])
    
    # Account and bank flags
    SUSPICIOUS_BANKS: List[int] = field(default_factory=lambda: [999, 888])
    
    # Rule weights (how much each rule affects final score)
    RULE_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        'high_amount_crypto': 0.25,
        'very_high_amount': 0.20,
        'new_account_large_transfer': 0.25,
        'cross_border_risky': 0.15,
        'suspicious_bank': 0.15,
    })
    
    # Risk score adjustments from rules (0-100 scale)
    RISK_BOOST_HIGH_AMOUNT_CRYPTO: int = 30
    RISK_BOOST_VERY_HIGH_AMOUNT: int = 25
    RISK_BOOST_NEW_ACCOUNT_LARGE_TRANSFER: int = 35
    RISK_BOOST_CROSS_BORDER_RISKY: int = 20
    RISK_BOOST_SUSPICIOUS_BANK: int = 15
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'high_amount_threshold': self.HIGH_AMOUNT_THRESHOLD,
            'very_high_amount_threshold': self.VERY_HIGH_AMOUNT_THRESHOLD,
            'crypto_currencies': self.CRYPTO_CURRENCIES,
            'suspicious_banks': self.SUSPICIOUS_BANKS,
        }


# ═══════════════════════════════════════════════════════════════════════════
# RULE BASE CLASS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RuleResult:
    """Result from evaluating a single rule."""
    rule_name: str
    triggered: bool
    risk_boost: int = 0  # 0-100 scale boost
    description: str = ""
    confidence: float = 1.0  # 0-1, how confident we are in this rule


class Rule(ABC):
    """Abstract base class for fraud detection rules."""
    
    def __init__(self, config: RuleConfig = None):
        """
        Initialize rule with configuration.
        
        Args:
            config (RuleConfig): Configuration for thresholds
        """
        self.config = config or RuleConfig()
        self.name = self.__class__.__name__
    
    @abstractmethod
    def evaluate(self, transaction: Dict[str, Any]) -> RuleResult:
        """
        Evaluate transaction against this rule.
        
        Args:
            transaction (dict): Transaction data
        
        Returns:
            RuleResult: Whether rule triggered and risk boost amount
        """
        pass


# ═══════════════════════════════════════════════════════════════════════════
# RULE IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════

class HighAmountCryptoRule(Rule):
    """
    Flag high amount + cryptocurrency combination.
    
    Logic: High amount (>$50K) + crypto currency = increased risk
    
    Examples:
    - $60,000 to Bitcoin address: HIGH RISK
    - $30,000 to Bitcoin address: OK (below threshold)
    """
    
    def evaluate(self, transaction: Dict[str, Any]) -> RuleResult:
        """Evaluate high amount + crypto combination."""
        
        amount_received = float(transaction.get('amount_received', 0))
        receiving_currency = str(transaction.get('receiving_currency', '')).lower()
        
        # Check conditions
        is_high_amount = amount_received > self.config.HIGH_AMOUNT_THRESHOLD
        is_crypto = any(
            crypto.lower() in receiving_currency 
            for crypto in self.config.CRYPTO_CURRENCIES
        )
        
        # Trigger if both conditions met
        triggered = is_high_amount and is_crypto
        
        return RuleResult(
            rule_name="HIGH_AMOUNT_CRYPTO",
            triggered=triggered,
            risk_boost=self.config.RISK_BOOST_HIGH_AMOUNT_CRYPTO if triggered else 0,
            description=f"High amount (${amount_received:,.0f}) + {receiving_currency}",
            confidence=0.90 if triggered else 1.0
        )


class VeryHighAmountRule(Rule):
    """
    Flag very high amounts regardless of currency.
    
    Logic: Amount > $100K = significant risk
    
    Examples:
    - $150,000 transfer: HIGH RISK
    - $75,000 transfer: OK
    """
    
    def evaluate(self, transaction: Dict[str, Any]) -> RuleResult:
        """Evaluate if amount is very high."""
        
        amount_received = float(transaction.get('amount_received', 0))
        
        triggered = amount_received > self.config.VERY_HIGH_AMOUNT_THRESHOLD
        
        return RuleResult(
            rule_name="VERY_HIGH_AMOUNT",
            triggered=triggered,
            risk_boost=self.config.RISK_BOOST_VERY_HIGH_AMOUNT if triggered else 0,
            description=f"Very high amount: ${amount_received:,.0f}",
            confidence=0.95 if triggered else 1.0
        )


class NewAccountLargeTransferRule(Rule):
    """
    Flag new accounts (low ID) making large transfers.
    
    Logic: Bank ID < 100 (new) + large transfer = suspicion
    
    Examples:
    - New bank (ID=50) transferring $80K: HIGH RISK
    - Established bank (ID=500) transferring $80K: OK
    """
    
    def evaluate(self, transaction: Dict[str, Any]) -> RuleResult:
        """Evaluate if new account making large transfer."""
        
        from_bank = int(transaction.get('from_bank', 999))
        to_bank = int(transaction.get('to_bank', 999))
        amount_received = float(transaction.get('amount_received', 0))
        
        # New account: low bank ID
        is_new_account = from_bank < 100
        
        # Large transfer: > $40K
        is_large_transfer = amount_received > 40000
        
        triggered = is_new_account and is_large_transfer
        
        return RuleResult(
            rule_name="NEW_ACCOUNT_LARGE_TRANSFER",
            triggered=triggered,
            risk_boost=self.config.RISK_BOOST_NEW_ACCOUNT_LARGE_TRANSFER if triggered else 0,
            description=f"New account (Bank {from_bank}) transferring ${amount_received:,.0f}",
            confidence=0.85 if triggered else 1.0
        )


class CrossBorderRiskyRule(Rule):
    """
    Flag risky cross-border currency pairs.
    
    Logic: Transfer between risky currency pairs = higher risk
    
    Examples:
    - USD to Ruble: Increased risk
    - USD to EUR: Normal
    """
    
    def evaluate(self, transaction: Dict[str, Any]) -> RuleResult:
        """Evaluate if high-risk currency pair."""
        
        receiving_currency = str(transaction.get('receiving_currency', ''))
        payment_currency = str(transaction.get('payment_currency', ''))
        
        # Check if this is a risky pair
        currency_pair = (receiving_currency, payment_currency)
        is_risky_pair = currency_pair in self.config.CROSS_BORDER_PAIRS
        
        triggered = is_risky_pair
        
        return RuleResult(
            rule_name="CROSS_BORDER_RISKY",
            triggered=triggered,
            risk_boost=self.config.RISK_BOOST_CROSS_BORDER_RISKY if triggered else 0,
            description=f"Risky currency pair: {receiving_currency} → {payment_currency}",
            confidence=0.80 if triggered else 1.0
        )


class SuspiciousBankRule(Rule):
    """
    Flag suspicious bank accounts.
    
    Logic: Known suspicious bank IDs = flag
    
    Examples:
    - Bank 999 (known problematic): FLAGGED
    - Bank 100 (normal): OK
    """
    
    def evaluate(self, transaction: Dict[str, Any]) -> RuleResult:
        """Evaluate if suspicious bank involved."""
        
        from_bank = int(transaction.get('from_bank', 0))
        to_bank = int(transaction.get('to_bank', 0))
        
        is_suspicious = (
            from_bank in self.config.SUSPICIOUS_BANKS or 
            to_bank in self.config.SUSPICIOUS_BANKS
        )
        
        triggered = is_suspicious
        
        suspicious_id = from_bank if from_bank in self.config.SUSPICIOUS_BANKS else to_bank
        
        return RuleResult(
            rule_name="SUSPICIOUS_BANK",
            triggered=triggered,
            risk_boost=self.config.RISK_BOOST_SUSPICIOUS_BANK if triggered else 0,
            description=f"Suspicious bank ID involved: {suspicious_id}",
            confidence=0.90 if triggered else 1.0
        )


# ═══════════════════════════════════════════════════════════════════════════
# RULE ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class RuleEngine:
    """
    Orchestrates all fraud detection rules.
    
    Evaluates transaction against all rules and provides combined risk score.
    
    Usage:
        engine = RuleEngine()
        results = engine.evaluate_transaction(transaction_data)
        print(results)
    """
    
    def __init__(self, config: RuleConfig = None):
        """
        Initialize rule engine.
        
        Args:
            config (RuleConfig): Configuration for all rules
        """
        self.config = config or RuleConfig()
        self.rules = self._initialize_rules()
    
    def _initialize_rules(self) -> List[Rule]:
        """Initialize all available rules."""
        return [
            HighAmountCryptoRule(self.config),
            VeryHighAmountRule(self.config),
            NewAccountLargeTransferRule(self.config),
            CrossBorderRiskyRule(self.config),
            SuspiciousBankRule(self.config),
        ]
    
    def add_custom_rule(self, rule: Rule) -> None:
        """
        Add custom rule to engine.
        
        Args:
            rule (Rule): Custom rule instance
        """
        self.rules.append(rule)
    
    def evaluate_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate transaction against all rules.
        
        Args:
            transaction (dict): Transaction data with keys like:
                - from_bank: int
                - to_bank: int
                - amount_received: float
                - amount_paid: float
                - receiving_currency: str
                - payment_currency: str
                - payment_format: str
        
        Returns:
            dict: {
                'rule_results': List[RuleResult],
                'triggered_count': int,
                'triggered_rules': List[str],
                'total_risk_boost': int (0-100),
                'summary': str
            }
        """
        
        rule_results = []
        triggered_count = 0
        triggered_rules = []
        total_risk_boost = 0
        
        # Evaluate each rule
        for rule in self.rules:
            result = rule.evaluate(transaction)
            rule_results.append(result)
            
            if result.triggered:
                triggered_count += 1
                triggered_rules.append(result.rule_name)
                total_risk_boost += result.risk_boost
        
        # Cap total boost at reasonable maximum
        total_risk_boost = min(total_risk_boost, 100)
        
        return {
            'rule_results': rule_results,
            'triggered_count': triggered_count,
            'triggered_rules': triggered_rules,
            'total_risk_boost': total_risk_boost,
            'summary': self._create_summary(rule_results, triggered_count)
        }
    
    def _create_summary(self, results: List[RuleResult], count: int) -> str:
        """Create human-readable summary of rule evaluation."""
        if count == 0:
            return "No rules triggered - transaction appears normal"
        
        triggered = [r for r in results if r.triggered]
        descriptions = [r.description for r in triggered]
        
        return f"{count} rule(s) triggered: {'; '.join(descriptions)}"
    
    def get_rule_report(self, transaction: Dict[str, Any]) -> str:
        """
        Get detailed text report of rule evaluation.
        
        Args:
            transaction (dict): Transaction data
        
        Returns:
            str: Formatted report
        """
        results = self.evaluate_transaction(transaction)
        
        report = "RULE ENGINE EVALUATION REPORT\n"
        report += "=" * 50 + "\n\n"
        
        report += f"Total Rules: {len(self.rules)}\n"
        report += f"Rules Triggered: {results['triggered_count']}\n"
        report += f"Total Risk Boost: +{results['total_risk_boost']} points\n\n"
        
        report += "Rule Results:\n"
        report += "-" * 50 + "\n"
        
        for i, rule_result in enumerate(results['rule_results'], 1):
            status = "✅ TRIGGERED" if rule_result.triggered else "⏭️ SKIPPED"
            report += f"{i}. {rule_result.rule_name} - {status}\n"
            report += f"   {rule_result.description}\n"
            if rule_result.triggered:
                report += f"   Risk Boost: +{rule_result.risk_boost}\n"
            report += "\n"
        
        report += "-" * 50 + "\n"
        report += f"\nSummary: {results['summary']}\n"
        
        return report


# ═══════════════════════════════════════════════════════════════════════════
# RISK SCORE BOOSTING - Simple post-processing for ML predictions
# ═══════════════════════════════════════════════════════════════════════════

class RiskBooster:
    """
    Post-processing risk boosting for ML predictions.
    
    Applies simple rule-based adjustments to ML-predicted risk scores
    WITHOUT modifying the underlying model or preprocessing.
    
    This is designed to handle low fraud probabilities from imbalanced
    training data while keeping the solution modular and transparent.
    
    Rules:
    1. High amount payment (>100K) → +30 boost
    2. Currency mismatch (received != paid) → +15 boost
    3. Suspicious bank IDs (>900) → +20 boost
    4. Amount mismatch (SCALED by severity):
       - Moderate (80-90% paid): +10 boost
       - Large (50-80% paid): +25 boost
       - Extreme (<50% paid): +50 boost
    
    Usage:
        booster = RiskBooster()
        result = booster.boost_score(
            ml_score=5,
            transaction={
                'amount_paid': 150000,
                'receiving_currency': 'USD',
                'payment_currency': 'EUR',
                'from_bank': 120,
                'to_bank': 130
            }
        )
        print(result['boosted_score'])  # 5 + 30 + 15 + 20 = 70
    """
    
    def __init__(self, config: RuleConfig = None):
        """Initialize booster with configuration."""
        self.config = config or RuleConfig()
    
    def boost_score(
        self,
        ml_score: int,
        transaction: Dict[str, Any],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Boost ML risk score with rule-based adjustments.
        
        Args:
            ml_score: ML-based risk score (0-100)
            transaction: Transaction data dict with keys:
                - amount_paid (float & required)
                - amount_received (float)
                - receiving_currency (str)
                - payment_currency (str)
                - from_bank (int)
                - to_bank (int)
            verbose: If True, log which rules were triggered
        
        Returns:
            Dict with:
            - 'original_score': Input ML score
            - 'boosted_score': Final score after rules (0-100)
            - 'boost_amount': Total points added
            - 'triggered_rules': List of rule names
            - 'details': List of (boost, reason) tuples
        """
        boosted_score = float(ml_score)
        triggered_rules = []
        details = []
        
        # Rule 1: Large amount payment (>100K)
        try:
            amount_paid = float(transaction.get('amount_paid', 0))
            if amount_paid > 100000:
                boost = 30
                boosted_score += boost
                triggered_rules.append('high_amount_payment')
                details.append((boost, f'Amount paid {amount_paid:,.0f} exceeds 100K'))
        except (ValueError, TypeError):
            pass
        
        # Rule 2: Currency mismatch
        try:
            receiving_currency = str(transaction.get('receiving_currency', '')).upper().strip()
            payment_currency = str(transaction.get('payment_currency', '')).upper().strip()
            
            if receiving_currency and payment_currency:
                if receiving_currency != payment_currency:
                    boost = 15
                    boosted_score += boost
                    triggered_rules.append('currency_mismatch')
                    details.append((boost, f'Currency mismatch: {receiving_currency} vs {payment_currency}'))
        except (ValueError, TypeError, AttributeError):
            pass
        
        # Rule 3: Suspicious bank IDs (>900)
        try:
            from_bank = int(transaction.get('from_bank', 0))
            to_bank = int(transaction.get('to_bank', 0))
            
            if from_bank > 900 or to_bank > 900:
                boost = 20
                boosted_score += boost
                triggered_rules.append('suspicious_bank_id')
                details.append((boost, f'Suspicious bank ID (from={from_bank}, to={to_bank})'))
        except (ValueError, TypeError):
            pass
        
        # Rule 4: Amount mismatch (paid significantly less than received) - SCALED by severity
        try:
            amount_received = float(transaction.get('amount_received', 0))
            amount_paid = float(transaction.get('amount_paid', 0))
            
            if amount_received > 0:
                ratio = (amount_paid / amount_received * 100) if amount_received > 0 else 0
                
                # Scale boost based on mismatch severity
                if amount_paid < amount_received * 0.50:  # Paid <50% = EXTREME
                    boost = 50
                    boosted_score += boost
                    triggered_rules.append('amount_mismatch_extreme')
                    details.append((boost, f'EXTREME amount mismatch: paid only {ratio:.1f}% of received'))
                elif amount_paid < amount_received * 0.80:  # Paid 50-80% = LARGE
                    boost = 25
                    boosted_score += boost
                    triggered_rules.append('amount_mismatch_large')
                    details.append((boost, f'Large amount mismatch: paid only {ratio:.1f}% of received'))
                elif amount_paid < amount_received * 0.90:  # Paid 80-90% = MODERATE
                    boost = 10
                    boosted_score += boost
                    triggered_rules.append('amount_mismatch')
                    details.append((boost, f'Amount mismatch: paid only {ratio:.1f}% of received'))
        except (ValueError, TypeError):
            pass
        
        # Cap final score at 100
        final_score = int(min(max(boosted_score, 0), 100))
        
        # Log if rules were triggered
        if verbose and triggered_rules:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(
                f"Risk boosting applied: {ml_score} → {final_score} "
                f"(+{final_score - ml_score}) | Rules: {', '.join(triggered_rules)}"
            )
        
        return {
            'original_score': int(ml_score),
            'boosted_score': final_score,
            'boost_amount': final_score - int(ml_score),
            'triggered_rules': triggered_rules,
            'details': details  # List of (boost, reason) tuples
        }
    
    def boost_batch(
        self,
        ml_scores: List[int],
        transactions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Boost multiple ML predictions at once.
        
        Args:
            ml_scores: List of ML scores
            transactions: List of transaction dicts
        
        Returns:
            List of boost result dicts
        """
        return [
            self.boost_score(score, transaction, verbose=False)
            for score, transaction in zip(ml_scores, transactions)
        ]


def integrate_rule_boosting_with_prediction(
    prediction_result: Dict[str, Any],
    transaction_data: Dict[str, Any],
    apply_boost: bool = True
) -> Dict[str, Any]:
    """
    Integrate rule-based boosting with threshold-based predictions.
    
    This is the main function to use with PredictionPipeline.predict_single_with_threshold()
    
    Args:
        prediction_result: Output from predict_single_with_threshold()
            Expected keys: 'score', 'level', 'prediction'
        
        transaction_data: Transaction details
        
        apply_boost: If False, return original result unchanged
    
    Returns:
        Enhanced prediction with boosted score and rule details
    
    Example:
        >>> # Get ML prediction
        >>> ml_result = pipeline.predict_single_with_threshold(X, threshold=0.05)
        >>> 
        >>> # Apply rule-based boosting
        >>> enhanced = integrate_rule_boosting_with_prediction(
        ...     ml_result,
        ...     transaction_data={
        ...         'amount_paid': 150000,
        ...         'receiving_currency': 'USD',
        ...         'payment_currency': 'EUR',
        ...         'from_bank': 120,
        ...         'to_bank': 130
        ...     }
        ... )
        >>> 
        >>> # Results will include:
        >>> # 'original_score': 5 (ML score)
        >>> # 'score': 70 (after boosting)
        >>> # 'boost_amount': 65
        >>> # 'triggered_rules': ['high_amount_payment', 'currency_mismatch', 'suspicious_bank_id']
    """
    if not apply_boost:
        return prediction_result
    
    try:
        booster = RiskBooster()
        original_score = prediction_result.get('score', 0)
        
        # Apply rule-based boosting
        boost_result = booster.boost_score(
            ml_score=original_score,
            transaction=transaction_data,
            verbose=True
        )
        
        # Determine new risk level based on boosted score
        boosted_score = boost_result['boosted_score']
        if boosted_score < 30:
            risk_level = 'Low'
        elif boosted_score < 70:
            risk_level = 'Medium'
        else:
            risk_level = 'High'
        
        # Recalculate probability based on boosted score to match the risk level
        # If score is 50/100, probability should be 0.5 (not the original ML probability)
        recalculated_probability = min(boosted_score / 100.0, 1.0)
        
        # Return enhanced prediction
        return {
            **prediction_result,
            'original_score': boost_result['original_score'],
            'score': boost_result['boosted_score'],
            'boost_amount': boost_result['boost_amount'],
            'triggered_rules': boost_result['triggered_rules'],
            'level': risk_level,
            'rule_details': boost_result['details'],
            'probability': recalculated_probability,  # Updated probability to match boosted score
            'boost_applied': len(boost_result['triggered_rules']) > 0
        }
    
    except Exception as e:
        import logging
        logging.error(f"Error applying rule boosting: {e}")
        return prediction_result

