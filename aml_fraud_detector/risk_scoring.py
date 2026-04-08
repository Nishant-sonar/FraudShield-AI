"""
Risk Scoring System
===================

Converts fraud probabilities into interpretable risk scores and levels.
This module provides reusable functions to transform model predictions
into actionable risk assessments.

Usage:
    from aml_fraud_detector.risk_scoring import RiskScore, calculate_risk_score
    
    # Single prediction
    fraud_prob = 0.15  # 15% fraud probability
    risk = calculate_risk_score(fraud_prob)
    print(risk)  # RiskScore(score=15, level='Low Risk', probability=0.15)
    
    # Multiple predictions
    probabilities = [0.05, 0.35, 0.75, 0.95]
    for prob in probabilities:
        risk = calculate_risk_score(prob)
        print(f"Probability: {prob:.2%} → Score: {risk.score} ({risk.level})")
"""

from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd
from enum import Enum


class RiskLevel(Enum):
    """Risk level classification."""
    LOW = "Low Risk"
    MEDIUM = "Medium Risk"
    HIGH = "High Risk"


@dataclass
class RiskScore:
    """
    Structured risk assessment output.
    
    Attributes:
        score (int): Risk score from 0 to 100
        level (str): Risk level ('Low Risk', 'Medium Risk', 'High Risk')
        probability (float): Original fraud probability (0-1)
        description (str): Human-readable explanation
        
    Example:
        >>> risk = RiskScore(
        ...     score=35,
        ...     level='Medium Risk',
        ...     probability=0.35,
        ...     description='Transaction flagged for moderate fraud indicators'
        ... )
        >>> print(risk)
        RiskScore(score=35, level=Medium Risk, probability=0.35)
    """
    score: int
    level: str
    probability: float
    description: str = ""
    
    def __str__(self) -> str:
        return f"RiskScore(score={self.score}, level={self.level}, probability={self.probability:.4f})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def to_dict(self) -> dict:
        """Convert to dictionary (useful for JSON serialization)."""
        return {
            'score': self.score,
            'level': self.level,
            'probability': float(self.probability),
            'description': self.description
        }


def calculate_risk_score(fraud_probability: float) -> RiskScore:
    """
    Convert fraud probability to risk score and level.
    
    Parameters:
        fraud_probability (float): Fraud probability from model (0.0 to 1.0)
                                  or as percentage (0.0 to 100.0)
    
    Returns:
        RiskScore: Structured risk assessment with score, level, and description
    
    Risk Mapping:
        0-30 (0-0.30)     → Low Risk
        30-70 (0.30-0.70) → Medium Risk
        70-100 (0.70-1.00) → High Risk
    
    Examples:
        >>> calculate_risk_score(0.05)
        RiskScore(score=5, level=Low Risk, probability=0.05)
        
        >>> calculate_risk_score(0.50)
        RiskScore(score=50, level=Medium Risk, probability=0.50)
        
        >>> calculate_risk_score(0.85)
        RiskScore(score=85, level=High Risk, probability=0.85)
    
    Raises:
        ValueError: If probability is not in range [0, 1] or [0, 100]
    """
    
    # Normalize if given as percentage (0-100)
    if fraud_probability > 1.0:
        if fraud_probability <= 100.0:
            fraud_probability = fraud_probability / 100.0
        else:
            raise ValueError(
                f"Fraud probability must be in range [0, 1] or [0, 100]. "
                f"Got: {fraud_probability}"
            )
    
    # Validate range
    if fraud_probability < 0.0 or fraud_probability > 1.0:
        raise ValueError(
            f"Fraud probability must be in range [0, 1]. Got: {fraud_probability}"
        )
    
    # Convert to 0-100 scale
    score = int(fraud_probability * 100)
    
    # Determine risk level and description
    if score < 30:
        level = RiskLevel.LOW.value
        description = (
            f"Transaction shows minimal fraud indicators. "
            f"Probability: {fraud_probability:.2%}"
        )
    elif score < 70:
        level = RiskLevel.MEDIUM.value
        description = (
            f"Transaction shows moderate fraud indicators and requires review. "
            f"Probability: {fraud_probability:.2%}"
        )
    else:
        level = RiskLevel.HIGH.value
        description = (
            f"Transaction shows significant fraud indicators and is flagged for investigation. "
            f"Probability: {fraud_probability:.2%}"
        )
    
    return RiskScore(
        score=score,
        level=level,
        probability=fraud_probability,
        description=description
    )


def calculate_risk_scores_batch(
    fraud_probabilities: List[float]
) -> List[RiskScore]:
    """
    Convert multiple fraud probabilities to risk scores.
    
    Parameters:
        fraud_probabilities (List[float]): List of fraud probabilities
    
    Returns:
        List[RiskScore]: List of risk assessments
    
    Example:
        >>> probs = [0.05, 0.35, 0.75, 0.95]
        >>> risks = calculate_risk_scores_batch(probs)
        >>> for risk in risks:
        ...     print(f"Score: {risk.score} ({risk.level})")
        Score: 5 (Low Risk)
        Score: 35 (Medium Risk)
        Score: 75 (High Risk)
        Score: 95 (High Risk)
    """
    return [calculate_risk_score(prob) for prob in fraud_probabilities]


def add_risk_scores_to_dataframe(
    df: pd.DataFrame,
    probability_column: str = 'fraud_probability',
    add_description: bool = True
) -> pd.DataFrame:
    """
    Add risk score columns to a DataFrame.
    
    Parameters:
        df (pd.DataFrame): DataFrame with fraud probabilities
        probability_column (str): Name of probability column
        add_description (bool): Whether to add description column
    
    Returns:
        pd.DataFrame: Original DataFrame with new columns:
                     - risk_score (0-100)
                     - risk_level (Low/Medium/High)
                     - risk_description (optional)
    
    Example:
        >>> df = pd.DataFrame({'fraud_probability': [0.05, 0.50, 0.85]})
        >>> df = add_risk_scores_to_dataframe(df)
        >>> print(df)
           fraud_probability  risk_score  risk_level
        0               0.05           5    Low Risk
        1               0.50          50  Medium Risk
        2               0.85          85   High Risk
    """
    df = df.copy()
    
    # Validate column exists
    if probability_column not in df.columns:
        raise ValueError(
            f"Column '{probability_column}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Calculate risk scores
    risk_scores = df[probability_column].apply(calculate_risk_score)
    
    # Add columns
    df['risk_score'] = [r.score for r in risk_scores]
    df['risk_level'] = [r.level for r in risk_scores]
    
    if add_description:
        df['risk_description'] = [r.description for r in risk_scores]
    
    return df


def get_risk_statistics(fraud_probabilities: List[float]) -> dict:
    """
    Get statistics about risk distribution.
    
    Parameters:
        fraud_probabilities (List[float]): List of fraud probabilities
    
    Returns:
        dict: Statistics including count by risk level, averages, etc.
    
    Example:
        >>> probs = [0.05, 0.10, 0.35, 0.50, 0.75, 0.90]
        >>> stats = get_risk_statistics(probs)
        >>> print(f"Low Risk: {stats['low_count']}, Medium Risk: {stats['medium_count']}, High Risk: {stats['high_count']}")
        Low Risk: 2, Medium Risk: 2, High Risk: 2
    """
    risks = calculate_risk_scores_batch(fraud_probabilities)
    
    low_risks = [r for r in risks if r.score < 30]
    medium_risks = [r for r in risks if 30 <= r.score < 70]
    high_risks = [r for r in risks if r.score >= 70]
    
    return {
        'total_transactions': len(risks),
        'low_count': len(low_risks),
        'medium_count': len(medium_risks),
        'high_count': len(high_risks),
        'low_percentage': len(low_risks) / len(risks) * 100,
        'medium_percentage': len(medium_risks) / len(risks) * 100,
        'high_percentage': len(high_risks) / len(risks) * 100,
        'average_probability': sum(fraud_probabilities) / len(fraud_probabilities),
        'average_score': sum(r.score for r in risks) / len(risks),
        'max_score': max(r.score for r in risks),
        'min_score': min(r.score for r in risks),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions for common use cases
# ═══════════════════════════════════════════════════════════════════════════

def is_high_risk(fraud_probability: float, threshold: int = 70) -> bool:
    """
    Check if transaction is high risk.
    
    Parameters:
        fraud_probability (float): Fraud probability (0-1)
        threshold (int): Score threshold for high risk (default: 70)
    
    Returns:
        bool: True if score >= threshold
    
    Example:
        >>> is_high_risk(0.75)
        True
        >>> is_high_risk(0.25)
        False
    """
    risk = calculate_risk_score(fraud_probability)
    return risk.score >= threshold


def get_risk_color(fraud_probability: float) -> str:
    """
    Get color for risk level (for UI/visualization).
    
    Parameters:
        fraud_probability (float): Fraud probability (0-1)
    
    Returns:
        str: Color code ('green', 'orange', 'red')
    
    Example:
        >>> get_risk_color(0.15)
        'green'
        >>> get_risk_color(0.50)
        'orange'
        >>> get_risk_color(0.85)
        'red'
    """
    risk = calculate_risk_score(fraud_probability)
    
    if risk.score < 30:
        return 'green'
    elif risk.score < 70:
        return 'orange'
    else:
        return 'red'


def get_risk_emoji(fraud_probability: float) -> str:
    """
    Get emoji representation of risk level.
    
    Parameters:
        fraud_probability (float): Fraud probability (0-1)
    
    Returns:
        str: Emoji ('✅', '⚠️', or '🚨')
    
    Example:
        >>> get_risk_emoji(0.15)
        '✅'
        >>> get_risk_emoji(0.50)
        '⚠️'
        >>> get_risk_emoji(0.85)
        '🚨'
    """
    risk = calculate_risk_score(fraud_probability)
    
    if risk.score < 30:
        return '✅'
    elif risk.score < 70:
        return '⚠️'
    else:
        return '🚨'
