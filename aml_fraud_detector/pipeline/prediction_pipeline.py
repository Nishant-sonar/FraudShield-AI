    
import sys
import os
import pandas as pd
import numpy as np

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
from aml_fraud_detector.utils.main_utils import load_object
from aml_fraud_detector.risk_scoring import calculate_risk_score, RiskScore
from aml_fraud_detector.rule_engine import RuleEngine, RuleConfig
from aml_fraud_detector.hybrid_scorer import HybridScorer, HybridScorerConfig


class PredictionPipeline:
    def __init__(self):
        """Initialize prediction pipeline with hybrid scoring capability."""
        self.hybrid_scorer = None
        self.rule_engine = None
    
    def _initialize_hybrid_system(self):
        """Initialize hybrid scoring system on first use."""
        if self.hybrid_scorer is None:
            try:
                rule_config = RuleConfig()
                hybrid_config = HybridScorerConfig()
                self.rule_engine = RuleEngine(rule_config)
                self.hybrid_scorer = HybridScorer(hybrid_config, rule_config)
                logging.info("Hybrid scoring system initialized")
            except Exception as e:
                logging.warning(f"Could not initialize hybrid scoring system: {e}")
                raise CustomerException(e, sys) 

    def predict(self, features):
        try: 
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            predictions = model.predict(data_scaled)
            return predictions
        except Exception as e:
            raise CustomerException(e, sys)
        
    def predict_proba(self, features):
        try: 
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            predictions_prob = model.predict_proba(data_scaled)
            return predictions_prob
        except Exception as e:
            raise CustomerException(e, sys)
    
    def predict_with_risk_score(self, features):
        """
        Get predictions with risk scores and levels.
        
        Returns:
            dict: {
                'predictions': array of binary predictions (0/1),
                'probabilities': array of fraud probabilities,
                'risk_scores': array of RiskScore objects,
                'scores': array of risk scores (0-100),
                'risk_levels': array of risk levels
            }
        """
        try:
            predictions = self.predict(features)
            probabilities = self.predict_proba(features)[:, 1]  # Get fraud probability
            
            # Calculate risk scores
            risk_scores = [calculate_risk_score(prob) for prob in probabilities]
            scores = [risk.score for risk in risk_scores]
            risk_levels = [risk.level for risk in risk_scores]
            
            logging.info(f"Predictions with risk scores generated for {len(predictions)} instances")
            
            return {
                'predictions': predictions,
                'probabilities': probabilities,
                'risk_scores': risk_scores,
                'scores': scores,
                'risk_levels': risk_levels
            }
        except Exception as e:
            raise CustomerException(e, sys)
    
    def predict_single_with_risk(self, features):
        """
        Get prediction with risk score for a single instance.
        
        Returns:
            dict: {
                'prediction': 0/1,
                'probability': fraud probability (0-1),
                'risk_score': RiskScore object,
                'score': risk score (0-100),
                'level': risk level string,
                'description': risk description
            }
        """
        try:
            result = self.predict_with_risk_score(features)
            
            return {
                'prediction': int(result['predictions'][0]),
                'probability': float(result['probabilities'][0]),
                'risk_score': result['risk_scores'][0],
                'score': result['scores'][0],
                'level': result['risk_levels'][0],
                'description': result['risk_scores'][0].description
            }
        except Exception as e:
            raise CustomerException(e, sys)
    
    def predict_with_hybrid_score(self, features, transaction_data=None):
        """
        Get predictions with hybrid scoring (ML + rules based).
        
        Args:
            features: DataFrame with model features
            transaction_data: Optional dict or DataFrame with transaction context for rules
                Expected keys: 'from_bank', 'amount_received', 'receiving_currency', 
                               'to_bank', 'amount_paid', 'payment_currency'
        
        Returns:
            dict: {
                'predictions': array of binary predictions (0/1),
                'probabilities': array of fraud probabilities,
                'risk_scores': array of RiskScore objects,
                'scores': array of ML-based risk scores (0-100),
                'risk_levels': array of ML-based risk levels,
                'hybrid_results': array of HybridScoringResult objects,
                'hybrid_scores': array of hybrid scores (0-100),
                'hybrid_levels': array of hybrid risk levels
            }
        """
        try:
            self._initialize_hybrid_system()
            
            # Get ML predictions
            predictions = self.predict(features)
            probabilities = self.predict_proba(features)[:, 1]
            
            # Calculate ML-based risk scores
            risk_scores = [calculate_risk_score(prob) for prob in probabilities]
            scores = [risk.score for risk in risk_scores]
            risk_levels = [risk.level for risk in risk_scores]
            
            # Calculate hybrid scores
            hybrid_results = []
            hybrid_scores = []
            hybrid_levels = []
            
            for idx, ml_prob in enumerate(probabilities):
                # Get transaction data for this instance
                transaction = {}
                if transaction_data is not None:
                    if isinstance(transaction_data, pd.DataFrame):
                        if idx < len(transaction_data):
                            transaction = transaction_data.iloc[idx].to_dict()
                    elif isinstance(transaction_data, dict):
                        transaction = transaction_data
                    elif isinstance(transaction_data, list) and idx < len(transaction_data):
                        transaction = transaction_data[idx]
                
                # Calculate hybrid score
                hybrid_result = self.hybrid_scorer.calculate_hybrid_score(ml_prob, transaction)
                hybrid_results.append(hybrid_result)
                hybrid_scores.append(hybrid_result.hybrid_score)
                hybrid_levels.append(hybrid_result.hybrid_level)
            
            logging.info(
                f"Hybrid scoring generated for {len(predictions)} instances. "
                f"Average hybrid score: {sum(hybrid_scores)/max(len(hybrid_scores), 1):.1f}"
            )
            
            return {
                'predictions': predictions,
                'probabilities': probabilities,
                'risk_scores': risk_scores,
                'scores': scores,
                'risk_levels': risk_levels,
                'hybrid_results': hybrid_results,
                'hybrid_scores': hybrid_scores,
                'hybrid_levels': hybrid_levels
            }
        except Exception as e:
            raise CustomerException(e, sys)
    
    def predict_single_with_hybrid(self, features, transaction_data=None):
        """
        Get prediction with hybrid scoring for a single instance.
        
        Args:
            features: DataFrame with model features (single row)
            transaction_data: Optional dict with transaction context for rules
                Expected keys: 'from_bank', 'amount_received', 'receiving_currency', 
                               'to_bank', 'amount_paid', 'payment_currency'
        
        Returns:
            dict: {
                'prediction': 0/1 (binary),
                'probability': fraud probability (0-1),
                'risk_score': RiskScore object,
                'score': ML risk score (0-100),
                'level': ML risk level string,
                'description': ML risk description,
                'hybrid_result': HybridScoringResult object,
                'hybrid_score': final hybrid score (0-100),
                'hybrid_level': final hybrid level,
                'triggered_rules': list of rule names,
                'rule_boosted': whether rules boosted the ML score,
                'confidence': confidence in hybrid assessment (0-1)
            }
        """
        try:
            result = self.predict_with_hybrid_score(features, transaction_data)
            
            hybrid_result = result['hybrid_results'][0]
            
            return {
                'prediction': int(result['predictions'][0]),
                'probability': float(result['probabilities'][0]),
                'risk_score': result['risk_scores'][0],
                'score': result['scores'][0],
                'level': result['risk_levels'][0],
                'description': result['risk_scores'][0].description,
                'hybrid_result': hybrid_result,
                'hybrid_score': hybrid_result.hybrid_score,
                'hybrid_level': hybrid_result.hybrid_level,
                'triggered_rules': hybrid_result.triggered_rules,
                'rule_boosted': len(hybrid_result.triggered_rules) > 0,
                'confidence': hybrid_result.confidence
            }
        except Exception as e:
            raise CustomerException(e, sys)
    
    def get_hybrid_report(self, features, transaction_data=None):
        """
        Get detailed text report comparing ML and hybrid scoring.
        
        Args:
            features: DataFrame with model features (single row)
            transaction_data: Optional dict with transaction context
        
        Returns:
            str: Formatted text report
        """
        try:
            result = self.predict_single_with_hybrid(features, transaction_data)
            hybrid_result = result['hybrid_result']
            
            report = hybrid_result.get_detailed_report()
            return report
        except Exception as e:
            raise CustomerException(e, sys)
    
    def predict_with_threshold(self, features, threshold=0.05, apply_soft_boost=False, verbose=True):
        """
        ⭐ IMPROVED PREDICTION METHOD - Probability-Based with Configurable Threshold
        
        This method fixes the fraud detection issue by:
        1. Using predict_proba instead of predict (gets actual probabilities)
        2. Applying configurable threshold (default 0.05 for high recall)
        3. Generating risk scores
        4. Adding comprehensive debug logging
        5. Handling edge cases (class imbalance warnings)
        6. Optional soft boost for enhanced sensitivity
        
        Args:
            features: DataFrame with model features (can be single or multiple rows)
            threshold: Fraud probability threshold (default 0.05 = 5%)
                      - Use 0.05 for high recall (catch more fraud)
                      - Use 0.5 for traditional balanced classification
            apply_soft_boost: If True, slightly boost scores for borderline cases
            verbose: If True, print detailed debug logs
        
        Returns:
            dict: {
                'predictions': array of binary predictions (0=Legitimate, 1=Fraud)
                'probabilities': array of fraud probability estimates (0-1)
                'scores': array of risk scores (0-100)
                'levels': array of risk levels ('Low', 'Medium', 'High', 'Critical')
                'thresholds_used': threshold value(s) for each prediction
                'debug_info': dict with debugging information
            }
        
        Example:
            >>> result = pipeline.predict_with_threshold(
            ...     features=transaction_df,
            ...     threshold=0.05,
            ...     apply_soft_boost=True,
            ...     verbose=True
            ... )
            >>> print(f"Fraud detected: {result['predictions'][0]}")
            >>> print(f"Risk: {result['levels'][0]} ({result['scores'][0]}%)")
        """
        try:
            # Load model and preprocessor
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            # Preprocess features
            data_scaled = preprocessor.transform(features)
            
            # ===== CRITICAL FIX: Use predict_proba instead of predict =====
            probabilities_raw = model.predict_proba(data_scaled)
            fraud_probabilities = probabilities_raw[:, 1]  # Extract fraud class probability
            
            # ===== STEP 1: THRESHOLD-BASED PREDICTION =====
            predictions = (fraud_probabilities >= threshold).astype(int)
            
            # ===== STEP 2: RISK SCORING =====
            scores_raw = []
            for prob in fraud_probabilities:
                # Convert probability (0-1) to risk score (0-100)
                score = int(prob * 100)
                
                # Optional soft boost for borderline cases
                if apply_soft_boost and 0.02 < prob < 0.10:
                    score = min(int(score * 1.2), 100)  # Boost by 20%, cap at 100
                
                scores_raw.append(score)
            
            # ===== STEP 3: RISK LEVELS =====
            risk_levels = []
            for score in scores_raw:
                if score < 30:
                    level = "Low"
                elif score < 70:
                    level = "Medium"
                else:
                    level = "High"
                risk_levels.append(level)
            
            # ===== STEP 4: EDGE CASE DETECTION =====
            avg_prob = float(fraud_probabilities.mean())
            max_prob = float(fraud_probabilities.max())
            min_prob = float(fraud_probabilities.min())
            
            class_imbalance_warning = max_prob < 0.05
            
            # ===== STEP 5: DEBUG LOGGING =====
            if verbose:
                logging.info("=" * 70)
                logging.info("🔍 THRESHOLD-BASED FRAUD DETECTION ANALYSIS")
                logging.info("=" * 70)
                logging.info(f"Total instances: {len(predictions)}")
                logging.info(f"Threshold used: {threshold} ({threshold*100:.1f}%)")
                logging.info(f"Model: {type(model).__name__}")
                logging.info("")
                logging.info("📊 PROBABILITY STATISTICS:")
                logging.info(f"  Min: {min_prob:.4f} ({min_prob*100:.2f}%)")
                logging.info(f"  Max: {max_prob:.4f} ({max_prob*100:.2f}%)")
                logging.info(f"  Avg: {avg_prob:.4f} ({avg_prob*100:.2f}%)")
                logging.info(f"  Median: {float(np.median(fraud_probabilities)):.4f}")
                logging.info("")
                logging.info("📈 PREDICTIONS:")
                logging.info(f"  Fraud cases: {predictions.sum()}/{len(predictions)}")
                logging.info(f"  Legitimate cases: {len(predictions) - predictions.sum()}/{len(predictions)}")
                logging.info(f"  Fraud rate: {(predictions.sum()/len(predictions)*100):.1f}%")
                logging.info("")
                logging.info("⚠️  WARNINGS:")
                
                if class_imbalance_warning:
                    logging.warning(
                        f"  ⚠️  CLASS IMBALANCE DETECTED: Max probability ({max_prob*100:.2f}%) is below threshold ({threshold*100:.1f}%). "
                        f"Model may be biased due to class imbalance in training data. "
                        f"Consider: (1) Threshold adjustment, (2) Model retraining with SMOTE, (3) Rule-based boosting."
                    )
                else:
                    logging.info("  ✓ No class imbalance warnings")
                
                if apply_soft_boost:
                    logging.info(f"  ✓ Soft boost enabled for borderline cases (0.02 < prob < 0.10)")
                
                logging.info("")
                logging.info("🎯 THRESHOLD LOGIC:")
                logging.info(f"  IF probability >= {threshold} → FRAUD (1)")
                logging.info(f"  IF probability < {threshold} → LEGITIMATE (0)")
                logging.info("=" * 70)
            
            # ===== STEP 6: RETURN COMPREHENSIVE RESULT =====
            return {
                'predictions': predictions,
                'probabilities': fraud_probabilities,
                'scores': np.array(scores_raw),
                'levels': np.array(risk_levels),
                'thresholds_used': np.full(len(predictions), threshold),
                'debug_info': {
                    'min_probability': float(min_prob),
                    'max_probability': float(max_prob),
                    'avg_probability': float(avg_prob),
                    'class_imbalance_warning': class_imbalance_warning,
                    'threshold': threshold,
                    'soft_boost_applied': apply_soft_boost,
                    'model_type': type(model).__name__,
                    'instances_count': len(predictions),
                    'fraud_count': int(predictions.sum()),
                    'legitimate_count': int(len(predictions) - predictions.sum())
                }
            }
        
        except Exception as e:
            logging.error(f"Error in predict_with_threshold: {str(e)}")
            raise CustomerException(e, sys)
    
    def predict_single_with_threshold(self, features, threshold=0.05, apply_soft_boost=False, verbose=True):
        """
        Single instance prediction with threshold-based fraud detection.
        
        Args:
            features: DataFrame with single transaction (1 row)
            threshold: Fraud probability threshold (default 0.05)
            apply_soft_boost: If True, apply soft boost
            verbose: If True, print debug logs
        
        Returns:
            dict: {
                'prediction': 0 or 1 (binary)
                'probability': fraud probability (0-1)
                'score': risk score (0-100)
                'level': risk level string
                'threshold': threshold used
                'is_fraud': boolean
                'confidence': confidence score
                'message': human-readable prediction message
            }
        """
        try:
            result = self.predict_with_threshold(
                features, 
                threshold=threshold, 
                apply_soft_boost=apply_soft_boost,
                verbose=verbose
            )
            
            prediction = int(result['predictions'][0])
            probability = float(result['probabilities'][0])
            score = int(result['scores'][0])
            level = str(result['levels'][0])
            
            # Calculate confidence
            if prediction == 1:
                confidence = min(probability / max(threshold, 0.01), 1.0)
                message = f"[ALERT] FRAUD DETECTED - {level} risk ({score}%)"
            else:
                confidence = min((1 - probability) / max(1 - threshold, 0.01), 1.0)
                message = f"[OK] LEGITIMATE - {level} risk ({score}%)"
            
            return {
                'prediction': prediction,
                'probability': probability,
                'score': score,
                'level': level,
                'threshold': threshold,
                'is_fraud': prediction == 1,
                'confidence': float(np.clip(confidence, 0, 1)),
                'message': message
            }
        
        except Exception as e:
            logging.error(f"Error in predict_single_with_threshold: {str(e)}")
            raise CustomerException(e, sys)


# Features used in the model:
# Numerical: ['From Bank', 'To Bank', 'Amount Received', 'Amount Paid']
# Categorical: ['Receiving Currency', 'Payment Currency', 'Payment Format']
# Dropped: ['Timestamp', 'Account', 'Account.1', 'Day']
class CustomData:
    def __init__(self,
            from_bank: int,
            to_bank: int,
            amount_received: float,
            amount_paid: float,
            receiving_currency: str,
            payment_currency: str,
            payment_format: str,
            account: str = None,
            account_1: str = None,
            day: str = None):
    
        self.from_bank = from_bank
        self.to_bank = to_bank
        self.amount_received = amount_received
        self.amount_paid = amount_paid
        self.receiving_currency = receiving_currency
        self.payment_currency = payment_currency
        self.payment_format = payment_format
        # These are kept for backward compatibility but won't be used by the model
        self.account = account
        self.account_1 = account_1
        self.day = day
    
    def get_data_as_DataFrame(self):
        try:
            custom_data_input_dict = {
                "From Bank": [self.from_bank],
                "To Bank": [self.to_bank],
                "Amount Received": [self.amount_received],
                "Amount Paid": [self.amount_paid],
                "Receiving Currency": [self.receiving_currency],
                "Payment Currency": [self.payment_currency],
                "Payment Format": [self.payment_format]
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomerException(e, sys)
