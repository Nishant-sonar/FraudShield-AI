"""
SHAP Explainability Utilities for Web Application Integration
===============================================================

This module provides utility functions and classes for integrating SHAP
explainability into web applications (Streamlit, Flask, etc.).

Features:
- Simple API for web app integration
- Pre-computed explanations
- JSON-serializable output
- Error handling
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import os
import sys

sys.path.insert(0, os.getcwd())

from aml_fraud_detector.shap_explainability import SHAPExplainer
from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging


class SHAPWebIntegration:
    """
    Web-friendly wrapper for SHAP explainability.
    
    Provides JSON-serializable outputs suitable for web applications.
    """
    
    def __init__(self, explainer: SHAPExplainer = None):
        """
        Initialize web integration wrapper.
        
        Args:
            explainer (SHAPExplainer): Optional pre-initialized explainer
        """
        self.explainer = explainer or SHAPExplainer()
    
    def explain_transaction(self, X: pd.DataFrame, 
                           include_visualizations: bool = True) -> Dict[str, Any]:
        """
        Generate JSON-serializable explanation for transaction(s).
        
        Args:
            X (pd.DataFrame): Input transaction features
            include_visualizations (bool): Whether to include plot data
            
        Returns:
            Dict with explanation, predictions, and metadata
        """
        try:
            explanation = self.explainer.explain_prediction(X)
            
            output = {
                'success': True,
                'transactions': []
            }
            
            for i in range(len(X)):
                trans_explanation = {
                    'instance_id': i,
                    'prediction': int(explanation['predictions'][i]),
                    'prediction_label': 'FRAUD ⚠️' if explanation['predictions'][i] == 1 else 'NOT FRAUD ✓',
                    'fraud_probability': float(explanation['predictions_proba'][i][1]),
                    'not_fraud_probability': float(explanation['predictions_proba'][i][0]),
                    'base_value': float(explanation['base_value']),
                    'top_features': self._get_top_features(
                        explanation, i, top_n=10
                    ),
                    'all_features': self._get_all_features(
                        explanation, i
                    )
                }
                output['transactions'].append(trans_explanation)
            
            return output
            
        except Exception as e:
            logging.error(f"Error in web explanation: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_top_features(self, explanation: Dict[str, Any], 
                         instance_idx: int, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get top contributing features for an instance."""
        try:
            shap_values = explanation['shap_values'][instance_idx]
            features = explanation['feature_names']
            values = explanation['feature_values'][instance_idx]
            
            # Get indices of top absolute contributions
            top_indices = np.argsort(np.abs(shap_values))[-top_n:][::-1]
            
            top_features = []
            for rank, idx in enumerate(top_indices, 1):
                top_features.append({
                    'rank': rank,
                    'feature_name': str(features[idx]),
                    'feature_value': float(values[idx]),
                    'shap_value': float(shap_values[idx]),
                    'direction': 'increases_fraud' if shap_values[idx] > 0 else 'decreases_fraud',
                    'impact_magnitude': float(np.abs(shap_values[idx]))
                })
            
            return top_features
        except Exception as e:
            logging.error(f"Error getting top features: {e}")
            return []
    
    def _get_all_features(self, explanation: Dict[str, Any], 
                         instance_idx: int) -> List[Dict[str, Any]]:
        """Get all features with their contributions."""
        try:
            shap_values = explanation['shap_values'][instance_idx]
            features = explanation['feature_names']
            values = explanation['feature_values'][instance_idx]
            
            all_features = []
            for idx in range(len(features)):
                all_features.append({
                    'feature_name': str(features[idx]),
                    'feature_value': float(values[idx]),
                    'shap_value': float(shap_values[idx]),
                    'direction': 'increases_fraud' if shap_values[idx] > 0 else 'decreases_fraud',
                    'impact_magnitude': float(np.abs(shap_values[idx]))
                })
            
            return all_features
        except Exception as e:
            logging.error(f"Error getting all features: {e}")
            return []
    
    def get_explanation_summary(self, X: pd.DataFrame, 
                               instance_idx: int = 0) -> str:
        """
        Get human-readable explanation summary.
        
        Args:
            X (pd.DataFrame): Input features
            instance_idx (int): Instance to explain
            
        Returns:
            Formatted explanation string
        """
        try:
            return self.explainer.get_explanation_report(X.iloc[[instance_idx]])
        except Exception as e:
            return f"Error generating explanation: {str(e)}"
    
    def get_prediction_confidence(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Get prediction and confidence metrics.
        
        Useful for displaying in web UI with confidence level.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            Dict with prediction and confidence info
        """
        try:
            explanation = self.explainer.explain_prediction(X)
            
            result = {
                'predictions': explanation['predictions'].tolist(),
                'fraud_probabilities': explanation['predictions_proba'][:, 1].tolist(),
                'not_fraud_probabilities': explanation['predictions_proba'][:, 0].tolist(),
                'confidence_levels': []
            }
            
            for prob in explanation['predictions_proba']:
                max_prob = max(prob)
                if max_prob >= 0.9:
                    level = 'Very High'
                elif max_prob >= 0.8:
                    level = 'High'
                elif max_prob >= 0.7:
                    level = 'Medium'
                elif max_prob >= 0.6:
                    level = 'Low'
                else:
                    level = 'Very Low'
                
                result['confidence_levels'].append(level)
            
            return result
        except Exception as e:
            logging.error(f"Error getting prediction confidence: {e}")
            return {'error': str(e)}


class SHAPVisualizationData:
    """Generate data suitable for web-based visualization."""
    
    def __init__(self, explainer: SHAPExplainer):
        """Initialize with explainer instance."""
        self.explainer = explainer
    
    def get_feature_importance_data(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Get feature importance data for bar chart.
        
        Returns data structure suitable for plotting libraries.
        """
        try:
            explanation = self.explainer.explain_prediction(X)
            
            # Calculate mean absolute SHAP values
            mean_abs_shap = np.mean(np.abs(explanation['shap_values']), axis=0)
            
            # Sort by importance
            indices = np.argsort(mean_abs_shap)[-10:][::-1]
            
            data = {
                'features': [explanation['feature_names'][i] for i in indices],
                'importances': mean_abs_shap[indices].tolist()
            }
            
            return data
        except Exception as e:
            logging.error(f"Error getting importance data: {e}")
            return {}
    
    def get_prediction_explanation_data(self, X: pd.DataFrame, 
                                       instance_idx: int = 0) -> Dict[str, Any]:
        """
        Get data for displaying prediction explanation.
        
        Suitable for waterfall-style visualizations.
        """
        try:
            explanation = self.explainer.explain_prediction(X.iloc[[instance_idx]])
            
            shap_values = explanation['shap_values'][0]
            features = explanation['feature_names']
            values = explanation['feature_values'][0]
            base_value = explanation['base_value']
            
            # Sort by absolute contribution
            indices = np.argsort(np.abs(shap_values))[-10:][::-1]
            
            data = {
                'base_value': float(base_value),
                'features': [],
                'values': [],
                'shap_values': [],
                'cumulative': []
            }
            
            cumulative = float(base_value)
            for idx in indices:
                data['features'].append(str(features[idx]))
                data['values'].append(float(values[idx]))
                data['shap_values'].append(float(shap_values[idx]))
                cumulative += float(shap_values[idx])
                data['cumulative'].append(cumulative)
            
            return data
        except Exception as e:
            logging.error(f"Error getting explanation data: {e}")
            return {}


# Singleton pattern for web app efficiency
_explainer_instance = None


def get_explainer() -> SHAPExplainer:
    """Get or create singleton SHAP explainer instance."""
    global _explainer_instance
    
    if _explainer_instance is None:
        _explainer_instance = SHAPExplainer()
    
    return _explainer_instance


def get_web_integration() -> SHAPWebIntegration:
    """Get web integration wrapper with singleton explainer."""
    return SHAPWebIntegration(get_explainer())
