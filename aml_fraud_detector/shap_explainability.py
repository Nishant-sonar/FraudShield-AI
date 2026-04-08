"""
SHAP Model Explainability Module for AML Fraud Detection
=========================================================

This module provides comprehensive SHAP-based explainability for the trained
Random Forest fraud detection model. It enables interpretation of model 
predictions through feature importance visualization and contribution analysis.

Key Features:
- Load trained model and preprocessing pipeline
- Generate SHAP values for individual predictions
- Create multiple visualization types
- Production-ready with error handling
- Modular and extensible design

Author: AML Fraud Detection Team
Date: 2026
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from typing import Tuple, Dict, Any, List

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project to path
sys.path.insert(0, os.getcwd())

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
from aml_fraud_detector.utils.main_utils import load_object


class SHAPExplainer:
    """
    SHAP-based explainability class for Random Forest fraud detection model.
    
    Provides methods to:
    - Load trained model and preprocessor
    - Generate SHAP explanations
    - Create visualizations
    - Export explanations
    
    Attributes:
        model: Trained Random Forest classifier
        preprocessor: Fitted preprocessing pipeline
        explainer: SHAP TreeExplainer instance
        shap_values: Cached SHAP values for training data
        training_data: Preprocessed training data used for SHAP base value
    """
    
    def __init__(self, model_path: str = "artifacts/model.pkl", 
                 preprocessor_path: str = "artifacts/preprocessor.pkl",
                 train_data_path: str = "artifacts/train.csv"):
        """
        Initialize SHAP explainer with model, preprocessor, and training data.
        
        Args:
            model_path (str): Path to saved Random Forest model
            preprocessor_path (str): Path to saved preprocessing pipeline
            train_data_path (str): Path to training data for SHAP baseline
            
        Raises:
            CustomerException: If files not found or loading fails
        """
        try:
            logging.info("Initializing SHAP Explainer...")
            
            # Load model and preprocessor
            self.model_path = model_path
            self.preprocessor_path = preprocessor_path
            
            self.model = self._load_model(model_path)
            self.preprocessor = self._load_preprocessor(preprocessor_path)
            
            # Load training data for SHAP background
            self.training_data = self._load_training_data(train_data_path)
            
            # Initialize SHAP explainer
            self.explainer = None
            self.shap_values = None
            self._initialize_explainer()
            
            logging.info("SHAP Explainer initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing SHAP Explainer: {str(e)}")
            raise CustomerException(e, sys)
    
    def _load_model(self, model_path: str):
        """Load trained Random Forest model from disk."""
        try:
            logging.info(f"Loading model from {model_path}")
            model = load_object(model_path)
            logging.info(f"Model loaded: {type(model).__name__}")
            return model
        except Exception as e:
            raise CustomerException(f"Failed to load model: {e}", sys)
    
    def _load_preprocessor(self, preprocessor_path: str):
        """Load preprocessing pipeline from disk."""
        try:
            logging.info(f"Loading preprocessor from {preprocessor_path}")
            preprocessor = load_object(preprocessor_path)
            logging.info(f"Preprocessor loaded: {type(preprocessor).__name__}")
            return preprocessor
        except Exception as e:
            raise CustomerException(f"Failed to load preprocessor: {e}", sys)
    
    def _load_training_data(self, train_data_path: str) -> np.ndarray:
        """
        Load and preprocess training data for SHAP background.
        
        Using training data as background provides better baseline for
        SHAP value calculation.
        """
        try:
            logging.info(f"Loading training data from {train_data_path}")
            df = pd.read_csv(train_data_path)
            
            # Remove target variable if present
            if 'Is Laundering' in df.columns:
                X_train = df.drop('Is Laundering', axis=1)
            else:
                X_train = df
            
            # Drop high-cardinality columns (same as training)
            high_cardinality_cols = ['Timestamp', 'Account', 'Account.1']
            X_train = X_train.drop(
                columns=[col for col in high_cardinality_cols if col in X_train.columns]
            )
            
            # Preprocess training data
            X_train_processed = self.preprocessor.transform(X_train)
            
            # Convert to DataFrame for feature names
            feature_names = self._get_feature_names()
            X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
            
            logging.info(f"Training data loaded and preprocessed: {X_train_df.shape}")
            return X_train_df
            
        except Exception as e:
            raise CustomerException(f"Failed to load training data: {e}", sys)
    
    def _initialize_explainer(self):
        """Initialize SHAP TreeExplainer with model and training data background."""
        try:
            logging.info("Initializing SHAP TreeExplainer...")
            
            # Use sample of training data for efficiency
            # SHAP can be slow with large datasets, so we sample
            background_data = self.training_data.sample(
                n=min(100, len(self.training_data)),
                random_state=42
            )
            
            # Create TreeExplainer
            self.explainer = shap.TreeExplainer(
                self.model,
                background_data
            )
            
            logging.info("SHAP TreeExplainer initialized successfully")
            
        except Exception as e:
            raise CustomerException(f"Failed to initialize SHAP explainer: {e}", sys)
    
    def _get_feature_names(self) -> List[str]:
        """
        Extract feature names from preprocessor.
        
        Returns:
            List of feature names after preprocessing
        """
        try:
            feature_names = self.preprocessor.get_feature_names_out()
            return list(feature_names)
        except Exception as e:
            logging.warning(f"Could not extract feature names: {e}")
            # Fallback: create generic names
            n_features = self.preprocessor.transform(
                pd.DataFrame({'From Bank': [0], 'To Bank': [0], 
                             'Amount Received': [0], 'Amount Paid': [0],
                             'Receiving Currency': ['USD'], 'Payment Currency': ['USD'],
                             'Payment Format': ['Wire']})
            ).shape[1]
            return [f"Feature_{i}" for i in range(n_features)]
    
    def explain_prediction(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate SHAP explanation for a single or batch of predictions.
        
        Args:
            X (pd.DataFrame): Input features (must have required columns)
            
        Returns:
            Dict containing:
            - shap_values: SHAP values array
            - base_value: SHAP base value (average model output)
            - prediction: Model prediction
            - prediction_proba: Prediction probabilities
            - feature_names: Feature names
            - feature_values: Preprocessed feature values
            - contributions: Feature contributions to prediction
            
        Raises:
            CustomerException: If explanation generation fails
        """
        try:
            logging.info(f"Generating SHAP explanation for {len(X)} instance(s)")
            
            # Preprocess input data
            X_processed = self.preprocessor.transform(X)
            
            # Convert to DataFrame with feature names
            feature_names = self._get_feature_names()
            X_df = pd.DataFrame(X_processed, columns=feature_names)
            
            # Generate SHAP values
            shap_values = self.explainer.shap_values(X_df)
            
            # Get predictions
            predictions = self.model.predict(X_processed)
            predictions_proba = self.model.predict_proba(X_processed)
            
            # For binary classification, shap_values is list [shap_0, shap_1]
            # We use shap_values for class 1 (fraud)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Fraud class
            
            # Calculate contributions (absolute impact on prediction)
            contributions = np.abs(shap_values)
            
            explanation = {
                'shap_values': shap_values,
                'base_value': self.explainer.expected_value[1],  # Fraud class
                'predictions': predictions,
                'predictions_proba': predictions_proba,
                'feature_names': feature_names,
                'feature_values': X_df.values,
                'contributions': contributions,
                'input_data': X
            }
            
            logging.info("SHAP explanation generated successfully")
            return explanation
            
        except Exception as e:
            logging.error(f"Error generating SHAP explanation: {str(e)}")
            raise CustomerException(e, sys)
    
    def plot_force_plot(self, X: pd.DataFrame, instance_idx: int = 0,
                       save_path: str = None, figsize: Tuple[int, int] = (16, 3)):
        """
        Create SHAP force plot showing contribution of each feature.
        
        Force plot visualizes how each feature pushes the prediction
        from the base value to the final prediction.
        
        Args:
            X (pd.DataFrame): Input features
            instance_idx (int): Index of instance to explain (default: 0)
            save_path (str): Path to save figure (optional)
            figsize (Tuple): Figure size
            
        Returns:
            matplotlib figure object
        """
        try:
            logging.info(f"Creating force plot for instance {instance_idx}")
            
            explanation = self.explain_prediction(X.iloc[[instance_idx]])
            
            X_processed = self.preprocessor.transform(X.iloc[[instance_idx]])
            feature_names = self._get_feature_names()
            X_df = pd.DataFrame(X_processed, columns=feature_names)
            
            # Create force plot
            plt.figure(figsize=figsize)
            shap.force_plot(
                explanation['base_value'],
                explanation['shap_values'][0],
                X_df.iloc[0],
                matplotlib=True,
                show=False
            )
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logging.info(f"Force plot saved to {save_path}")
            
            return plt.gcf()
            
        except Exception as e:
            logging.error(f"Error creating force plot: {str(e)}")
            raise CustomerException(e, sys)
    
    def plot_waterfall(self, X: pd.DataFrame, instance_idx: int = 0,
                      save_path: str = None, figsize: Tuple[int, int] = (10, 6)):
        """
        Create SHAP waterfall plot showing cumulative feature contributions.
        
        Waterfall plot shows how each feature progressively changes the
        prediction starting from the base value.
        
        Args:
            X (pd.DataFrame): Input features
            instance_idx (int): Index of instance to explain (default: 0)
            save_path (str): Path to save figure (optional)
            figsize (Tuple): Figure size
            
        Returns:
            matplotlib figure object
        """
        try:
            logging.info(f"Creating waterfall plot for instance {instance_idx}")
            
            explanation = self.explain_prediction(X.iloc[[instance_idx]])
            X_processed = self.preprocessor.transform(X.iloc[[instance_idx]])
            feature_names = self._get_feature_names()
            X_df = pd.DataFrame(X_processed, columns=feature_names)
            
            # Create explanation object for waterfall
            explainer_obj = shap.Explanation(
                values=explanation['shap_values'][0],
                base_values=explanation['base_value'],
                data=X_df.iloc[0],
                feature_names=feature_names
            )
            
            # Create waterfall plot
            plt.figure(figsize=figsize)
            shap.waterfall_plot(explainer_obj, show=False)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logging.info(f"Waterfall plot saved to {save_path}")
            
            return plt.gcf()
            
        except Exception as e:
            logging.error(f"Error creating waterfall plot: {str(e)}")
            raise CustomerException(e, sys)
    
    def plot_summary_bar(self, X: pd.DataFrame, save_path: str = None,
                        figsize: Tuple[int, int] = (10, 6), max_display: int = 15):
        """
        Create SHAP summary bar plot showing average absolute SHAP values.
        
        Bar plot shows which features are most important overall for
        the model's predictions.
        
        Args:
            X (pd.DataFrame): Input features (can be multiple instances)
            save_path (str): Path to save figure (optional)
            figsize (Tuple): Figure size
            max_display (int): Maximum features to display
            
        Returns:
            matplotlib figure object
        """
        try:
            logging.info(f"Creating summary bar plot for {len(X)} instances")
            
            explanation = self.explain_prediction(X)
            
            # Create summary plot
            plt.figure(figsize=figsize)
            shap.summary_plot(
                explanation['shap_values'],
                explanation['feature_values'],
                feature_names=explanation['feature_names'],
                plot_type='bar',
                max_display=max_display,
                show=False
            )
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logging.info(f"Summary bar plot saved to {save_path}")
            
            return plt.gcf()
            
        except Exception as e:
            logging.error(f"Error creating summary bar plot: {str(e)}")
            raise CustomerException(e, sys)
    
    def plot_summary_beeswarm(self, X: pd.DataFrame, save_path: str = None,
                             figsize: Tuple[int, int] = (10, 6), max_display: int = 15):
        """
        Create SHAP beeswarm summary plot showing feature value effects.
        
        Beeswarm plot shows:
        - Which features are most important
        - Whether high/low feature values increase/decrease fraud prediction
        - Color: Red (high value) vs Blue (low value)
        
        Args:
            X (pd.DataFrame): Input features (can be multiple instances)
            save_path (str): Path to save figure (optional)
            figsize (Tuple): Figure size
            max_display (int): Maximum features to display
            
        Returns:
            matplotlib figure object
        """
        try:
            logging.info(f"Creating beeswarm plot for {len(X)} instances")
            
            explanation = self.explain_prediction(X)
            
            # Create beeswarm plot
            plt.figure(figsize=figsize)
            shap.summary_plot(
                explanation['shap_values'],
                explanation['feature_values'],
                feature_names=explanation['feature_names'],
                plot_type='violin',
                max_display=max_display,
                show=False
            )
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logging.info(f"Beeswarm plot saved to {save_path}")
            
            return plt.gcf()
            
        except Exception as e:
            logging.error(f"Error creating beeswarm plot: {str(e)}")
            raise CustomerException(e, sys)
    
    def plot_dependence(self, X: pd.DataFrame, feature_name: str, save_path: str = None,
                       figsize: Tuple[int, int] = (10, 6)):
        """
        Create SHAP dependence plot for a specific feature.
        
        Dependence plot shows relationship between feature value and
        SHAP contribution to fraud prediction.
        
        Args:
            X (pd.DataFrame): Input features
            feature_name (str): Name of feature to analyze
            save_path (str): Path to save figure (optional)
            figsize (Tuple): Figure size
            
        Returns:
            matplotlib figure object
        """
        try:
            logging.info(f"Creating dependence plot for feature: {feature_name}")
            
            explanation = self.explain_prediction(X)
            feature_names = explanation['feature_names']
            
            if feature_name not in feature_names:
                raise ValueError(f"Feature '{feature_name}' not found in model features")
            
            feature_idx = feature_names.index(feature_name)
            
            # Create dependence plot
            plt.figure(figsize=figsize)
            shap.dependence_plot(
                feature_idx,
                explanation['shap_values'],
                explanation['feature_values'],
                feature_names=feature_names,
                show=False
            )
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logging.info(f"Dependence plot saved to {save_path}")
            
            return plt.gcf()
            
        except Exception as e:
            logging.error(f"Error creating dependence plot: {str(e)}")
            raise CustomerException(e, sys)
    
    def get_explanation_report(self, X: pd.DataFrame, instance_idx: int = 0) -> str:
        """
        Generate Human-readable explanation report for a prediction.
        
        Args:
            X (pd.DataFrame): Input features
            instance_idx (int): Index of instance to explain (default: 0)
            
        Returns:
            Formatted explanation text
        """
        try:
            logging.info(f"Generating explanation report for instance {instance_idx}")
            
            explanation = self.explain_prediction(X.iloc[[instance_idx]])
            
            prediction = explanation['predictions'][0]
            prediction_text = "FRAUD DETECTED 🚨" if prediction == 1 else "NOT FRAUD ✅"
            
            proba = explanation['predictions_proba'][0]
            fraud_confidence = proba[1] * 100
            
            # Top contributing features
            contributions = explanation['contributions'][0]
            feature_names = explanation['feature_names']
            
            # Get top 5 features
            top_indices = np.argsort(contributions)[-5:][::-1]
            
            report = f"""
{'='*70}
                 FRAUD PREDICTION EXPLANATION REPORT
{'='*70}

PREDICTION: {prediction_text}
Fraud Confidence: {fraud_confidence:.2f}%
Not Fraud Confidence: {(proba[0]*100):.2f}%

BASE VALUE (Average Model Prediction): {explanation['base_value']:.4f}

{'='*70}
TOP 5 FEATURES CONTRIBUTING TO PREDICTION:
{'='*70}

"""
            for rank, idx in enumerate(top_indices, 1):
                feature_name = feature_names[idx]
                shap_value = explanation['shap_values'][0][idx]
                feature_value = explanation['feature_values'][0][idx]
                direction = "↑ Increases Fraud" if shap_value > 0 else "↓ Decreases Fraud"
                
                report += f"""
{rank}. {feature_name}
   Value: {feature_value:.4f}
   SHAP Contribution: {shap_value:.4f}
   Effect: {direction}
"""
            
            report += f"\n{'='*70}\n"
            
            return report
            
        except Exception as e:
            logging.error(f"Error generating explanation report: {str(e)}")
            raise CustomerException(e, sys)


def create_explainer() -> SHAPExplainer:
    """
    Factory function to create SHAP explainer instance.
    
    Returns:
        SHAPExplainer: Initialized explainer ready for use
    """
    try:
        explainer = SHAPExplainer()
        return explainer
    except Exception as e:
        print(f"Error creating SHAP explainer: {e}")
        raise
