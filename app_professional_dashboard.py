"""
Professional AML Fraud Detection Dashboard
===========================================

Modern, production-ready Streamlit application with:
- Premium UI/UX design
- Two-column responsive layout
- Styled result cards with risk indicators
- SHAP explainability visualization
- Batch analysis capabilities
- Dark theme with fintech aesthetics

Usage:
    streamlit run app_professional_dashboard.py

Author: AML Team
Version: 2.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# Add project to path
sys.path.insert(0, os.getcwd())

from aml_fraud_detector.pipeline.prediction_pipeline import CustomData, PredictionPipeline
from aml_fraud_detector.utils.main_utils import load_object
from aml_fraud_detector.improved_predictor import ImprovedPredictor
from aml_fraud_detector.rule_engine import integrate_rule_boosting_with_prediction
from aml_fraud_detector.utils.mongo_handler import get_mongo_handler
from shap_explainer import FraudExplainer
from aml_fraud_detector.logger import logging

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="AML Fraud Detection Dashboard",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# THEME CONFIGURATION (Professional SaaS Dark/Light Theme System)
# =============================================================================

# Theme color palette
THEMES = {
    "dark": {
        "background": "#0E1117",
        "card": "#1E1E2F",
        "text_primary": "#FFFFFF",
        "text_secondary": "#B0B8C1",
        "accent": "#00C853",
        "border": "#2D333B",
        "header_bg": "#0066cc",
        "metric_bg": "rgba(30, 30, 47, 0.6)",
        "metric_border": "rgba(255, 255, 255, 0.08)",
        "success": "#10b981",
        "warning": "#f59e0b",
        "danger": "#ef4444",
        "primary": "#0066cc",
    },
    "light": {
        "background": "#F5F7FA",
        "card": "#FFFFFF",
        "text_primary": "#111111",
        "text_secondary": "#666666",
        "accent": "#4CAF50",
        "border": "#E5E7EB",
        "header_bg": "#0052a3",
        "metric_bg": "rgba(15, 23, 42, 0.04)",
        "metric_border": "rgba(0, 0, 0, 0.08)",
        "success": "#059669",
        "warning": "#d97706",
        "danger": "#dc2626",
        "primary": "#2563eb",
    }
}


def inject_theme_css():
    """Inject CSS based on current theme at the TOP of the page."""
    theme = st.session_state.get('theme_mode', 'dark')
    colors = THEMES.get(theme, THEMES["dark"])
    
    CUSTOM_CSS = f"""
<style>
    /* Main theme colors */
    :root {{
        --primary: {colors['primary']};
        --success: {colors['success']};
        --warning: {colors['warning']};
        --danger: {colors['danger']};
        --dark-bg: {colors['background']};
        --card-bg: {colors['card']};
        --text-primary: {colors['text_primary']};
        --text-secondary: {colors['text_secondary']};
        --border: {colors['border']};
        --metric-bg: {colors['metric_bg']};
        --metric-border: {colors['metric_border']};
        --accent: {colors['accent']};
        --header-bg: {colors['header_bg']};
    }}
    /* Main background */
    .stApp {{
        background-color: var(--dark-bg) !important;
        color: var(--text-primary) !important;
    }}
    
    /* All text elements */
    h1, h2, h3, h4, h5, h6, p, span, label, div {{
        color: var(--text-primary) !important;
    }}
    
    /* Headings - explicit styling */
    h1 {{
        color: var(--text-primary) !important;
        font-weight: 800 !important;
    }}
    
    h2 {{
        color: var(--text-primary) !important;
        font-weight: 700 !important;
    }}
    
    h3 {{
        color: var(--text-primary) !important;
        font-weight: 700 !important;
    }}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background-color: var(--card-bg) !important;
        border-right: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
    }}
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,  
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p {{
        color: var(--text-primary) !important;
    }}
    
    /* Header styling */
    .header-main {{
        background: linear-gradient(135deg, var(--primary) 0%, var(--header-bg) 100%) !important;
        padding: 32px 24px !important;
        border-radius: 16px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 10px 40px rgba(0, 102, 204, 0.2) !important;
    }}
    
    .header-main * {{
        color: white !important;
    }}
    
    .header-title {{
        font-size: 36px !important;
        font-weight: 800 !important;
        color: white !important;
        margin: 0 !important;
        letter-spacing: -0.5px !important;
    }}
    
    .header-subtitle {{
        font-size: 16px !important;
        color: rgba(255, 255, 255, 0.95) !important;
        margin-top: 8px !important;
        font-weight: 500 !important;
    }}
    
    /* Input form styling */
    .input-section {{
        background: var(--card-bg) !important;
        padding: 24px !important;
        border-radius: 16px !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
    }}
    
    .input-section-title {{
        font-size: 18px !important;
        font-weight: 700 !important;
        margin-bottom: 16px !important;
        color: var(--text-primary) !important;
    }}
    
    /* Section title */
    .section-title {{
        font-size: 18px !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        margin-top: 24px !important;
        margin-bottom: 12px !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
        letter-spacing: 0.3px !important;
    }}
    
    /* Button styling */
    .stButton > button {{
        background: linear-gradient(135deg, var(--primary) 0%, var(--header-bg) 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        box-shadow: 0 8px 16px rgba(0, 102, 204, 0.3) !important;
        transition: all 0.3s ease !important;
    }}
    
    .stButton > button:hover {{
        box-shadow: 0 12px 24px rgba(0, 102, 204, 0.4) !important;
        transform: translateY(-2px) !important;
        color: white !important;
    }}
    
    /* Metric styling */
    .metric-item {{
        background: var(--metric-bg) !important;
        border-radius: 12px !important;
        padding: 16px !important;
        border: 1px solid var(--metric-border) !important;
        color: var(--text-primary) !important;
    }}
    
    .metric-label {{
        font-size: 12px !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        color: var(--text-secondary) !important;
        margin-bottom: 8px !important;
        font-weight: 600 !important;
    }}
    
    .metric-value {{
        font-size: 18px !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
    }}
    
    /* Streamlit metric value styling */
    [data-testid="metric-container"] div:nth-child(1) {{
        font-size: 12px !important;
    }}
    
    [data-testid="metric-container"] div:nth-child(2) {{
        font-size: 12px !important;
        line-height: 1.2 !important;
    }}
    
    /* Metric container all text */
    [data-testid="metric-container"] {{
        font-size: 12px !important;
    }}
    
    [data-testid="metric-container"] div {{
        font-size: 12px !important;
    }}
    
    [data-testid="metric-container"] span {{
        font-size: 12px !important;
    }}
    
    [data-testid="metric-container"] * {{
        font-size: 12px !important;
    }}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {{
        background: transparent !important;
        border-bottom: 2px solid var(--border) !important;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        color: var(--text-secondary) !important;
        border-radius: 0 !important;
        border-bottom: 3px solid transparent !important;
    }}
    
    .stTabs [data-baseweb="tab"] p {{
        color: var(--text-secondary) !important;
    }}
    
    .stTabs [aria-selected="true"] {{
        color: var(--primary) !important;
        border-bottom: 3px solid var(--primary) !important;
    }}
    
    .stTabs [aria-selected="true"] p {{
        color: var(--primary) !important;
    }}
    
    /* Messages */
    .stSuccess {{
        background: rgba(16, 185, 129, 0.15) !important;
        border-left: 4px solid var(--success) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
    }}
    
    .stSuccess p, .stSuccess span, .stSuccess h3 {{
        color: var(--text-primary) !important;
    }}
    
    .stWarning {{
        background: rgba(245, 158, 11, 0.15) !important;
        border-left: 4px solid var(--warning) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
    }}
    
    .stWarning p, .stWarning span, .stWarning h3 {{
        color: var(--text-primary) !important;
    }}
    
    .stError {{
        background: rgba(239, 68, 68, 0.15) !important;
        border-left: 4px solid var(--danger) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
    }}
    
    .stError p, .stError span, .stError h3 {{
        color: var(--text-primary) !important;
    }}
    
    .stInfo {{
        background: rgba(0, 102, 204, 0.15) !important;
        border-left: 4px solid var(--primary) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
    }}
    
    .stInfo p, .stInfo span, .stInfo h3 {{
        color: var(--text-primary) !important;
    }}
    
    /* Input fields */
    .stNumberInput input {{
        background: var(--card-bg) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
    }}
    
    .stSelectbox {{
        color: var(--text-primary) !important;
    }}
    
    .stSelectbox > div > div {{
        background: var(--card-bg) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
    }}
    
    /* Tables */
    .stDataFrame {{
        background: var(--card-bg) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
    }}
    
    .stDataFrame thead th {{
        color: var(--text-primary) !important;
        background: var(--metric-bg) !important;
    }}
    
    .stDataFrame tbody td {{
        color: var(--text-primary) !important;
    }}
    
    /* Result cards */
    .result-card {{
        background: var(--card-bg) !important;
        border-radius: 16px !important;
        padding: 24px !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
    }}
</style>
"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def initialize_session_state():
    """Initialize all session state variables."""
    if 'theme_mode' not in st.session_state:
        st.session_state.theme_mode = "dark"
    
    if 'last_explanation' not in st.session_state:
        st.session_state.last_explanation = None
        st.session_state.last_X = None
        st.session_state.last_prediction = None
        st.session_state.last_probability = None
        st.session_state.last_risk_score = None
        st.session_state.batch_data = None


def render_theme_toggle():
    """Render theme toggle in sidebar with buttons."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🌗 Theme Settings")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🌙 Dark", use_container_width=True, key="btn_dark_theme"):
            st.session_state.theme_mode = "dark"
            st.rerun()
    
    with col2:
        if st.button("☀️ Light", use_container_width=True, key="btn_light_theme"):
            st.session_state.theme_mode = "light"
            st.rerun()
    
    # Show current theme
    current = "🌙 Dark Mode" if st.session_state.theme_mode == "dark" else "☀️ Light Mode"
    st.sidebar.success(f"**Active:** {current}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

@st.cache_resource
def load_explainer():
    """Load SHAP explainer (cached for performance)."""
    with st.spinner("🔄 Loading SHAP explainer..."):
        try:
            model = load_object('artifacts/model.pkl')
            preprocessor = load_object('artifacts/preprocessor.pkl')
            explainer = FraudExplainer(model, preprocessor)
            return explainer
        except Exception as e:
            st.error(f"❌ Error loading explainer: {e}")
            logging.error(f"Error loading explainer: {e}")
            return None


def get_risk_level(risk_score: float) -> tuple:
    """Determine risk level from risk score (0-100 scale).
    
    Args:
        risk_score: Risk score on 0-100 scale
        
    Returns:
        (risk_level: str, card_type: str, emoji: str, badge_class: str)
        
    Risk Levels:
    - Low Risk: 0-29
    - Medium Risk: 30-69
    - High Risk: 70-100
    """
    if risk_score < 30:
        return ("Low Risk", "safe", "OK", "risk-badge-low")
    elif risk_score < 70:
        return ("Medium Risk", "warning", "WARN", "risk-badge-medium")
    else:
        return ("High Risk", "fraud", "ALERT", "risk-badge-high")


def render_result_card(
    prediction: int,
    fraud_probability: float,
    risk_score: float,
    timestamp: str = None
):
    """Render styled result card with prediction details using Streamlit components."""
    
    # Use risk_score (0-100) for risk level determination
    risk_level, card_type, risk_emoji, badge_class = get_risk_level(risk_score)
    
    # Display prediction result
    if risk_score >= 70:
        st.error(f"🚨 **HIGH RISK** - Score {risk_score:.0f}/100")
    elif risk_score >= 30:
        st.warning(f"⚠️ **MEDIUM RISK** - Score {risk_score:.0f}/100")
    else:
        st.success(f"✅ **LOW RISK** - Score {risk_score:.0f}/100 - Transaction appears safe")
    
    # Show metrics in three columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-label">Fraud Probability</div>
            <div class="metric-value" style="font-size: 16px !important;">{fraud_probability*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-label">Risk Score</div>
            <div class="metric-value" style="font-size: 16px !important;">{risk_score:.0f}/100</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-item">
            <div class="metric-label">Risk Level</div>
            <div class="metric-value" style="font-size: 16px !important;">{risk_emoji} {risk_level}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Risk progress bar
    st.progress(min(int(risk_score), 100) / 100.0)
    st.caption(f"📊 Risk Score: {risk_score:.0f}%")
    
    if timestamp:
        st.caption(f"⏰ Analyzed at {timestamp}")


def render_transaction_input():
    """Render styled input form and return transaction data + raw values."""
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<div class="input-section-title">📝 Transaction Details</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        from_bank = st.number_input("🏦 From Bank ID", min_value=0, value=100, step=1)
        amount_received = st.number_input("💰 Amount Received", min_value=0.0, value=5000.75, step=100.0)
        receiving_currency = st.selectbox(
            "💱 Receiving Currency",
            ["US Dollar", "Bitcoin", "Euro", "Shekel", "Ruble", "Canadian Dollar",
             "Yen", "Yuan", "Mexican Peso", "Australian Dollar", "Swiss Franc",
             "UK Pound", "Rupee", "Brazil Real", "Saudi Riyal"]
        )
    
    with col2:
        to_bank = st.number_input("🏦 To Bank ID", min_value=0, value=200, step=1)
        amount_paid = st.number_input("💸 Amount Paid", min_value=0.0, value=5100.00, step=100.0)
        payment_currency = st.selectbox(
            "💱 Payment Currency",
            ["US Dollar", "Bitcoin", "Euro", "Shekel", "Ruble", "Canadian Dollar",
             "Yen", "Yuan", "Mexican Peso", "Australian Dollar", "Swiss Franc",
             "UK Pound", "Rupee", "Brazil Real", "Saudi Riyal"],
            index=0
        )
    
    payment_format = st.selectbox(
        "📋 Payment Format",
        ["Wire", "ACH", "Cheque", "Credit Card", "Reinvestment", "Bitcoin", "Cash"]
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Validate amounts (realistic transaction limit: $100 million)
    MAX_TRANSACTION_AMOUNT = 100_000_000  # $100M realistic limit
    
    if amount_received > MAX_TRANSACTION_AMOUNT or amount_paid > MAX_TRANSACTION_AMOUNT:
        st.warning(
            f"⚠️ WARNING: Unrealistic amount detected! "
            f"Typical transactions should not exceed ${MAX_TRANSACTION_AMOUNT:,}. "
            f"Please verify the amount."
        )
    
    # Create transaction
    transaction = CustomData(
        from_bank=int(from_bank),
        to_bank=int(to_bank),
        amount_received=amount_received,
        amount_paid=amount_paid,
        receiving_currency=receiving_currency,
        payment_currency=payment_currency,
        payment_format=payment_format
    )
    
    # Return DataFrame and raw values for boosting
    raw_values = {
        'from_bank': int(from_bank),
        'to_bank': int(to_bank),
        'amount_received': amount_received,
        'amount_paid': amount_paid,
        'receiving_currency': receiving_currency,
        'payment_currency': payment_currency,
        'payment_format': payment_format
    }
    
    return transaction.get_data_as_DataFrame(), raw_values


def render_shap_section(explanation: dict):
    """Render SHAP explainability section."""
    st.markdown('<div class="section-title">🧠 Why this decision?</div>', unsafe_allow_html=True)
    
    if explanation.get('error'):
        st.warning(f"⚠️ {explanation['error']}")
        return
    
    try:
        shap_values = explanation.get('shap_values')
        feature_names = explanation.get('feature_names', [])
        
        if shap_values is None or len(feature_names) == 0:
            st.info("ℹ️ SHAP analysis not available for this transaction")
            return
        
        # Handle 3D SHAP values
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
        
        # Get top 3 features
        sample_shap = shap_values[0]
        top_indices = np.argsort(np.abs(sample_shap))[::-1][:3]
        
        # Get theme colors
        theme = st.session_state.theme_mode
        colors = THEMES.get(theme, THEMES["dark"])
        
        col1, col2, col3 = st.columns(3)
        
        for i, (idx, col) in enumerate(zip(top_indices, [col1, col2, col3])):
            idx = int(idx)
            shap_value = float(sample_shap[idx])
            feature_name = feature_names[idx]
            magnitude = abs(shap_value)
            
            direction = "↑ Increases Fraud Risk" if shap_value > 0 else "↓ Decreases Fraud Risk"
            color = colors['danger'] if shap_value > 0 else colors['success']
            
            with col:
                st.markdown(f"""
                <div class="result-card" style="border-left: 5px solid {color};">
                    <div style="font-weight: 600; color: {color};">#{i+1} Factor</div>
                    <div style="font-size: 14px; font-weight: 700; margin: 8px 0; color: var(--text-primary);">{feature_name}</div>
                    <div style="font-size: 12px; color: {colors['text_secondary']};\">{direction}</div>
                    <div style="font-size: 12px; color: {colors['text_secondary']}; margin-top: 6px;">
                        Impact: {magnitude:.4f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Show all features in table
        with st.expander("📊 View All Contributing Features"):
            df_features = pd.DataFrame([
                {
                    '🔢 Rank': i+1,
                    'Feature': feature_names[int(idx)],
                    'SHAP Value': f"{sample_shap[int(idx)]:+.6f}",
                    'Direction': '↑ Risk' if sample_shap[int(idx)] > 0 else '↓ Safe',
                    'Magnitude': f"{abs(sample_shap[int(idx)]):.6f}"
                }
                for i, idx in enumerate(top_indices[:10])
            ])
            st.dataframe(df_features, use_container_width=True, hide_index=True)
    
    except Exception as e:
        st.error(f"❌ Error displaying SHAP analysis: {e}")
        logging.error(f"Error displaying SHAP: {e}")


def render_feature_importance(explainer: FraudExplainer, X_train: pd.DataFrame):
    """Render feature importance visualization."""
    st.markdown('<div class="section-title">📈 Feature Importance Overview</div>', unsafe_allow_html=True)
    
    try:
        with st.spinner("🔄 Computing feature importance..."):
            importance_df = explainer.get_feature_importance(X_train)
            top_importance = importance_df.head(10)
            
            # Get theme colors dynamically
            theme = st.session_state.theme_mode
            colors = THEMES.get(theme, THEMES["dark"])
            
            # Create visualization with theme-aware colors
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor(colors['background'])
            ax.set_facecolor(colors['card'])
            
            bars = ax.barh(range(len(top_importance)), top_importance['importance'].values)
            ax.set_yticks(range(len(top_importance)))
            ax.set_yticklabels(top_importance['feature'].values, color=colors['text_primary'])
            ax.set_xlabel('Importance Score', color=colors['text_secondary'])
            ax.set_title('Top 10 Most Important Features', color=colors['text_primary'], fontsize=14, fontweight='bold', pad=20)
            ax.invert_yaxis()
            
            # Color bars with gradient
            for i, bar in enumerate(bars):
                normalized = top_importance['importance'].values[i] / top_importance['importance'].max()
                bar.set_color(plt.cm.RdYlGn_r(normalized))
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color(colors['border'])
            ax.spines['bottom'].set_color(colors['border'])
            ax.tick_params(colors=colors['text_secondary'])
            ax.grid(axis='x', alpha=0.2, color=colors['border'])
            
            plt.tight_layout()
            st.pyplot(fig)
    
    except Exception as e:
        st.error(f"❌ Error computing feature importance: {e}")


def render_batch_results(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    batch_data: pd.DataFrame
):
    """Render batch analysis results with color coding."""
    st.markdown('<div class="section-title">📊 Batch Analysis Results</div>', unsafe_allow_html=True)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'Status': ['🚨 FRAUD' if p == 1 else '✅ SAFE' for p in predictions],
        'Fraud Probability': [f"{float(prob)*100:.1f}%" for prob in probabilities[:, 1]],
        'Risk Score': [int(float(prob)*100) for prob in probabilities[:, 1]],
        'Confidence': [f"{float(max(p))*100:.1f}%" for p in probabilities]
    })
    
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    
    # Summary statistics
    st.markdown('<div class="section-title">📈 Summary Statistics</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    fraud_count = (predictions == 1).sum()
    legitimate_count = (predictions == 0).sum()
    avg_fraud_prob = float(probabilities[:, 1].mean())
    
    with col1:
        st.metric("📋 Total Transactions", len(predictions))
    
    with col2:
        st.metric("🚨 Fraudulent", int(fraud_count), delta=f"{fraud_count/len(predictions)*100:.1f}%")
    
    with col3:
        st.metric("✅ Legitimate", int(legitimate_count), delta=f"{legitimate_count/len(predictions)*100:.1f}%")
    
    with col4:
        st.metric("⚠️ Avg Fraud Prob", f"{avg_fraud_prob:.1%}")
    
    # Detailed fraud analysis
    fraud_indices = np.where(predictions == 1)[0]
    if len(fraud_indices) > 0:
        st.markdown('<div class="section-title">🚨 High-Risk Transactions</div>', unsafe_allow_html=True)
        
        for idx in fraud_indices[:5]:  # Show first 5
            fraud_pct = float(probabilities[idx][1])*100
            risk_level, _, risk_emoji, _ = get_risk_level(probabilities[idx][1])
            
            with st.expander(f"{risk_emoji} Transaction #{idx}: {fraud_pct:.1f}% Fraud Risk"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Fraud Probability", f"{fraud_pct:.1f}%")
                with col2:
                    st.metric("Risk Score", int(fraud_pct))
                with col3:
                    st.metric("Risk Level", risk_level)
                
                # Show transaction details
                if idx < len(batch_data):
                    st.write("**Transaction Details:**")
                    st.dataframe(batch_data.iloc[[idx]], use_container_width=True, hide_index=True)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def save_transaction_to_db(raw_values: dict, risk_score: int, risk_level: str, prediction: int):
    """
    Save transaction prediction to MongoDB (Production-Safe)
    
    Features:
    - Safe error handling (MongoDB failures don't crash app)
    - Prevents duplicate inserts using session state
    - Timestamp validation and auto-addition
    - Schema validation
    - Comprehensive logging
    
    Args:
        raw_values: Dictionary with transaction details
        risk_score: Calculated risk score (0-100)
        risk_level: Risk level classification (Low, Medium, High)
        prediction: Binary prediction (0=safe, 1=fraud)
    
    Returns:
        tuple: (success: bool, doc_id: str or None, message: str)
    """
    try:
        # =====================================================================
        # 1. DUPLICATE PREVENTION: Check if already saved in this session
        # =====================================================================
        current_transaction_hash = hash(
            (raw_values.get('from_bank'), raw_values.get('to_bank'), 
             raw_values.get('amount_paid'), prediction)
        )
        
        if not hasattr(st.session_state, 'last_saved_transaction_hash'):
            st.session_state.last_saved_transaction_hash = None
        
        if st.session_state.last_saved_transaction_hash == current_transaction_hash:
            logging.debug("Transaction already saved in this session - skipping duplicate insert")
            return False, None, "⏭️ Transaction already saved in this session"
        
        # =====================================================================
        # 2. SCHEMA VALIDATION: Ensure required fields exist
        # =====================================================================
        required_fields = {
            'amount_paid': (float, int),
            'payment_currency': str,
            'receiving_currency': str,
        }
        
        for field, field_type in required_fields.items():
            value = raw_values.get(field)
            if value is None:
                logging.warning(f"Missing required field: {field}")
                return False, None, f"⚠️ Schema error: Missing {field}"
            
            if not isinstance(value, field_type):
                try:
                    raw_values[field] = field_type(value)
                except:
                    logging.warning(f"Invalid type for field {field}: expected {field_type}, got {type(value)}")
                    return False, None, f"⚠️ Invalid data type for {field}"
        
        # =====================================================================
        # 3. SCHEMA VALIDATION: Risk score bounds
        # =====================================================================
        if not (0 <= risk_score <= 100):
            logging.warning(f"Risk score out of bounds: {risk_score}")
            return False, None, f"⚠️ Invalid risk score: {risk_score}"
        
        # =====================================================================
        # 4. INITIALIZE MONGODB HANDLER
        # =====================================================================
        try:
            handler = get_mongo_handler()
        except Exception as e:
            logging.warning(f"Failed to initialize MongoDB handler: {str(e)}")
            return False, None, "⚠️ Database not available (app continues normally)"
        
        # =====================================================================
        # 5. CHECK MONGODB CONNECTION
        # =====================================================================
        if not handler.is_connected():
            logging.info("MongoDB not available - transaction analysis complete, data not persisted")
            return False, None, "ℹ️ Database offline (analysis complete)"
        
        # =====================================================================
        # 6. BUILD TRANSACTION DOCUMENT
        # =====================================================================
        transaction_data = {
            # Core transaction data
            "from_bank": int(raw_values.get('from_bank', 0)),
            "to_bank": int(raw_values.get('to_bank', 0)),
            "amount_paid": float(raw_values.get('amount_paid', 0)),
            "amount_received": float(raw_values.get('amount_received', 0)),
            "payment_currency": str(raw_values.get('payment_currency', 'USD')).upper(),
            "receiving_currency": str(raw_values.get('receiving_currency', 'USD')).upper(),
            "payment_format": str(raw_values.get('payment_format', 'Unknown')),
            
            # Prediction data
            "risk_score": int(risk_score),
            "risk_level": str(risk_level),
            "prediction": int(prediction),
            
            # Metadata
            "timestamp": datetime.now(),
            "is_fraud": prediction == 1,
            "saved_via": "streamlit_dashboard",
            "app_version": "2.1.0"
        }
        
        # =====================================================================
        # 7. INSERT TO MONGODB (with error handling)
        # =====================================================================
        try:
            doc_id = handler.insert_transaction(transaction_data)
            
            if doc_id:
                # Mark as saved in session to prevent duplicates
                st.session_state.last_saved_transaction_hash = current_transaction_hash
                st.session_state.last_saved_doc_id = doc_id
                
                logging.info(f"✓ Transaction saved to MongoDB: {doc_id}")
                return True, doc_id, f"✅ Saved (ID: {doc_id[:12]}...)"
            else:
                logging.warning("MongoDB insert returned None")
                return False, None, "⚠️ Save operation failed (retrying might help)"
                
        except Exception as insert_error:
            logging.error(f"MongoDB insert failed: {str(insert_error)}")
            return False, None, "⚠️ Save operation failed (app continues normally)"
    
    except Exception as e:
        logging.error(f"Unexpected error in save_transaction_to_db: {str(e)}")
        return False, None, f"⚠️ Unexpected error: app continues normally"


def main():

    """Main application entry point."""
    
    # Initialize session state
    initialize_session_state()
    
    # Inject dynamic theme CSS at the very top
    inject_theme_css()
    
    # Render sidebar theme toggle
    render_theme_toggle()
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📊 Dashboard Info
    - **Model**: Random Forest Classifier
    - **Features**: 15+ transaction attributes
    - **Explainability**: SHAP analysis
    - **Database**: MongoDB integration
    """)
    
    # Header section
    st.markdown("""
    <div class="header-main">
        <div class="header-title">🛡️ FraudShield AI</div>
        <div class="header-subtitle">Intelligent AML Fraud Detection System with Explainable AI
</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Load explainer
    explainer = load_explainer()
    
    if explainer is None:
        st.error("❌ Failed to load AI model. System unavailable.")
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📊 Explainability", "📦 Batch Analysis"])
    
    # ==========================================================================
    # TAB 1: PREDICTION
    # ==========================================================================
    with tab1:
        st.markdown("---")
        
        # Two-column layout: Input (left) | Results (right)
        col_input, col_results = st.columns([1, 1.2], gap="large")
        
        with col_input:
            st.markdown("### Transaction Input")
            X, raw_values = render_transaction_input()
            
            # Prediction button
            predict_button = st.button(
                "🔍 Analyze Transaction",
                use_container_width=True,
                key="predict_btn"
            )
        
        with col_results:
            st.markdown("### Analysis Results")
            
            if predict_button:
                with st.spinner("⏳ Analyzing transaction..."):
                    try:
                        # Step 1: Get ML prediction with threshold
                        pipeline = PredictionPipeline()
                        ml_result = pipeline.predict_single_with_threshold(X, threshold=0.05, verbose=False)
                        
                        # Step 2: Apply rule-based boosting
                        enhanced_result = integrate_rule_boosting_with_prediction(
                            ml_result,
                            transaction_data={
                                'amount_paid': raw_values['amount_paid'],
                                'amount_received': raw_values['amount_received'],
                                'payment_currency': raw_values['payment_currency'],
                                'receiving_currency': raw_values['receiving_currency'],
                                'from_bank': raw_values['from_bank'],
                                'to_bank': raw_values['to_bank']
                            },
                            apply_boost=True
                        )
                        
                        # Debug logging
                        print("="*60)
                        print(f"ML Score: {ml_result.get('score', 0)}")
                        print(f"Boosted Score: {enhanced_result.get('score', 0)}")
                        print(f"Boost Applied: {enhanced_result.get('boost_applied', False)}")
                        print(f"Boost Amount: {enhanced_result.get('boost_amount', 0)}")
                        print("="*60)
                        
                        # Use enhanced result for display
                        prediction = 1 if enhanced_result.get('level') == 'High' else 0
                        probability = enhanced_result.get('probability', 0)
                        risk_score = enhanced_result.get('score', 0)
                        risk_level = enhanced_result.get('level', 'Low')
                        
                        # Get SHAP explanation (using original ML prediction)
                        explanation = explainer.explain_prediction(X)
                        
                        # Store in session
                        st.session_state.last_explanation = explanation
                        st.session_state.last_X = X
                        st.session_state.last_prediction = prediction
                        st.session_state.last_probability = probability
                        st.session_state.last_risk_score = risk_score
                        
                        # Render result card with enhanced result
                        render_result_card(
                            prediction,
                            probability,
                            risk_score,
                            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        )
                        
                        # User feedback based on enhanced result
                        risk_level_display = enhanced_result.get('level', 'Low')
                        
                        if risk_level_display == 'High':
                            st.error(f"🚨 HIGH RISK: Score {risk_score}/100 - This transaction requires immediate manual review.")
                        elif risk_level_display == 'Medium':
                            st.warning(f"⚠️ MEDIUM RISK: Score {risk_score}/100 - This transaction shows suspicious patterns and requires attention.")
                        else:  # Low
                            st.success(f"✅ SAFE: Score {risk_score}/100 - This transaction passed all security checks.")
                        
                        # Show boost info
                        if enhanced_result.get('boost_applied'):
                            st.info("✨ Enhanced risk analysis applied")
                        
                        # Transaction details
                        st.markdown("### Transaction Details")
                        st.dataframe(X, use_container_width=True, hide_index=True)
                        
                        # Save transaction to MongoDB
                        success, doc_id, message = save_transaction_to_db(
                            raw_values=raw_values,
                            risk_score=int(risk_score),
                            risk_level=risk_level,
                            prediction=prediction
                        )
                        
                        # Show result message (only on success)
                        if success:
                            st.success(message)
                        
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {e}")
                        logging.error(f"Error in prediction: {e}")
            else:
                st.info("👈 Enter transaction details and click 'Analyze Transaction' to begin")
    
    # ==========================================================================
    # TAB 2: EXPLAINABILITY
    # ==========================================================================
    with tab2:
        st.markdown("---")
        
        if not hasattr(st.session_state, 'last_explanation') or st.session_state.last_explanation is None:
            st.info("👈 Go to the Prediction tab and run an analysis first")
        else:
            try:
                # SHAP feature contributions
                render_shap_section(st.session_state.last_explanation)
                
                st.markdown("---")
                
                # Feature importance
                st.markdown("### 📈 Model Feature Importance")
                st.info("Shows which features are most influential across all transactions")
                
                with st.spinner("Loading feature importance..."):
                    try:
                        train_df = pd.read_csv('artifacts/train.csv')
                        render_feature_importance(explainer, train_df)
                    except Exception as e:
                        st.warning(f"⚠️ Could not load training data: {e}")
                
                st.markdown("---")
                
                # Explanation guide
                st.markdown("### 📚 How to Read SHAP Results")
                with st.expander("Learn about SHAP Explainability", expanded=False):
                    st.markdown("""
                    **SHAP (SHapley Additive exPlanations)** explains machine learning predictions:
                    
                    - **↑ Increases Fraud Risk**: Features pointing up contribute to fraud detection
                    - **↓ Decreases Fraud Risk**: Features pointing down suggest legitimacy
                    - **Magnitude**: Larger values = stronger influence on prediction
                    - **Feature Importance**: Shows which features matter most overall
                    
                    SHAP provides **transparent, interpretable AI** - you can trust what the model decides!
                    """)
            
            except Exception as e:
                st.error(f"❌ Error displaying explainability: {e}")
    
    # ==========================================================================
    # TAB 3: BATCH ANALYSIS
    # ==========================================================================
    with tab3:
        st.markdown("---")
        
        st.markdown("### 📦 Batch Transaction Analysis")
        st.markdown("Analyze multiple transactions at once")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("📊 Load Sample Transactions", use_container_width=True):
                samples = pd.DataFrame({
                    'From Bank': [100, 500, 1000, 200, 800],
                    'To Bank': [200, 888, 1100, 300, 999],
                    'Amount Received': [5000, 50000, 25000, 10000, 100000],
                    'Amount Paid': [5100, 51000, 25500, 10500, 105000],
                    'Receiving Currency': ['US Dollar', 'Bitcoin', 'Euro', 'US Dollar', 'Bitcoin'],
                    'Payment Currency': ['US Dollar', 'Bitcoin', 'Euro', 'US Dollar', 'Ethereum'],
                    'Payment Format': ['WIRE', 'Bitcoin', 'ACH', 'WIRE', 'Bitcoin']
                })
                st.session_state.batch_data = samples
                st.success("✅ Sample transactions loaded")
        
        with col2:
            uploaded_file = st.file_uploader("📁 Or upload CSV", type=['csv'])
            if uploaded_file is not None:
                try:
                    st.session_state.batch_data = pd.read_csv(uploaded_file)
                    st.success(f"✅ Loaded {len(st.session_state.batch_data)} transactions")
                except Exception as e:
                    st.error(f"❌ Error loading file: {e}")
        
        if 'batch_data' in st.session_state and st.session_state.batch_data is not None:
            st.markdown("### 📋 Review Transactions")
            st.dataframe(st.session_state.batch_data, use_container_width=True, hide_index=True)
            
            if st.button("🔍 Analyze All Transactions", use_container_width=True):
                with st.spinner("⏳ Analyzing batch..."):
                    try:
                        batch_data = st.session_state.batch_data
                        
                        # Get predictions with high recall optimized predictor
                        predictor = ImprovedPredictor(threshold=0.05, recall_priority=True, calibrate_probabilities=False)
                        batch_results = predictor.predict_batch(batch_data)
                        predictions = np.array([r.fraud_label for r in batch_results])
                        probabilities = np.array([[1 - r.fraud_probability, r.fraud_probability] for r in batch_results])
                        
                        # Render results
                        render_batch_results(predictions, probabilities, batch_data)
                        
                        st.success("✅ Batch analysis complete!")
                    
                    except Exception as e:
                        st.error(f"❌ Error analyzing batch: {e}")
                        logging.error(f"Error in batch analysis: {e}")
        else:
            st.info("📁 Load sample transactions or upload a CSV file to begin batch analysis")
    
    # Footer
    st.markdown("---")
    theme = st.session_state.theme_mode
    colors = THEMES.get(theme, THEMES["dark"])
    footer_color = colors['text_secondary']
    st.markdown(f"""
    <div style="text-align: center; color: {footer_color}; font-size: 16px; margin-top: 24px;">
        <p>🛡️<strong> FraudShield AI</strong> • Powered by Advanced ML + SHAP Explainability</p>
        <p>© 2026 FraudShield AI • Developed by Nishant Sonar </p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
