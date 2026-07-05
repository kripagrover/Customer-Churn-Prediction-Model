import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from predict import predict_churn, model_files_exist

# Page configuration
st.set_page_config(
    page_title="Telco Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    /* CSS Glassmorphism & custom variables */
    :root {
        --primary: #6366f1;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 2rem;
    }
    
    .card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #334155;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin-top: 5px;
    }
    
    .badge {
        padding: 6px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.875rem;
        display: inline-block;
    }
    
    .badge-low {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .badge-medium {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .badge-high {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .factor-item {
        display: flex;
        align-items: center;
        margin-bottom: 8px;
        padding: 8px 12px;
        background-color: rgba(255, 255, 255, 0.03);
        border-left: 3px solid #6366f1;
        border-radius: 4px;
    }
    
    .rec-item {
        margin-bottom: 12px;
        padding: 10px 14px;
        background-color: rgba(16, 185, 129, 0.05);
        border-left: 3px solid #10b981;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Sample Profiles
SAMPLE_PROFILES = {
    "Select a Profile...": None,
    "Low Churn Risk (Loyal, Fiber, Auto-pay)": {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
        "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes", "InternetService": "DSL",
        "OnlineSecurity": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "Yes",
        "TechSupport": "Yes", "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "Contract": "Two year", "PaperlessBilling": "No", "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 85.0, "TotalCharges": 5100.0
    },
    "High Churn Risk (Month-to-month, Fiber, Electronic check)": {
        "gender": "Male", "SeniorCitizen": 1, "Partner": "No", "Dependents": "No",
        "tenure": 3, "PhoneService": "Yes", "MultipleLines": "Yes", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 90.0, "TotalCharges": 270.0
    },
    "Average Churn Risk (1 Year Contract, DSL)": {
        "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 24, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL",
        "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "One year", "PaperlessBilling": "Yes", "PaymentMethod": "Mailed check",
        "MonthlyCharges": 45.0, "TotalCharges": 1080.0
    }
}

def render_gauge(probability):
    """Render a beautiful gauge chart for churn probability."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 36, "color": "#f8fafc"}},
        title={"text": "Churn Probability", "font": {"size": 18, "color": "#94a3b8"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
            "bar": {"color": "#6366f1"},
            "bgcolor": "#1e293b",
            "borderwidth": 2,
            "bordercolor": "#334155",
            "steps": [
                {"range": [0, 40], "color": "rgba(16, 185, 129, 0.2)"},
                {"range": [40, 70], "color": "rgba(245, 158, 11, 0.2)"},
                {"range": [70, 100], "color": "rgba(239, 68, 68, 0.2)"}
            ],
            "threshold": {
                "line": {"color": "#ef4444", "width": 4},
                "thickness": 0.75,
                "value": probability * 100
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f8fafc"},
        margin=dict(l=20, r=20, t=40, b=20),
        height=250
    )
    return fig

# Main header
st.markdown("<h1 class='main-title'>Customer Churn Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Predict customer retention risk and generate actionable retention recommendations in real-time.</p>", unsafe_allow_html=True)

if not model_files_exist():
    st.error("🚨 Trained model files not found! Make sure you run the ML training pipeline (`customer_churn_prediction_model_using_ml.py`) to generate `customer_churn_model.pkl` and `encoders.pkl`.")
    st.stop()

# Dropdown to load profiles
st.sidebar.markdown("### Profile Presets")
selected_preset = st.sidebar.selectbox("Load Sample Customer:", list(SAMPLE_PROFILES.keys()))

# Manage session state for presets
if selected_preset != "Select a Profile..." and SAMPLE_PROFILES[selected_preset] is not None:
    preset_data = SAMPLE_PROFILES[selected_preset]
    for key, value in preset_data.items():
        st.session_state[f"input_{key}"] = value

# Tabs
tab_single, tab_batch = st.tabs(["🔮 Single Customer Prediction", "📂 Batch CSV Prediction"])

with tab_single:
    col_input, col_result = st.columns([3, 2])
    
    with col_input:
        st.subheader("Customer Characteristics")
        
        tab_demo, tab_serv, tab_bill = st.tabs(["👥 Demographics", "🔌 Services", "💳 Billing & Contract"])
        
        with tab_demo:
            gender = st.selectbox(
                "Gender", ["Female", "Male"], 
                key="input_gender", 
                index=0 if st.session_state.get("input_gender") == "Female" else 1
            )
            senior_citizen = st.radio(
                "Senior Citizen", [0, 1], 
                format_func=lambda x: "Yes" if x == 1 else "No",
                key="input_SeniorCitizen",
                index=st.session_state.get("input_SeniorCitizen", 0)
            )
            partner = st.selectbox(
                "Partner", ["No", "Yes"], 
                key="input_Partner",
                index=1 if st.session_state.get("input_Partner") == "Yes" else 0
            )
            dependents = st.selectbox(
                "Dependents", ["No", "Yes"], 
                key="input_Dependents",
                index=1 if st.session_state.get("input_Dependents") == "Yes" else 0
            )
            
        with tab_serv:
            phone_service = st.selectbox(
                "Phone Service", ["No", "Yes"], 
                key="input_PhoneService",
                index=1 if st.session_state.get("input_PhoneService") == "Yes" else 0
            )
            
            # Conditionally disable MultipleLines based on Phone Service
            if phone_service == "No":
                multiple_lines = "No phone service"
                st.session_state["input_MultipleLines"] = "No phone service"
                st.caption("Multiple Lines: No phone service (Phone Service disabled)")
            else:
                current_ml = st.session_state.get("input_MultipleLines", "No")
                if current_ml == "No phone service":
                    current_ml = "No"
                multiple_lines = st.selectbox(
                    "Multiple Lines", ["No", "Yes"], 
                    key="input_MultipleLines",
                    index=1 if current_ml == "Yes" else 0
                )
                
            internet_service = st.selectbox(
                "Internet Service Provider", ["DSL", "Fiber optic", "No"], 
                key="input_InternetService",
                index=["DSL", "Fiber optic", "No"].index(st.session_state.get("input_InternetService", "DSL"))
            )
            
            if internet_service == "No":
                st.caption("Other Internet services are set to 'No internet service'")
                online_security = "No internet service"
                online_backup = "No internet service"
                device_protection = "No internet service"
                tech_support = "No internet service"
                streaming_tv = "No internet service"
                streaming_movies = "No internet service"
            else:
                # Security and backup
                cols_s1, cols_s2 = st.columns(2)
                with cols_s1:
                    current_os = st.session_state.get("input_OnlineSecurity", "No")
                    if current_os == "No internet service": current_os = "No"
                    online_security = st.selectbox(
                        "Online Security", ["No", "Yes"], 
                        key="input_OnlineSecurity",
                        index=1 if current_os == "Yes" else 0
                    )
                    
                    current_ob = st.session_state.get("input_OnlineBackup", "No")
                    if current_ob == "No internet service": current_ob = "No"
                    online_backup = st.selectbox(
                        "Online Backup", ["No", "Yes"], 
                        key="input_OnlineBackup",
                        index=1 if current_ob == "Yes" else 0
                    )
                with cols_s2:
                    current_dp = st.session_state.get("input_DeviceProtection", "No")
                    if current_dp == "No internet service": current_dp = "No"
                    device_protection = st.selectbox(
                        "Device Protection", ["No", "Yes"], 
                        key="input_DeviceProtection",
                        index=1 if current_dp == "Yes" else 0
                    )
                    
                    current_ts = st.session_state.get("input_TechSupport", "No")
                    if current_ts == "No internet service": current_ts = "No"
                    tech_support = st.selectbox(
                        "Tech Support", ["No", "Yes"], 
                        key="input_TechSupport",
                        index=1 if current_ts == "Yes" else 0
                    )
                
                # Streaming services
                cols_st1, cols_st2 = st.columns(2)
                with cols_st1:
                    current_st = st.session_state.get("input_StreamingTV", "No")
                    if current_st == "No internet service": current_st = "No"
                    streaming_tv = st.selectbox(
                        "Streaming TV", ["No", "Yes"], 
                        key="input_StreamingTV",
                        index=1 if current_st == "Yes" else 0
                    )
                with cols_st2:
                    current_sm = st.session_state.get("input_StreamingMovies", "No")
                    if current_sm == "No internet service": current_sm = "No"
                    streaming_movies = st.selectbox(
                        "Streaming Movies", ["No", "Yes"], 
                        key="input_StreamingMovies",
                        index=1 if current_sm == "Yes" else 0
                    )

        with tab_bill:
            tenure = st.slider(
                "Customer Tenure (Months)", 0, 72, 
                key="input_tenure",
                value=int(st.session_state.get("input_tenure", 12))
            )
            contract = st.selectbox(
                "Contract Type", ["Month-to-month", "One year", "Two year"], 
                key="input_Contract",
                index=["Month-to-month", "One year", "Two year"].index(st.session_state.get("input_Contract", "Month-to-month"))
            )
            paperless_billing = st.selectbox(
                "Paperless Billing", ["No", "Yes"], 
                key="input_PaperlessBilling",
                index=1 if st.session_state.get("input_PaperlessBilling") == "Yes" else 0
            )
            payment_method = st.selectbox(
                "Payment Method", 
                ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"],
                key="input_PaymentMethod",
                index=["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"].index(
                    st.session_state.get("input_PaymentMethod", "Electronic check")
                )
            )
            monthly_charges = st.number_input(
                "Monthly Charges ($)", 0.0, 150.0, 
                key="input_MonthlyCharges",
                value=float(st.session_state.get("input_MonthlyCharges", 50.0)),
                step=1.0
            )
            
            # Simple automatic helper for TotalCharges calculation
            auto_total = tenure * monthly_charges
            total_charges = st.number_input(
                "Total Charges ($)", 0.0, 9000.0, 
                key="input_TotalCharges",
                value=float(st.session_state.get("input_TotalCharges", auto_total)),
                step=10.0
            )
            
    with col_result:
        st.subheader("Prediction Results")
        
        # Prepare data dict
        input_data = {
            "gender": gender,
            "SeniorCitizen": int(senior_citizen),
            "Partner": partner,
            "Dependents": dependents,
            "tenure": int(tenure),
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": float(monthly_charges),
            "TotalCharges": float(total_charges)
        }
        
        # Run prediction
        res = predict_churn(input_data)
        
        prob_churn = res["prob_churn"]
        risk_level = res["risk_level"]
        
        # Select badge class based on risk level
        badge_cls = "badge-low"
        if risk_level == "Medium":
            badge_cls = "badge-medium"
        elif risk_level == "High":
            badge_cls = "badge-high"
            
        st.markdown(f"""
        <div class="card">
            <h3>Prediction: <span class="badge {badge_cls}">{res["label"]}</span></h3>
            <p>Risk Tier: <strong>{risk_level} Risk</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Gauge representation
        st.plotly_chart(render_gauge(prob_churn), use_container_width=True)
        
        # Risk Factor breakdown & explanation
        st.subheader("Key Risk Drivers")
        factors = []
        if contract == "Month-to-month":
            factors.append("<strong>Month-to-month Contract</strong>: Significantly increases attrition likelihood.")
        if tenure < 6:
            factors.append("<strong>New Customer (Short Tenure)</strong>: Very high risk of early churn.")
        if internet_service == "Fiber optic" and monthly_charges > 80:
            factors.append("<strong>High Fiber Optic Cost</strong>: Premium pricing increases price sensitivity.")
        if online_security == "No" and internet_service != "No":
            factors.append("<strong>No Online Security</strong>: Higher churn due to lack of security add-ons.")
        if tech_support == "No" and internet_service != "No":
            factors.append("<strong>No Tech Support</strong>: Missing troubleshooting assistance.")
        if payment_method == "Electronic check":
            factors.append("<strong>Electronic Check Payment</strong>: Historically correlates with elevated churn.")
            
        if not factors:
            factors.append("No critical risk drivers identified. Customer shows typical attributes of a loyal subscriber.")
            
        for f in factors:
            st.markdown(f'<div class="factor-item">{f}</div>', unsafe_allow_html=True)
            
        # Actionable recommendations
        st.subheader("Actionable Recommendations")
        recs = []
        if contract == "Month-to-month":
            recs.append("<strong>Offer Contract Upgrade</strong>: Promote a 1-year or 2-year subscription with a small incentive (e.g. 10% discount).")
        if online_security == "No" and internet_service != "No":
            recs.append("<strong>Cross-sell Security Bundle</strong>: Offer 3 months of complimentary Online Security service.")
        if tech_support == "No" and internet_service != "No":
            recs.append("<strong>Cross-sell Support Services</strong>: Pitch dedicated Tech Support to improve service stability.")
        if payment_method == "Electronic check":
            recs.append("<strong>Promote Autopay Signup</strong>: Offer a $5 bill credit for setting up automatic payment (credit card or bank transfer).")
        if tenure < 12 and monthly_charges > 70:
            recs.append("<strong>Introductory Loyalty Discount</strong>: Propose a temporary discount to sustain retention through the critical first year.")
            
        if not recs:
            recs.append("No primary intervention needed. Continue regular billing cycle updates.")
            
        for r in recs:
            st.markdown(f'<div class="rec-item">{r}</div>', unsafe_allow_html=True)

with tab_batch:
    st.subheader("Batch Customer Prediction")
    st.markdown("Upload a CSV file containing customer profiles to run batch predictions and export the result.")
    
    # Download template example
    st.markdown("##### CSV Formatting Reference")
    example_cols = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", 
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", 
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", 
        "PaymentMethod", "MonthlyCharges", "TotalCharges"
    ]
    template_df = pd.DataFrame([
        SAMPLE_PROFILES["Low Churn Risk (Loyal, Fiber, Auto-pay)"],
        SAMPLE_PROFILES["High Churn Risk (Month-to-month, Fiber, Electronic check)"]
    ])
    
    st.dataframe(template_df[example_cols], height=120)
    
    csv_file = st.file_uploader("Upload CSV File:", type=["csv"])
    
    if csv_file is not None:
        try:
            df_upload = pd.read_csv(csv_file)
            
            # Check if all required columns are present
            missing_cols = [col for col in example_cols if col not in df_upload.columns]
            
            if missing_cols:
                st.error(f"❌ Uploaded CSV is missing the following required columns: {', '.join(missing_cols)}")
            else:
                # Predict
                predictions = []
                prob_no_churns = []
                prob_churns = []
                risk_levels = []
                labels = []
                
                # Perform prediction on row by row
                with st.spinner("Processing batch predictions..."):
                    for _, row in df_upload.iterrows():
                        cust_dict = row[example_cols].to_dict()
                        # Ensure correct types
                        cust_dict["SeniorCitizen"] = int(cust_dict["SeniorCitizen"])
                        cust_dict["tenure"] = int(cust_dict["tenure"])
                        cust_dict["MonthlyCharges"] = float(cust_dict["MonthlyCharges"])
                        cust_dict["TotalCharges"] = float(cust_dict["TotalCharges"])
                        
                        pred_res = predict_churn(cust_dict)
                        predictions.append(pred_res["prediction"])
                        prob_no_churns.append(pred_res["prob_no_churn"])
                        prob_churns.append(pred_res["prob_churn"])
                        risk_levels.append(pred_res["risk_level"])
                        labels.append(pred_res["label"])
                
                # Append outputs
                df_results = df_upload.copy()
                df_results["Prediction_Label"] = labels
                df_results["Prediction_Value"] = predictions
                df_results["Churn_Probability"] = prob_churns
                df_results["Risk_Level"] = risk_levels
                
                # Show summary widgets
                total_custs = len(df_results)
                churn_custs = sum(predictions)
                churn_rate = (churn_custs / total_custs) * 100
                
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.markdown(f"""
                    <div class="card">
                        <p style="color: #94a3b8; font-size: 0.9rem; text-transform: uppercase;">Total Customers Evaluated</p>
                        <div class="metric-value">{total_custs}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_m2:
                    st.markdown(f"""
                    <div class="card">
                        <p style="color: #94a3b8; font-size: 0.9rem; text-transform: uppercase;">Predicted Churn Count</p>
                        <div class="metric-value" style="color: #ef4444;">{churn_custs}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_m3:
                    st.markdown(f"""
                    <div class="card">
                        <p style="color: #94a3b8; font-size: 0.9rem; text-transform: uppercase;">Expected Churn Rate</p>
                        <div class="metric-value" style="color: #fbbf24;">{churn_rate:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show charts
                col_chart, col_empty = st.columns([3, 2])
                with col_chart:
                    risk_counts = pd.Series(risk_levels).value_counts()
                    risk_order = ["Low", "Medium", "High"]
                    ordered_counts = [risk_counts.get(lvl, 0) for lvl in risk_order]
                    
                    fig_bar = go.Figure(data=[go.Bar(
                        x=risk_order,
                        y=ordered_counts,
                        marker_color=["#10b981", "#f59e0b", "#ef4444"]
                    )])
                    fig_bar.update_layout(
                        title="Distribution of Churn Risk Tiers",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font={"color": "#f8fafc"},
                        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                        height=300
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                st.subheader("Predictions Table")
                st.dataframe(df_results, height=350)
                
                # Download predictions CSV button
                csv_data = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=csv_data,
                    file_name="churn_predictions_export.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"❌ Error parsing CSV file: {str(e)}")
