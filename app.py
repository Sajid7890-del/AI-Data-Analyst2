import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import tempfile
from gtts import gTTS
import base64
import google.generativeai as genai
from openai import OpenAI

import config
import database
import auth
import utils
import charts
import ai_engine
import report_generator

# Page setup
st.set_page_config(
    page_title="InsightEngine AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling (Dark Mode theme with custom fonts and colors)
def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Premium Headers */
    .app-header {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
    }
    
    .app-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Styled Containers (Glassmorphism effect) */
    .glass-card {
        background-color: rgba(17, 24, 39, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }
    
    /* Sidebar styling customization */
    div[data-testid="stSidebar"] {
        background-color: #0d121f;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Buttons Customization */
    div.stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.4rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
        transition: all 0.2s ease-in-out;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.4);
    }
    
    /* Custom metric boxes */
    .metric-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #3b82f6;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Chat bubbles */
    .user-bubble {
        background-color: #1e293b;
        color: #f1f5f9;
        border-radius: 12px 12px 0 12px;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0;
        display: inline-block;
        max-width: 80%;
        float: right;
        clear: both;
    }
    
    .assistant-bubble {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
        border: 1px solid rgba(124, 58, 237, 0.2);
        color: #f1f5f9;
        border-radius: 12px 12px 12px 0;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0;
        display: inline-block;
        max-width: 80%;
        float: left;
        clear: both;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize database
database.init_db()

# Initialize session variables
auth.init_auth_session()

# Custom function for voice synthesis (TTS)
def generate_tts_player(text):
    """Generate inline HTML5 audio element for TTS."""
    clean_text = utils.sanitize_text(text) if hasattr(utils, 'sanitize_text') else text
    # Clean markdown formatting for clean reading
    clean_text = re.sub(r'[*_`#\-]', '', clean_text)
    try:
        tts = gTTS(text=clean_text[:600], lang='en') # limit to first 600 chars for speed
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        audio_bytes = fp.read()
        b64 = base64.b64encode(audio_bytes).decode()
        md = f'<audio src="data:audio/mp3;base64,{b64}" controls autoplay style="width: 100%; margin-top: 10px;"></audio>'
        return md
    except Exception as e:
        st.warning(f"Voice synthesis failed: {e}")
        return None

import re

# Transcribe Voice Input
def transcribe_audio(audio_data, provider, api_key):
    """Transcribe raw audio bytes using Gemini Multimodal or OpenAI Whisper."""
    if provider == "Gemini (Google)":
        try:
            genai_model = st.session_state.model_name
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(genai_model)
            # Gemini 1.5 Flash accepts audio bytes directly
            audio_part = {
                "mime_type": "audio/wav",
                "data": audio_data
            }
            response = model.generate_content([
                "You are an expert transcriber. Listen to the audio input and transcribe it word for word. Output ONLY the transcription and nothing else.",
                audio_part
            ])
            return response.text.strip()
        except Exception as e:
            st.error(f"Gemini transcription error: {e}")
            return None
    elif provider == "OpenAI":
        try:
            client = OpenAI(api_key=api_key)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_data)
                temp_path = f.name
                
            with open(temp_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            os.remove(temp_path)
            return transcript.text.strip()
        except Exception as e:
            st.error(f"OpenAI transcription error: {e}")
            return None
    else:
        st.info("Voice input requires a Gemini or OpenAI API Key.")
        return None

# Load dataset helper
def load_session_df():
    """Retrieve active dataset from session state or load from database path."""
    if "df" not in st.session_state or st.session_state.df is None:
        if st.session_state.current_dataset_id:
            ds = database.get_dataset_by_id(st.session_state.current_dataset_id)
            if ds and os.path.exists(ds['file_path']):
                st.session_state.df = utils.load_data(ds['file_path'])
            else:
                st.session_state.current_dataset_id = None
                st.session_state.df = None
    return st.session_state.df

# Main router
def main():
    inject_custom_css()
    
    # Header Branding
    st.markdown('<div class="app-header">InsightEngine AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Automated Data Analyst & Machine Learning Studio</div>', unsafe_allow_html=True)
    
    # Sidebar: Provider config & Auth
    with st.sidebar:
        st.title("⚙️ Operations Panel")
        
        # User details
        if st.session_state.logged_in:
            st.success(f"Logged in as: **{st.session_state.user['username']}**")
            if st.button("Logout", key="logout_btn"):
                auth.run_logout()
            st.markdown("---")
        else:
            st.info("Please login or register to analyze files.")
            
        # LLM Settings Block
        st.subheader("🤖 AI Brain Setup")
        provider = st.selectbox(
            "Select LLM Provider",
            config.AI_PROVIDERS,
            index=2 # Default to Mock
        )
        st.session_state.provider = provider
        
        api_key = ""
        model_name = ""
        
        if provider == "Gemini (Google)":
            env_key = os.getenv("GEMINI_API_KEY", "")
            api_key = st.text_input("Gemini API Key", value=env_key, type="password")
            model_name = st.selectbox("Gemini Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
        elif provider == "OpenAI":
            env_key = os.getenv("OPENAI_API_KEY", "")
            api_key = st.text_input("OpenAI API Key", value=env_key, type="password")
            model_name = st.selectbox("OpenAI Model", ["gpt-4o-mini", "gpt-4o"])
            
        st.session_state.api_key = api_key
        st.session_state.model_name = model_name
        
        st.markdown("---")
        
        # Navigation Options (Enabled only if logged in)
        if st.session_state.logged_in:
            st.subheader("🧭 Navigation")
            page = st.radio(
                "Go to Page",
                ["Dashboard & Upload", "Data Profiler & Outliers", "Interactive Auto Charts", "Machine Learning Studio", "Executive AI Chat", "Export Reports"]
            )
        else:
            page = "Auth"
            
    # Page Routing
    if not st.session_state.logged_in:
        render_auth_page()
    else:
        df = load_session_df()
        
        # If no dataset uploaded, force user to dashboard upload page first
        if df is None and page != "Dashboard & Upload":
            st.warning("Please upload or select a dataset on the Dashboard before navigating pages.")
            render_dashboard_page()
        else:
            if page == "Dashboard & Upload":
                render_dashboard_page()
            elif page == "Data Profiler & Outliers":
                render_profiler_page(df)
            elif page == "Interactive Auto Charts":
                render_charts_page(df)
            elif page == "Machine Learning Studio":
                render_ml_page(df)
            elif page == "Executive AI Chat":
                render_chat_page(df)
            elif page == "Export Reports":
                render_reports_page(df)

# --- PAGES IMPLEMENTATION ---

# Auth page
def render_auth_page():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
    
    with tab1:
        st.subheader("Login to your account")
        user_in = st.text_input("Username", key="login_user")
        pw_in = st.text_input("Password", type="password", key="login_pw")
        if st.button("Login", key="login_submit"):
            if auth.run_login_flow(user_in, pw_in):
                st.success("Successfully logged in!")
                st.rerun()
            else:
                st.error("Invalid username or password.")
                
    with tab2:
        st.subheader("Create a new account")
        reg_user = st.text_input("Username", key="reg_user")
        reg_pw = st.text_input("Password", type="password", key="reg_pw")
        reg_pw_confirm = st.text_input("Confirm Password", type="password", key="reg_pw_conf")
        
        if st.button("Register", key="reg_submit"):
            if reg_pw != reg_pw_confirm:
                st.error("Passwords do not match.")
            else:
                success, msg = auth.register_user(reg_user, reg_pw)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
    st.markdown('</div>', unsafe_allow_html=True)

# Dashboard & Upload page
def render_dashboard_page():
    st.header("📊 Dataset Management Dashboard")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📥 Upload New Dataset")
        uploaded_file = st.file_uploader("Choose CSV or Excel file", type=["csv", "xlsx", "xls"])
        
        if uploaded_file is not None:
            filename = uploaded_file.name
            if not utils.validate_file_extension(filename):
                st.error("Invalid file format. Please upload .csv, .xlsx, or .xls.")
            else:
                # Save locally to uploads/
                save_path = config.UPLOADS_DIR / f"{st.session_state.user['id']}_{filename}"
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
                # Load to read metadata
                temp_df = utils.load_data(str(save_path))
                row_count, col_count = temp_df.shape
                
                # Save metadata
                ds_id = database.save_dataset_meta(
                    st.session_state.user['id'],
                    filename,
                    str(save_path),
                    row_count,
                    col_count
                )
                st.session_state.current_dataset_id = ds_id
                st.session_state.df = temp_df
                
                # Clear chat sessions and cached insights on new upload
                st.session_state.current_session_id = None
                if "insights" in st.session_state:
                    del st.session_state.insights
                if "charts_for_pdf" in st.session_state:
                    st.session_state.charts_for_pdf = []
                    
                st.success(f"Uploaded and loaded: **{filename}**!")
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📂 Select Existing Dataset")
        datasets = database.get_datasets_by_user(st.session_state.user['id'])
        
        if datasets:
            ds_options = {d['filename']: d['id'] for d in datasets}
            selected_filename = st.selectbox("Select dataset from history", list(ds_options.keys()))
            
            if st.button("Load Dataset", key="load_ds_btn"):
                st.session_state.current_dataset_id = ds_options[selected_filename]
                st.session_state.df = None  # Will force load in load_session_df()
                # Clear chat sessions
                st.session_state.current_session_id = None
                if "insights" in st.session_state:
                    del st.session_state.insights
                if "charts_for_pdf" in st.session_state:
                    st.session_state.charts_for_pdf = []
                st.success(f"Loaded: **{selected_filename}**")
                st.rerun()
        else:
            st.write("No datasets uploaded yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Render Active Dataset details
    if st.session_state.df is not None:
        df = st.session_state.df
        ds_row = database.get_dataset_by_id(st.session_state.current_dataset_id)
        
        st.markdown(f"### Active Dataset: `{ds_row['filename']}`")
        
        # Display Basic Metrics
        info = utils.get_basic_info(df)
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{info["num_rows"]:,}</div><div class="metric-label">Rows</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{info["num_cols"]}</div><div class="metric-label">Columns</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{info["num_duplicate_rows"]}</div><div class="metric-label">Duplicate Rows</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{info["total_missing"]}</div><div class="metric-label">Missing Cells</div></div>', unsafe_allow_html=True)
        with c5:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{info["memory_usage_mb"]} MB</div><div class="metric-label">Memory</div></div>', unsafe_allow_html=True)
            
        st.ln = 10 # break line
        st.write(" ")
        
        # Cleaning and modification options
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🛠️ Data Wrangling & Cleaning Studio")
        
        tab_clean1, tab_clean2 = st.tabs(["🧹 Clean Duplicates", "🩹 Handle Missing Values"])
        
        with tab_clean1:
            if info["num_duplicate_rows"] > 0:
                st.warning(f"Your dataset contains **{info['num_duplicate_rows']}** exact duplicate rows.")
                if st.button("Drop Duplicate Rows", key="drop_dupes"):
                    cleaned_df = utils.clean_duplicates(df)
                    # Update local file
                    cleaned_df.to_csv(ds_row['file_path'], index=False) if ds_row['file_path'].endswith('.csv') else cleaned_df.to_excel(ds_row['file_path'], index=False)
                    st.session_state.df = cleaned_df
                    st.success("Successfully removed duplicate records.")
                    st.rerun()
            else:
                st.success("Perfect! No duplicate rows detected.")
                
        with tab_clean2:
            missing_report = utils.get_missing_values_report(df)
            if not missing_report.empty:
                st.write(missing_report)
                
                # Cleaning controls
                clean_col = st.selectbox("Choose column to fix", missing_report.index)
                clean_strat = st.selectbox("Imputation Strategy", ["mean", "median", "mode", "constant", "ffill", "drop"])
                
                fill_val = None
                if clean_strat == "constant":
                    fill_val = st.text_input("Value to fill")
                    
                if st.button("Apply Clean Operation"):
                    cleaned_df = utils.handle_missing_values(df, clean_col, clean_strat, fill_val)
                    # Update local file
                    cleaned_df.to_csv(ds_row['file_path'], index=False) if ds_row['file_path'].endswith('.csv') else cleaned_df.to_excel(ds_row['file_path'], index=False)
                    st.session_state.df = cleaned_df
                    st.success(f"Fixed missing values in '{clean_col}' using '{clean_strat}'.")
                    st.rerun()
            else:
                st.success("Perfect! No missing cells detected in the active dataset.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Dataset preview
        st.subheader("🔍 Table Preview")
        st.dataframe(df.head(50))

# Profiler & Outliers Page
def render_profiler_page(df):
    st.header("📊 Automated Data Profiler & Quality Metrics")
    
    col_types = utils.detect_column_types(df)
    
    tab1, tab2, tab3 = st.tabs(["📐 Descriptive Statistics", "🔗 Correlation Heatmap", "🚨 Outlier Report"])
    
    with tab1:
        st.subheader("Numerical Summary Stats")
        summary = utils.get_statistical_summary(df)
        if not summary.empty:
            st.dataframe(summary)
        else:
            st.info("No numerical fields available to describe.")
            
        st.subheader("Columns Categorisation")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write("**Numerical Columns:**")
            st.write(col_types["numeric"])
        with c2:
            st.write("**Categorical/Index Columns:**")
            st.write(col_types["categorical"])
        with c3:
            st.write("**Text/High Cardinality:**")
            st.write(col_types["text"])
            
    with tab2:
        st.subheader("Linear Correlation Matrix")
        corr = utils.get_correlation_matrix(df)
        if not corr.empty:
            fig = charts.plot_heatmap(corr)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write(corr)
        else:
            st.info("Need at least 2 numerical columns to calculate correlations.")
            
    with tab3:
        st.subheader("Interquartile Range Outlier Check")
        numeric_cols = col_types["numeric"]
        if numeric_cols:
            outlier_summary = utils.get_all_outliers_summary(df, numeric_cols)
            if outlier_summary:
                for col, info in outlier_summary.items():
                    st.error(f"🚨 Column **'{col}'** has **{info['count']} outliers** ({info['percentage']}% of dataset).")
                    
                selected_col = st.selectbox("Inspect Outliers for Column", list(outlier_summary.keys()))
                outlier_rows, _, _ = utils.detect_outliers_iqr(df, selected_col)
                st.write(f"Displaying outlier records in **{selected_col}**:")
                st.dataframe(outlier_rows)
            else:
                st.success("Awesome! No IQR outliers found in any numerical column.")
        else:
            st.info("No numerical columns found to run outlier scans.")

# Charts page
def render_charts_page(df):
    st.header("📈 Interactive Dashboard Chart Studio")
    
    col_types = utils.detect_column_types(df)
    all_cols = list(df.columns)
    numeric_cols = col_types["numeric"]
    categorical_cols = col_types["categorical"]
    
    chart_type = st.selectbox(
        "Choose Visualization Type",
        ["Bar Chart", "Line Chart", "Pie Chart", "Histogram", "Scatter Plot", "Box Plot"]
    )
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Configuration")
        x_axis = st.selectbox("X-Axis Variable", all_cols)
        
        # Configure axes based on chart type
        y_axis = None
        color_by = None
        
        if chart_type in ["Bar Chart", "Line Chart", "Scatter Plot", "Box Plot"]:
            y_axis = st.selectbox("Y-Axis Variable", numeric_cols if numeric_cols else all_cols)
            
        if chart_type in ["Bar Chart", "Line Chart", "Scatter Plot"]:
            color_opt = ["None"] + categorical_cols
            color_choice = st.selectbox("Group/Color By (Optional)", color_opt)
            color_by = None if color_choice == "None" else color_choice
            
        chart_title = st.text_input("Chart Title", f"{chart_type} of {y_axis if y_axis else x_axis} by {x_axis}")
        
        # Action button to add to report PDF list
        add_to_pdf = st.checkbox("Include in Downloadable PDF Report")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        fig = None
        static_img_bytes = None
        
        if chart_type == "Bar Chart" and y_axis:
            fig = charts.plot_bar(df, x_axis, y_axis, chart_title, color_by)
            static_img_bytes = charts.generate_static_plot(df, 'bar', x_axis, y_axis, chart_title)
        elif chart_type == "Line Chart" and y_axis:
            fig = charts.plot_line(df, x_axis, y_axis, chart_title, color_by)
            static_img_bytes = charts.generate_static_plot(df, 'line', x_axis, y_axis, chart_title)
        elif chart_type == "Pie Chart":
            # Needs numerical y_axis values
            pie_val = st.selectbox("Values Column", numeric_cols if numeric_cols else all_cols, key="pie_val_select")
            fig = charts.plot_pie(df, x_axis, pie_val, chart_title)
            static_img_bytes = charts.generate_static_plot(df, 'pie', x_axis, pie_val, chart_title)
        elif chart_type == "Histogram":
            bins_count = st.slider("Bins Count", 5, 100, 30)
            fig = charts.plot_histogram(df, x_axis, chart_title, bins_count)
            static_img_bytes = charts.generate_static_plot(df, 'histogram', x_axis, title=chart_title)
        elif chart_type == "Scatter Plot" and y_axis:
            fig = charts.plot_scatter(df, x_axis, y_axis, chart_title, color_by)
            static_img_bytes = charts.generate_static_plot(df, 'scatter', x_axis, y_axis, chart_title)
        elif chart_type == "Box Plot" and y_axis:
            fig = charts.plot_box(df, y_axis, x_axis if x_axis != y_axis else None, chart_title)
            static_img_bytes = charts.generate_static_plot(df, 'box', x_axis if x_axis != y_axis else None, y_axis, chart_title)
            
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
            # Export controls
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                if static_img_bytes:
                    st.download_button(
                        label="💾 Download Chart (PNG)",
                        data=static_img_bytes,
                        file_name=f"{chart_type.lower().replace(' ', '_')}.png",
                        mime="image/png"
                    )
            with col_dl2:
                if add_to_pdf and static_img_bytes:
                    if "charts_for_pdf" not in st.session_state:
                        st.session_state.charts_for_pdf = []
                    # Avoid duplicates
                    if static_img_bytes not in st.session_state.charts_for_pdf:
                        st.session_state.charts_for_pdf.append(static_img_bytes)
                        st.toast(f"Added chart to PDF queue! ({len(st.session_state.charts_for_pdf)} total)")

# Machine Learning Studio
def render_ml_page(df):
    st.header("🤖 Machine Learning Prediction & Analytics Studio")
    
    col_types = utils.detect_column_types(df)
    numeric_cols = col_types["numeric"]
    all_cols = list(df.columns)
    
    if len(numeric_cols) < 2:
        st.warning("Machine learning models require at least two numerical columns to run analyses.")
        return
        
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    task_type = st.radio("ML Goal", ["Regression (Sales/Continuous Forecasts)", "Classification (Group/Category Predictions)"])
    
    col_y, col_x = st.columns(2)
    
    with col_y:
        target_variable = st.selectbox("Select Target Column (Y)", all_cols)
    with col_x:
        feature_options = [c for c in all_cols if c != target_variable]
        feature_variables = st.multiselect("Select Feature Columns (X)", feature_options, default=feature_options[:3] if len(feature_options) >=3 else feature_options)
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("Train Predictive Model"):
        if not feature_variables:
            st.error("Please select at least one feature column (X).")
            return
            
        with st.spinner("Training predictive models and computing accuracy..."):
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder
            from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, classification_report
            from sklearn.linear_model import LinearRegression
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            
            # Prepare data
            df_ml = df[[target_variable] + feature_variables].dropna().copy()
            
            if df_ml.empty:
                st.error("The dataset contains empty values in the selected columns. Please impute or drop missing values.")
                return
                
            # Encode categorical features
            label_encoders = {}
            for col in df_ml.columns:
                if df_ml[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df_ml[col]):
                    le = LabelEncoder()
                    df_ml[col] = le.fit_transform(df_ml[col].astype(str))
                    label_encoders[col] = le
                    
            X = df_ml[feature_variables]
            y = df_ml[target_variable]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            st.subheader("📊 Model Performance Report")
            
            if task_type.startswith("Regression"):
                # Linear Regression
                lr = LinearRegression()
                lr.fit(X_train, y_train)
                y_pred_lr = lr.predict(X_test)
                
                # Random Forest
                rf = RandomForestRegressor(random_state=42)
                rf.fit(X_train, y_train)
                y_pred_rf = rf.predict(X_test)
                
                # Calculate scores
                r2_lr = r2_score(y_test, y_pred_lr)
                mae_lr = mean_absolute_error(y_test, y_pred_lr)
                
                r2_rf = r2_score(y_test, y_pred_rf)
                mae_rf = mean_absolute_error(y_test, y_pred_rf)
                
                c_lr, c_rf = st.columns(2)
                with c_lr:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{r2_lr:.3f}</div><div class="metric-label">Linear Regression R²</div></div>', unsafe_allow_html=True)
                    st.write(f"Mean Absolute Error (MAE): **{mae_lr:,.2f}**")
                with c_rf:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{r2_rf:.3f}</div><div class="metric-label">Random Forest R²</div></div>', unsafe_allow_html=True)
                    st.write(f"Mean Absolute Error (MAE): **{mae_rf:,.2f}**")
                    
                # Feature importance (Random Forest)
                importances = rf.feature_importances_
                importance_df = pd.DataFrame({"Feature": feature_variables, "Importance": importances}).sort_values(by="Importance", ascending=True)
                fig = px.bar(importance_df, x="Importance", y="Feature", orientation="h", title="Random Forest Feature Importance")
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                # Check target cardinality
                if y.nunique() > 50:
                    st.error("Classification models require a target variable (Y) with high cardinality/few classes. Your target has over 50 classes. Choose a regression task instead.")
                    return
                    
                # Classifier
                rfc = RandomForestClassifier(random_state=42)
                rfc.fit(X_train, y_train)
                y_pred_rfc = rfc.predict(X_test)
                
                acc = accuracy_score(y_test, y_pred_rfc)
                
                st.markdown(f'<div class="metric-card"><div class="metric-value">{acc*100:.2f}%</div><div class="metric-label">Random Forest Classifier Accuracy</div></div>', unsafe_allow_html=True)
                
                st.write("**Classification Report:**")
                report_dict = classification_report(y_test, y_pred_rfc, output_dict=True)
                st.dataframe(pd.DataFrame(report_dict).T)
                
                # Feature importance
                importances = rfc.feature_importances_
                importance_df = pd.DataFrame({"Feature": feature_variables, "Importance": importances}).sort_values(by="Importance", ascending=True)
                fig = px.bar(importance_df, x="Importance", y="Feature", orientation="h", title="Feature Importance")
                st.plotly_chart(fig, use_container_width=True)

# Executive AI Chat Page
def render_chat_page(df):
    st.header("💬 Executive Business AI Assistant")
    
    # Session Management in database
    sessions = database.get_chat_sessions(st.session_state.user['id'], st.session_state.current_dataset_id)
    
    col_sess, col_chat = st.columns([1, 3])
    
    with col_sess:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Chat Sessions")
        
        # Form to add new session
        new_sess_name = st.text_input("New Session Name", placeholder="Sales Q&A")
        if st.button("Create Session", key="create_sess"):
            if new_sess_name.strip():
                sess_id = database.create_chat_session(st.session_state.user['id'], st.session_state.current_dataset_id, new_sess_name.strip())
                st.session_state.current_session_id = sess_id
                st.rerun()
                
        # Select active session
        if sessions:
            sess_options = {s['name']: s['id'] for s in sessions}
            active_name = list(sess_options.keys())[0]
            
            # Find active index
            active_idx = 0
            if st.session_state.current_session_id:
                for idx, s in enumerate(sessions):
                    if s['id'] == st.session_state.current_session_id:
                        active_idx = idx
                        break
                        
            selected_sess = st.selectbox("Active Session", list(sess_options.keys()), index=active_idx)
            st.session_state.current_session_id = sess_options[selected_sess]
            
            # Delete active session
            if st.button("Delete Active Session", key="del_sess_btn"):
                database.delete_chat_session(st.session_state.current_session_id)
                st.session_state.current_session_id = None
                st.success("Session deleted.")
                st.rerun()
        else:
            st.info("No active chat sessions. Create one above.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_chat:
        if not st.session_state.current_session_id:
            st.info("Please select or create a chat session in the sidebar to start asking questions.")
            return
            
        # Display Message Log
        history = database.get_chat_history(st.session_state.current_session_id)
        
        st.markdown("### Chat Log")
        chat_container = st.container(height=400)
        
        with chat_container:
            for msg in history:
                role = msg['role']
                content = msg['content']
                
                if role == 'user':
                    st.markdown(f'<div class="user-bubble">🧑‍💻 <b>You:</b><br>{content}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="assistant-bubble">🤖 <b>InsightEngine:</b><br>{content}</div>', unsafe_allow_html=True)
                    
        # Voice Input Panel
        st.write(" ")
        st.markdown("**🎙️ Voice Control (Bonus)**")
        audio_val = st.audio_input("Record your question")
        
        voice_query = None
        if audio_val is not None:
            audio_bytes = audio_val.read()
            with st.spinner("Transcribing speech..."):
                voice_query = transcribe_audio(audio_bytes, st.session_state.provider, st.session_state.api_key)
                if voice_query:
                    st.success(f"Transcribed: *\"{voice_query}\"*")
                    
        # Chat input box
        input_placeholder = "Ask something (e.g. Which product sold the most? or predict next month sales)"
        if voice_query:
            # Pre-populate voice query
            user_input = st.text_input("Edit question & submit", value=voice_query)
        else:
            user_input = st.text_input("Ask a question about the dataset", placeholder=input_placeholder)
            
        if st.button("Send Query", key="send_chat_btn") and user_input.strip():
            # Save User Message
            database.save_chat_message(st.session_state.current_session_id, 'user', user_input.strip())
            
            # Call AI Analyst engine
            with st.spinner("InsightEngine is analyzing dataset..."):
                response_text, fig, code_run = ai_engine.query_dataset(
                    df,
                    user_input.strip(),
                    st.session_state.provider,
                    st.session_state.api_key,
                    st.session_state.model_name
                )
                
            # Save Assistant Response
            database.save_chat_message(st.session_state.current_session_id, 'assistant', response_text)
            
            # Log Analysis Activity in DB
            database.log_analysis(st.session_state.user['id'], st.session_state.current_dataset_id, "AI Chat Query", f"Question: {user_input[:40]}")
            
            st.rerun()
            
        # Display the most recent Assistant response's TTS play option
        if history:
            last_msg = history[-1]
            if last_msg['role'] == 'assistant':
                st.write(" ")
                st.markdown("**🔊 Voice Summary (Bonus)**")
                if st.button("Play Audio Summary"):
                    audio_html = generate_tts_player(last_msg['content'])
                    if audio_html:
                        st.markdown(audio_html, unsafe_allow_html=True)

# Export Reports Page
def render_reports_page(df):
    st.header("📋 Automated Executive PDF & Excel Reporting Studio")
    
    col_rep, col_act = st.columns([2, 1])
    
    with col_rep:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("💡 Business Insights Generation")
        
        # Check cache
        if "insights" not in st.session_state:
            if st.button("Generate Strategic Business Report"):
                with st.spinner("AI is analyzing column relations and structural trends..."):
                    st.session_state.insights = ai_engine.generate_business_insights(
                        df,
                        st.session_state.provider,
                        st.session_state.api_key,
                        st.session_state.model_name
                    )
                    st.rerun()
        
        if "insights" in st.session_state:
            st.markdown(st.session_state.insights)
            if st.button("Regenerate Insights", key="regen_ins"):
                del st.session_state.insights
                st.rerun()
        else:
            st.info("Click the button above to generate a full business analysis report.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_act:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📥 Download Center")
        
        # 1. Excel Report
        with st.spinner("Preparing Excel report..."):
            summary_stats = utils.get_statistical_summary(df)
            corr_matrix = utils.get_correlation_matrix(df)
            excel_bytes = report_generator.generate_excel_report(df, summary_stats, corr_matrix)
            
        st.download_button(
            label="📊 Download Excel Summary Report",
            data=excel_bytes,
            file_name="analytical_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        st.write(" ")
        
        # 2. PDF Report
        if "insights" in st.session_state:
            basic_info = utils.get_basic_info(df)
            pdf_charts = st.session_state.get("charts_for_pdf", [])
            
            with st.spinner("Generating styled PDF report..."):
                pdf_bytes = report_generator.generate_pdf_report(
                    df,
                    basic_info,
                    summary_stats,
                    st.session_state.insights,
                    pdf_charts
                )
                
            st.download_button(
                label="📄 Download Executive PDF Report",
                data=pdf_bytes,
                file_name="executive_summary.pdf",
                mime="application/pdf"
            )
            
            if pdf_charts:
                st.caption(f"✓ Embedded {len(pdf_charts)} queued chart(s) from Visual Studio.")
            else:
                st.caption("💡 Hint: Go to the 'Interactive Auto Charts' page and check 'Include in Downloadable PDF Report' to embed charts in the PDF.")
        else:
            st.warning("Please generate the Strategic Business Report first before downloading the PDF.")
            
        st.write(" ")
        
        # 3. CSV Clean Data
        csv_bytes = report_generator.generate_csv_report(df)
        st.download_button(
            label="📝 Download Clean CSV Dataset",
            data=csv_bytes,
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
