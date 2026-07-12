# InsightEngine AI - Interactive AI Data Analyst & ML Studio

InsightEngine AI is a production-ready, highly aesthetic web application built with Streamlit and Python. It enables users to upload datasets (CSV/Excel), perform automated profiling, clean data, create interactive charts, build machine learning models, and converse with an AI analyst (supporting Google Gemini, OpenAI, or a Smart Mock Engine) using text or voice input.

## Features

- **🔑 Secure Authentication**: Hashed password storage with PBKDF2 salting.
- **📥 Dataset Management**: Upload CSV/XLSX files, preview tables, detect duplicate rows, and clean missing values.
- **📊 Auto Profiler & Quality Metrics**: Descriptive statistics, outlier checks using IQR, and interactive correlation matrices.
- **📈 Interactive Chart Studio**: Instantly generate and customize Bar, Line, Pie, Scatter, Box, and Histogram charts. Export charts as PNG.
- **🤖 Predictive ML Studio**: Train classification or regression models dynamically using Random Forest and Linear Regression, with feature importance plots and test evaluations.
- **💬 Executive AI Chat (Text & Voice)**: Ask natural language questions. The AI writes safe Python code, queries the database, displays outputs, and provides a voice readout of the analysis.
- **📋 Branded PDF & Excel Exports**: Download formatted multi-sheet Excel reports and professional corporate PDF reports with embedded custom charts and executive summaries.

---

## Folder Structure

```text
ai-data-analyst/
├── app.py                  # Main Streamlit dashboard interface
├── config.py               # Application paths & global configuration
├── database.py             # SQLite relational schema & operations
├── auth.py                 # User authentication (salting/hashing)
├── utils.py                # Data profiling, typing, & cleaning helpers
├── charts.py               # Visualizations (Plotly & Matplotlib)
├── ai_engine.py            # AI Query router (Gemini / OpenAI / Mock)
├── report_generator.py     # Multi-sheet Excel & styled PDF exporters
├── requirements.txt        # Package requirements
├── README.md               # User guide & developer instructions
├── uploads/                # Temporary directory for uploaded datasets
├── reports/                # Cache directory for report builds
├── database/               # Relational SQLite database storage
├── assets/                 # Brand assets & layout styles
└── models/                 # Cached serialized machine learning models
```

---

## Local Setup

### 1. Prerequisites
Ensure you have **Python 3.9+** installed on your machine.

### 2. Clone and Navigate
Clone this repository and open the project directory:
```bash
git clone https://github.com/yourusername/ai-data-analyst.git
cd ai-data-analyst
```

### 3. Install Dependencies
Install all required libraries using `pip`:
```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables (Optional)
If you wish to use Google Gemini or OpenAI APIs without typing the keys manually inside the UI, create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Launch the Application
Run the Streamlit server:
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## Deployment Instructions

### 1. Streamlit Community Cloud (Recommended)
Streamlit Cloud offers free, fast hosting for Streamlit applications:
1. Push your project to a public repository on **GitHub**.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **New app**, select your repository, branch, and set the entry file to `app.py`.
4. (Optional) Under **Advanced settings**, add your `GEMINI_API_KEY` or `OPENAI_API_KEY` as Secrets.
5. Click **Deploy!**

### 2. Render
To deploy on Render as a Web Service:
1. Create a `render.yaml` or define a Web Service connected to your GitHub repository.
2. Set the Environment to **Python**.
3. Set the Build Command: `pip install -r requirements.txt`
4. Set the Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Under Environment variables, add key/value configurations if needed.
6. Click **Deploy**.

---

## Technology Stack

- **Frontend**: Streamlit
- **Data Engine**: Pandas, NumPy, OpenPyXL
- **Visuals**: Plotly Express, Matplotlib, Seaborn
- **Storage**: SQLite3
- **AI Brain**: Google Gemini API, OpenAI API
- **Reporting**: FPDF2, OpenPyXL Exporter
- **TTS/Voice**: gTTS, Streamlit Audio Input
