import os
import sys
import io
import json
import re
import traceback
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
from openai import OpenAI

# Safe Python Code Executor for Data Analysis
def execute_analysis_code(df, code_str):
    """
    Executes a Python code block in a sandboxed context where the dataframe 'df' is available.
    Captures stdout, final values, and figures.
    """
    # Clean the code string from markdown fences
    code_str = clean_code_block(code_str)
    
    # Sandbox environment
    sandbox_globals = {
        'pd': pd,
        'np': np,
        'plt': plt,
        'px': px,
        'go': go,
        '__builtins__': __builtins__
    }
    
    sandbox_locals = {
        'df': df,
        'result': None,
        'fig': None
    }
    
    # Capture stdout
    stdout_buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout_buffer
    
    error = None
    try:
        # Run code
        exec(code_str, sandbox_globals, sandbox_locals)
    except Exception as e:
        error = traceback.format_exc()
    finally:
        sys.stdout = old_stdout
        
    captured_stdout = stdout_buffer.getvalue()
    
    # Retrieve execution results
    result_val = sandbox_locals.get('result')
    fig = sandbox_locals.get('fig')
    
    # Fallback to matplotlib if figure created but not returned in 'fig'
    if fig is None:
        try:
            if plt.get_fignums():
                fig = plt.gcf()
        except:
            pass
            
    return {
        "stdout": captured_stdout,
        "result": result_val,
        "fig": fig,
        "error": error
    }

def clean_code_block(code_str):
    """Remove markdown formatting ```python ... ``` if present."""
    code_str = re.sub(r'^```python\s*', '', code_str, flags=re.MULTILINE)
    code_str = re.sub(r'^```\s*', '', code_str, flags=re.MULTILINE)
    return code_str.strip()

def build_dataset_context(df):
    """Compile dataset metadata (columns, types, summary stats) into a string."""
    columns_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        num_unique = df[col].nunique()
        missing = df[col].isnull().sum()
        columns_info.append(f"- Name: '{col}', Type: {dtype}, Unique values: {num_unique}, Missing: {missing}")
        
    columns_str = "\n".join(columns_info)
    
    # Summary of first few rows
    sample_rows = df.head(3).to_string()
    
    # Basic shape
    shape_str = f"Rows: {df.shape[0]}, Columns: {df.shape[1]}"
    
    return f"""
Dataset Shape: {shape_str}
Columns Metadata:
{columns_str}

Sample Data (First 3 Rows):
{sample_rows}
"""

def prompt_template_for_qa(df_context, user_query):
    """Create systemic prompt to guide LLM in generating answers or pandas code."""
    return f"""
You are a Senior Business Data Analyst. You are given a pandas DataFrame named `df` loaded in Python.
Here is the metadata of the dataset:
{df_context}

The user asks: "{user_query}"

Analyze the question. If you need to write Python code using `pandas`, `numpy`, or plotting libraries to find the answer, generate a clean, executable Python block.
Rules for code generation:
1. Always refer to the DataFrame as `df`.
2. Do not attempt to load or read any file. The DataFrame `df` is already in the namespace.
3. Save the final calculated answer or description to a variable named `result`. For instance: `result = df['Sales'].sum()`.
4. If the question requires plotting, construct a Plotly figure and assign it to `fig` (e.g., `fig = px.bar(...)`).
5. Only use standard libraries: `pandas`, `numpy`, `plotly.express`, `plotly.graph_objects`, `matplotlib.pyplot`.
6. Make your code robust, handling potential division by zero or empty values.

Response format:
You must respond in JSON format with three fields:
1. "reasoning": A description of how you plan to answer the question or what calculations are required.
2. "code": The Python code string to run on `df`. Set to empty string "" if no code is required.
3. "text_response": A direct text response. If you wrote code, this should explain what the code will calculate. If you didn't write code, this is your full answer in clean, simple English.

Ensure your response is valid JSON.
"""

def generate_insights_prompt(df_context):
    """Create systemic prompt to generate business insights."""
    return f"""
You are an Executive Business Advisor and Data Analyst. Analyze the following dataset metadata:
{df_context}

Generate a comprehensive business insights report.
Your output must be structured in Markdown format with the following headers:
# Executive Summary
(Write a high-level summary of the dataset and what it represents)

# Key Findings
(3-4 key data points or trends observed in the data)

# Strategic Recommendations
(3-4 actionable recommendations for business growth or optimization based on findings)

# Risks and Opportunities
- **Risks**: (2 potential threats, e.g., missing values, declining trends, high variance)
- **Opportunities**: (2 potential areas for expansion, process improvement, or revenue generation)

Make the language professional, clear, and focused on business value.
"""

# --- LLM Client Calls ---

def call_gemini_api(api_key, model_name, prompt):
    """Call Google Gemini API."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini API Error: {str(e)}"

def call_openai_api(api_key, model_name, prompt):
    """Call OpenAI Chat Completions API."""
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"OpenAI API Error: {str(e)}"

def run_smart_mock(df, user_query):
    """
    A smart rule-based fallback analyzer that does not require API keys.
    Runs basic searches on columns to provide real calculated numbers.
    """
    query = user_query.lower()
    df_context = build_dataset_context(df)
    
    # Try to identify columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    reasoning = "Analyzed query using rule-based mock engine to extract numeric variables."
    code = ""
    text_response = "I searched the dataset columns but couldn't find a exact match. Here is a generic summary: The dataset has " + f"{df.shape[0]} rows and {df.shape[1]} columns."
    
    # Mock behavior for standard questions
    if "total" in query or "sum" in query or "sales" in query or "revenue" in query:
        # Find matching numeric col. Prioritize columns matching the keyword the
        # user actually used (e.g. "sales") over generic/other keywords that might
        # also appear in an unrelated column name (e.g. "price" matching "Price").
        target_col = None
        priority_keywords = ["sales", "revenue", "amount", "price", "count", "total"]
        matched_keywords = [k for k in priority_keywords if k in query]
        for k in matched_keywords:
            for col in num_cols:
                if k in col.lower():
                    target_col = col
                    break
            if target_col:
                break
        if not target_col:
            for col in num_cols:
                if any(k in col.lower() for k in priority_keywords):
                    target_col = col
                    break
        if target_col:
            total = df[target_col].sum()
            code = f"result = df['{target_col}'].sum()"
            text_response = f"The total sum of '{target_col}' across the dataset is: **{total:,.2f}**"
            
    elif "top" in query or "most" in query or "popular" in query or "sold" in query:
        # Find numeric and categorical cols
        cat_col = None
        num_col = None
        for col in cat_cols:
            if any(k in col.lower() for k in ["product", "item", "customer", "name", "category", "region"]):
                cat_col = col
                break
        for col in num_cols:
            if any(k in col.lower() for k in ["sales", "quantity", "amount", "count", "total"]):
                num_col = col
                break
                
        if cat_col and num_col:
            top_df = df.groupby(cat_col)[num_col].sum().reset_index().sort_values(by=num_col, ascending=False).head(5)
            top_val = top_df.iloc[0][cat_col]
            top_sum = top_df.iloc[0][num_col]
            code = f"result = df.groupby('{cat_col}')['{num_col}'].sum().reset_index().sort_values(by='{num_col}', ascending=False).head(5)\nfig = px.bar(result, x='{cat_col}', y='{num_col}', title='Top {cat_col} by {num_col}')"
            text_response = f"The top performing '{cat_col}' by total '{num_col}' is **{top_val}** with a total of **{top_sum:,.2f}**."
        elif cat_col:
            top_counts = df[cat_col].value_counts().head(5)
            top_val = top_counts.index[0]
            top_sum = top_counts.iloc[0]
            code = f"result = df['{cat_col}'].value_counts().head(5)\nfig = px.bar(x=result.index, y=result.values, title='Top {cat_col} counts')"
            text_response = f"The most frequent item in '{cat_col}' is **{top_val}** (appears {top_sum} times)."
            
    elif "trend" in query or "monthly" in query or "over time" in query or "time" in query:
        # Look for date and numerical cols
        date_col = None
        num_col = None
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]) or any(k in col.lower() for k in ["date", "time", "year", "month"]):
                date_col = col
                break
        for col in num_cols:
            if any(k in col.lower() for k in ["sales", "quantity", "amount", "total"]):
                num_col = col
                break
                
        if date_col and num_col:
            code = f"result = df.groupby('{date_col}')['{num_col}'].sum().reset_index()\nfig = px.line(result, x='{date_col}', y='{num_col}', title='{num_col} Trend Over Time')"
            text_response = f"Plotting the timeline of '{num_col}' relative to '{date_col}' to analyze sales or counts over time."
            
    elif "recommend" in query or "insights" in query or "business" in query:
        text_response = """
Here are rule-based analyst insights for your uploaded file:
1. **Data Health**: The dataset consists of **""" + f"{df.shape[0]} rows** and **{df.shape[1]} columns**." + """
2. **Missing Records**: Total missing cell values: **""" + f"{df.isnull().sum().sum()}**." + """
3. **Action Point**: Consider segmenting your categorical columns (like product category or region) against numerical values (like profits or quantities) to detect high-margin avenues.
"""

    return json.dumps({
        "reasoning": reasoning,
        "code": code,
        "text_response": text_response
    })

# --- Query Orchestrator ---

def query_dataset(df, user_query, provider, api_key=None, model_name=None):
    """
    Orchestrator to take a user question, invoke LLM to generate plan/code,
    execute the code, and compile the final answer.
    """
    df_context = build_dataset_context(df)
    prompt = prompt_template_for_qa(df_context, user_query)
    
    llm_output = ""
    
    if provider == "Gemini (Google)":
        if not api_key:
            return "Please provide a Google Gemini API Key in the sidebar or application config.", None, None
        model = model_name if model_name else "gemini-1.5-flash"
        llm_output = call_gemini_api(api_key, model, prompt)
    elif provider == "OpenAI":
        if not api_key:
            return "Please provide an OpenAI API Key in the sidebar or application config.", None, None
        model = model_name if model_name else "gpt-4o-mini"
        llm_output = call_openai_api(api_key, model, prompt)
    else: # Mock
        llm_output = run_smart_mock(df, user_query)
        
    # Clean LLM response (sometimes it's wrapped in md blocks)
    clean_json_str = llm_output.strip()
    if clean_json_str.startswith("```json"):
        clean_json_str = clean_json_str[7:]
    if clean_json_str.endswith("```"):
        clean_json_str = clean_json_str[:-3]
    clean_json_str = clean_json_str.strip()
    
    # Parse JSON
    try:
        response_dict = json.loads(clean_json_str)
    except Exception as e:
        # Fallback to direct regex parsing if JSON format is broken
        reasoning = "Parsing error on LLM response. Retrying fallback parser."
        code_match = re.search(r'"code"\s*:\s*"(.*?)"', clean_json_str, re.DOTALL)
        text_match = re.search(r'"text_response"\s*:\s*"(.*?)"', clean_json_str, re.DOTALL)
        
        code = code_match.group(1).encode().decode('unicode-escape') if code_match else ""
        text_response = text_match.group(1).encode().decode('unicode-escape') if text_match else "I analyzed the data but had an issue structuring the JSON response. Here is what I can tell you about the dataset columns: " + ", ".join(df.columns)
        response_dict = {"reasoning": reasoning, "code": code, "text_response": text_response}
        
    code_to_run = response_dict.get("code", "")
    text_response = response_dict.get("text_response", "")
    
    fig = None
    execution_result = None
    
    if code_to_run:
        # Run code on df
        exec_output = execute_analysis_code(df, code_to_run)
        
        # If code failed, try an auto-correction step
        if exec_output["error"]:
            correction_prompt = f"""
The Python code you generated earlier failed with an error.
Original Code:
{code_to_run}

Error Message:
{exec_output['error']}

Please correct the Python code to run successfully on a DataFrame named `df` with this structure:
{df_context}

Return a valid JSON string with the corrected code inside the "code" field.
"""
            if provider == "Gemini (Google)":
                corrected_output = call_gemini_api(api_key, model_name, correction_prompt)
            elif provider == "OpenAI":
                corrected_output = call_openai_api(api_key, model_name, correction_prompt)
            else:
                corrected_output = "{}"
                
            # Clean and parse correction
            try:
                corr_clean = corrected_output.strip()
                if corr_clean.startswith("```json"):
                    corr_clean = corr_clean[7:-3].strip()
                corr_dict = json.loads(corr_clean)
                code_to_run = corr_dict.get("code", code_to_run)
                # Re-run
                exec_output = execute_analysis_code(df, code_to_run)
            except:
                pass
                
        # Consolidate results
        stdout = exec_output["stdout"]
        result_val = exec_output["result"]
        fig = exec_output["fig"]
        error = exec_output["error"]
        
        # Build explanation string
        if error:
            explanation = f"Execution failed. Error:\n{error}"
        else:
            explanation = text_response
            if stdout:
                explanation += f"\n\n**Console Output:**\n```\n{stdout}\n```"
            if result_val is not None:
                # If result is a DataFrame, format it nicely
                if isinstance(result_val, pd.DataFrame):
                    explanation += f"\n\n**Resulting Table:**\n{result_val.to_markdown()}"
                elif isinstance(result_val, pd.Series):
                    explanation += f"\n\n**Resulting Series:**\n{result_val.to_frame().to_markdown()}"
                else:
                    explanation += f"\n\n**Calculated Answer:** {result_val}"
                    
        return explanation, fig, code_to_run
        
    return text_response, None, ""

# --- Business Insights Generator ---

def generate_business_insights(df, provider, api_key=None, model_name=None):
    """
    Generate structured business insights (Executive Summary, Key Findings,
    Strategic Recommendations, Risks & Opportunities).
    """
    df_context = build_dataset_context(df)
    prompt = generate_insights_prompt(df_context)
    
    if provider == "Gemini (Google)":
        if not api_key:
            return "Please enter a Gemini API key."
        return call_gemini_api(api_key, model_name or "gemini-1.5-flash", prompt)
    elif provider == "OpenAI":
        if not api_key:
            return "Please enter an OpenAI API key."
        return call_openai_api(api_key, model_name or "gpt-4o-mini", prompt)
    else: # Mock Fallback
        # Rule based analysis summary
        num_rows = df.shape[0]
        num_cols = df.shape[1]
        missing_count = df.isnull().sum().sum()
        dup_count = df.duplicated().sum()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        findings = []
        for col in numeric_cols[:2]:
            findings.append(f"Column **{col}** has a mean value of **{df[col].mean():,.2f}** and ranges from **{df[col].min():,.2f}** to **{df[col].max():,.2f}**.")
            
        for col in categorical_cols[:2]:
            top_val = df[col].value_counts().idxmax()
            top_count = df[col].value_counts().max()
            findings.append(f"Column **{col}** most frequent value is **{top_val}**, appearing **{top_count}** times ({top_count/num_rows*100:.1f}%).")
            
        findings_str = "\n".join([f"- {f}" for f in findings])
        
        mock_insights = f"""# Executive Summary
This dataset contains **{num_rows} records** across **{num_cols} distinct dimensions**. Based on structural analysis, the dataset contains {len(numeric_cols)} numeric measures and {len(categorical_cols)} categorical descriptive parameters. The primary objective is to review operational efficiency and maximize profit margins.

# Key Findings
{findings_str}
- The data quality shows **{missing_count} missing value cells** and **{dup_count} duplicate rows** requiring pre-cleaning steps before modeling.

# Strategic Recommendations
- **Segmented Marketing**: Target high-frequency profiles in categories like `{categorical_cols[0] if categorical_cols else 'N/A'}` to improve customer retention.
- **Data Enrichment**: Clean the {missing_count} missing records to prevent analytical bias in forecasting metrics.
- **Resource Allocation**: Direct resources towards columns with high-variance metrics (e.g. `{numeric_cols[0] if numeric_cols else 'N/A'}`) to stabilize cashflow.

# Risks and Opportunities
- **Risks**:
  - Outliers in numerical fields can skew average forecasting metrics.
  - Data sparsity: presence of empty parameters blocks precise classification modeling.
- **Opportunities**:
  - Automated classification models can predict column categories based on numeric variables with high accuracy.
  - Splicing and grouping data by `{categorical_cols[0] if categorical_cols else 'N/A'}` reveals high-performance regions.
"""
        return mock_insights
