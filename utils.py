import pandas as pd
import numpy as np
import os
from pathlib import Path

def validate_file_extension(filename):
    """Validate if the uploaded file has a valid extension (CSV or XLSX)."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in ['.csv', '.xlsx', '.xls']

def load_data(file_path):
    """Load dataset into a pandas DataFrame based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.csv':
        return pd.read_csv(file_path)
    elif ext in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload CSV or Excel.")

def get_basic_info(df):
    """Generate basic dataset shape, size, memory, and duplicate details."""
    info = {
        "num_rows": int(df.shape[0]),
        "num_cols": int(df.shape[1]),
        "num_duplicate_rows": int(df.duplicated().sum()),
        "total_missing": int(df.isnull().sum().sum()),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    }
    return info

def detect_column_types(df):
    """Categorize columns into numeric, categorical, datetime, and text."""
    column_types = {
        "numeric": [],
        "categorical": [],
        "datetime": [],
        "text": []
    }
    
    for col in df.columns:
        # Check datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            column_types["datetime"].append(col)
            continue
            
        # Try parsing as datetime if object type and has date-like name
        if df[col].dtype == 'object':
            lower_col = col.lower()
            if any(k in lower_col for k in ['date', 'time', 'timestamp', 'day', 'month', 'year']):
                try:
                    pd.to_datetime(df[col].dropna().head(10))
                    column_types["datetime"].append(col)
                    continue
                except:
                    pass
                    
        # Check numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            # If binary or has very few unique values, treat as categorical
            if df[col].nunique() <= 5:
                column_types["categorical"].append(col)
            else:
                column_types["numeric"].append(col)
        # Check categorical (objects/categories with low cardinality)
        elif df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]):
            if df[col].nunique() < 50:
                column_types["categorical"].append(col)
            else:
                column_types["text"].append(col)
        else:
            column_types["text"].append(col)
            
    return column_types

def get_missing_values_report(df):
    """Returns a summary of missing values per column."""
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    
    report = pd.DataFrame({
        "Missing Count": missing_count,
        "Percentage (%)": missing_pct
    })
    
    return report[report["Missing Count"] > 0].sort_values(by="Missing Count", ascending=False)

def clean_duplicates(df):
    """Remove duplicate rows from a DataFrame and return the cleaned copy."""
    return df.drop_duplicates().reset_ok() if hasattr(df, 'reset_ok') else df.drop_duplicates().reset_index(drop=True)

def handle_missing_values(df, column, strategy='mean', fill_value=None):
    """Impute or drop missing values inside a column."""
    df_clean = df.copy()
    if strategy == 'mean' and pd.api.types.is_numeric_dtype(df_clean[column]):
        df_clean[column] = df_clean[column].fillna(df_clean[column].mean())
    elif strategy == 'median' and pd.api.types.is_numeric_dtype(df_clean[column]):
        df_clean[column] = df_clean[column].fillna(df_clean[column].median())
    elif strategy == 'mode':
        mode_val = df_clean[column].mode()
        if not mode_val.empty:
            df_clean[column] = df_clean[column].fillna(mode_val[0])
    elif strategy == 'constant' and fill_value is not None:
        df_clean[column] = df_clean[column].fillna(fill_value)
    elif strategy == 'drop':
        df_clean = df_clean.dropna(subset=[column]).reset_index(drop=True)
    elif strategy == 'ffill':
        df_clean[column] = df_clean[column].ffill()
    
    return df_clean

def get_statistical_summary(df):
    """Generate detailed description statistical summary for numeric and object columns."""
    numeric_summary = df.describe().T
    
    # Calculate additional metrics for numeric columns
    if not numeric_summary.empty:
        numeric_summary['skewness'] = df[numeric_summary.index].skew()
        numeric_summary['kurtosis'] = df[numeric_summary.index].kurt()
        numeric_summary = numeric_summary.round(4)
        
    return numeric_summary

def detect_outliers_iqr(df, column):
    """Detect outliers using the Interquartile Range (IQR) method."""
    if not pd.api.types.is_numeric_dtype(df[column]):
        return pd.DataFrame(), 0, 0.0
        
    col_data = df[column].dropna()
    q1 = col_data.quantile(0.25)
    q3 = col_data.quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    outlier_count = len(outliers)
    outlier_percentage = (outlier_count / len(df)) * 100
    
    return outliers, outlier_count, round(outlier_percentage, 2)

def get_all_outliers_summary(df, numeric_cols):
    """Get a dictionary summarizing the number and percent of outliers per column."""
    summary = {}
    for col in numeric_cols:
        _, count, pct = detect_outliers_iqr(df, col)
        if count > 0:
            summary[col] = {"count": count, "percentage": pct}
    return summary

def get_correlation_matrix(df):
    """Calculate pearson correlation matrix for numeric fields."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return pd.DataFrame()
    return numeric_df.corr().round(4)
