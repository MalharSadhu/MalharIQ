import json
import numpy as np
import ollama
import pandas as pd


def generate_data_summary(df: pd.DataFrame) -> dict:
    """Extracts lightweight metadata without sending raw rows to LLM."""
    summary = {
        "num_rows": len(df),
        "columns": {},
    }
    for col in df.columns:
        col_data = df[col]
        summary["columns"][col] = {
            "dtype": str(col_data.dtype),
            "null_count": int(col_data.isnull().sum()),
            "null_pct": round(float(col_data.isnull().mean() * 100), 2),
            "unique_values": int(col_data.nunique()),
            "sample_values": [str(x) for x in col_data.dropna().head(3).tolist()],
        }
        if pd.api.types.is_numeric_dtype(col_data):
            summary["columns"][col]["min"] = float(col_data.min()) if not col_data.empty else None
            summary["columns"][col]["max"] = float(col_data.max()) if not col_data.empty else None
    return summary


def infer_ml_task_and_target(df_summary: dict, user_prompt: str = "") -> dict:
    """Agent deduces whether to Cluster, Classify, or Regress + identifies target."""
    system_prompt = """
    You are an AI Machine Learning Architect.
    Analyze the dataset summary and user prompt to determine the optimal ML task.
    
    Decision Rules:
    - If user prompt explicitly mentions clustering/segmentation, or if there is no obvious target label -> "Clustering / Segmentation" (target_col: null)
    - If there is a categorical/discrete target (e.g., 'Species', 'Churn', 'Status') -> "Classification"
    - If there is a continuous numeric target (e.g., 'Price', 'Salary', 'Sales') -> "Regression"
    
    Output ONLY valid JSON:
    {
      "task": "Clustering / Segmentation" | "Classification" | "Regression",
      "target_col": "column_name" or null,
      "reasoning": "1 short sentence explaining why this task and target were selected"
    }
    """

    user_content = f"User Intent: {user_prompt if user_prompt else 'Auto-detect best objective'}\nDataset Summary:\n{json.dumps(df_summary, indent=2)}"

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        format="json",
        options={"temperature": 0.1},
    )
    return json.loads(response["message"]["content"])


def get_cleaning_recipe(summary: dict, user_goal: str) -> dict:
    """Generates an explicit JSON data cleaning plan."""
    system_prompt = """
    You are an expert MLOps Data Preparation Agent.
    Analyze tabular metadata and return ONLY a JSON cleaning plan:
    {
      "drop_columns": ["col_id", "col_useless"],
      "impute_median": ["numeric_with_nulls"],
      "impute_mode": ["categorical_with_nulls"],
      "replace_negatives_with_null": ["strictly_positive_numeric_cols"],
      "drop_rows_with_nulls_in": ["target_col"],
      "reasoning": "Short 1-sentence explanation"
    }
    """

    user_prompt = f"Objective: {user_goal}\nMetadata:\n{json.dumps(summary, indent=2)}"

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        format="json",
        options={"temperature": 0.1},
    )
    return json.loads(response["message"]["content"])


def execute_cleaning_recipe(df: pd.DataFrame, recipe: dict) -> tuple[pd.DataFrame, list[str]]:
    """Applies cleaning recipe deterministically via Pandas."""
    df_clean = df.copy()
    logs = []

    # 1. Negative values to NaN
    for col in recipe.get("replace_negatives_with_null", []):
        if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
            neg_count = (df_clean[col] < 0).sum()
            if neg_count > 0:
                df_clean[col] = df_clean[col].apply(lambda x: np.nan if pd.notnull(x) and x < 0 else x)
                logs.append(f"Replaced {neg_count} negative entries in '{col}' with NaN")

    # 2. Impute Medians
    for col in recipe.get("impute_median", []):
        if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            logs.append(f"Imputed missing values in '{col}' using median ({median_val})")

    # 3. Impute Modes
    for col in recipe.get("impute_mode", []):
        if col in df_clean.columns and not df_clean[col].mode().empty:
            mode_val = df_clean[col].mode()[0]
            df_clean[col] = df_clean[col].fillna(mode_val)
            logs.append(f"Imputed missing values in '{col}' using mode ('{mode_val}')")

    # 4. Drop Rows with Nulls in critical targets
    for col in recipe.get("drop_rows_with_nulls_in", []):
        if col in df_clean.columns:
            before_len = len(df_clean)
            df_clean = df_clean.dropna(subset=[col])
            logs.append(f"Dropped {before_len - len(df_clean)} rows missing target column '{col}'")

    # 5. Drop useless/ID columns
    cols_to_drop = [c for c in recipe.get("drop_columns", []) if c in df_clean.columns]
    if cols_to_drop:
        df_clean = df_clean.drop(columns=cols_to_drop)
        logs.append(f"Dropped non-predictive/ID columns: {', '.join(cols_to_drop)}")

    return df_clean, logs