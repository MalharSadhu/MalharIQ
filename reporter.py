import json
import re
import numpy as np
import ollama
import pandas as pd


def sanitize_for_json(data):
    """Prunes large arrays and converts pandas/numpy objects for clean serialization."""
    EXCLUDE_KEYS = {
        "factor_scores_df",
        "result_df",
        "model_obj",
        "model",
        "fitted_pipeline",
        "predictions",
        "probabilities",
        "silhouette_samples",
        "confusion_matrix",
    }

    if isinstance(data, pd.DataFrame):
        return data.head(5).to_dict(orient="records")
    if isinstance(data, pd.Series):
        return data.to_dict()
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, (np.floating, np.integer)):
        return data.item()
    if isinstance(data, dict):
        return {
            str(k): sanitize_for_json(v)
            for k, v in data.items()
            if k not in EXCLUDE_KEYS
        }
    if isinstance(data, list):
        return [sanitize_for_json(i) for i in data]
    return data


def generate_dual_reports(
    cleaning_logs: list[str],
    stats_results: dict,
    ml_results: dict,
    user_goal: str,
    model_name: str = "llama3.2:1b",
) -> dict:
    system_prompt = """
    You are a Senior Engagement Manager at an elite management consultancy.
    Generate a high-impact, client-ready advisory report based on the provided pipeline data.
    
    You MUST respond with ONLY a valid JSON object matching this exact structure:
    {
      "business_report": {
        "executive_verdict": "2-3 high-impact sentences summarizing strategic takeaway.",
        "segment_insights": [
          "Detailed insight on Segment/Cluster 1.",
          "Detailed insight on Segment/Cluster 2."
        ],
        "data_readiness_score": "High Quality",
        "data_readiness_detail": "Plain-English audit of data health and missing values resolved.",
        "strategic_recommendations": [
          "Recommendation 1: Concrete operational action.",
          "Recommendation 2: Targeted resource allocation initiative."
        ],
        "potential_business_risks": [
          "Risk 1 regarding variance.",
          "Risk 2 regarding feature drift."
        ]
      },
      "technical_report": {
        "architecture_summary": "Summary of winning algorithm and validation methodology.",
        "data_engineering_audit": [
          "Imputation and cleaning transformation details.",
          "Feature scaling methodologies applied."
        ],
        "statistical_quality_gates": {
          "adequacy_verdict": "KMO & Bartlett test assessment.",
          "factor_reduction": "Latent factor extraction breakdown."
        },
        "production_monitoring_plan": [
          "Drift detection protocol frequency.",
          "Automated retraining triggers."
        ]
      }
    }
    """

    user_payload = {
        "user_goal": user_goal,
        "cleaning_logs": sanitize_for_json(cleaning_logs[:6]),
        "statistical_validation": sanitize_for_json(stats_results),
        "ml_engine_results": sanitize_for_json(ml_results),
    }

    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Pipeline Data:\n{json.dumps(user_payload, default=str)}",
                },
            ],
            format="json",
            keep_alive="1h",
            options={
                "temperature": 0.1,
                "num_predict": 1200,
                "num_ctx": 2048,
            },
        )

        content = response["message"]["content"].strip()
        content = re.sub(r"^```json\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        return json.loads(content)

    except Exception as e:
        print(f"\n[!] OLLAMA FAILED: {repr(e)}\n")
        return {
            "business_report": {
                "executive_verdict": f"Error running local report: {str(e)}",
                "segment_insights": ["Could not parse cluster insights."],
                "data_readiness_score": "Action Required",
                "data_readiness_detail": "Pipeline ran into an inference exception.",
                "strategic_recommendations": ["Check terminal output for error logs."],
                "potential_business_risks": ["Model generation fallback active."],
            },
            "technical_report": {
                "architecture_summary": f"Selected model: {ml_results.get('best_algorithm', 'N/A')}",
                "data_engineering_audit": cleaning_logs[:5],
                "statistical_quality_gates": {"adequacy_verdict": "Verified", "factor_reduction": "Completed"},
                "production_monitoring_plan": ["Verify Ollama service is active locally."],
            },
        }