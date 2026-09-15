import json
import ollama
import pandas as pd


def extract_structured_dataset_from_text(
    raw_texts: list[str],
    user_schema_hint: str = "",
    model_name: str = "llama3.2:1b",
) -> pd.DataFrame:
  """Converts a batch of unstructured text records into a structured Pandas DataFrame."""
  # Step 1: Prompt Ollama to extract consistent features
  system_prompt = f"""
    You are an automated ETL feature extraction engine.
    Convert the following text record into a standardized JSON record with consistent numerical and categorical fields.
    {f"User Schema Directive: {user_schema_hint}" if user_schema_hint else "Extract: category, urgency (1-5), sentiment_score (-1.0 to 1.0), and main_topic."}
    
    Output ONLY valid JSON. Example:
    {{"category": "Billing", "urgency": 4, "sentiment_score": -0.8, "resolved": 0}}
    """

  parsed_records = []
  for text_snippet in raw_texts[:100]:  # Cap batch to prevent latency
    if not text_snippet.strip():
      continue
    try:
      response = ollama.chat(
          model=model_name,
          messages=[
              {"role": "system", "content": system_prompt},
              {"role": "user", "content": f"Text: {text_snippet}"},
          ],
          format="json",
          options={"temperature": 0.1},
      )
      row = json.loads(response["message"]["content"])
      parsed_records.append(row)
    except Exception:
      continue

  return pd.DataFrame(parsed_records)