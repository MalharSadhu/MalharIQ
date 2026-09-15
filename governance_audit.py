import pandas as pd


def audit_disparate_impact(
    df: pd.DataFrame,
    sensitive_col: str,
    outcome_col: str,
    favorable_outcome=None,
) -> dict:
  """Calculates Disparate Impact ratio across subgroups of a sensitive column

  using the US EEOC Four-Fifths (80%) regulatory standard.
  """
  if sensitive_col not in df.columns or outcome_col not in df.columns:
    return {
        "status": "ERROR",
        "message": "Sensitive or outcome column missing from dataset.",
    }

  # Determine favorable outcome if not specified
  if favorable_outcome is None:
    if pd.api.types.is_numeric_dtype(df[outcome_col]):
      favorable_outcome = df[outcome_col].max()
    else:
      favorable_outcome = df[outcome_col].mode()[0]

  # Calculate selection / favorable rate per group
  group_rates = {}
  group_counts = {}
  for group_val, sub_df in df.groupby(sensitive_col):
    if len(sub_df) == 0:
      continue
    rate = (sub_df[outcome_col] == favorable_outcome).mean()
    group_rates[str(group_val)] = round(float(rate), 4)
    group_counts[str(group_val)] = len(sub_df)

  if not group_rates:
    return {
        "status": "ERROR",
        "message": "No valid group subsets could be calculated.",
    }

  benchmark_group = max(group_rates, key=group_rates.get)
  benchmark_rate = group_rates[benchmark_group]

  # Compute parity ratios against benchmark
  ratios = {}
  violations = []
  for grp, rate in group_rates.items():
    ratio = (rate / benchmark_rate) if benchmark_rate > 0 else 1.0
    ratio_val = round(ratio, 3)
    ratios[grp] = ratio_val
    if ratio_val < 0.80:
      violations.append({
          "group": grp,
          "selection_rate": rate,
          "disparate_impact_ratio": ratio_val,
      })

  is_compliant = len(violations) == 0

  return {
      "status": "PASS" if is_compliant else "WARNING",
      "favorable_outcome_evaluated": str(favorable_outcome),
      "benchmark_group": benchmark_group,
      "benchmark_rate": benchmark_rate,
      "subgroup_rates": group_rates,
      "group_counts": group_counts,
      "disparate_impact_ratios": ratios,
      "violating_groups": violations,
      "verdict": (
          "Model & data distribution satisfy the EEOC 80% Four-Fifths parity"
          " rule."
          if is_compliant
          else f"Disparate impact risk: {len(violations)} cohort(s) fall below"
          f" the 80% parity threshold relative to '{benchmark_group}'."
      ),
  }