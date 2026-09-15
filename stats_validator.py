import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def calculate_kmo_bartlett(df: pd.DataFrame) -> dict:
    """Computes KMO and Bartlett's Test cleanly using pure numpy/scipy."""
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    if numeric_df.shape[1] < 2:
        return {"error": "Need at least 2 numeric columns."}

    # Correlation Matrix
    corr = numeric_df.corr().values
    n, p = numeric_df.shape

    # Determinant for Bartlett's test
    det_corr = np.linalg.det(corr)
    if det_corr <= 0:
        det_corr = 1e-10

    # Bartlett's Sphericity calculation
    statistic = -1 * (n - 1 - ((2 * p + 5) / 6)) * np.log(det_corr)
    degrees_of_freedom = p * (p - 1) / 2
    p_value = stats.chi2.sf(statistic, degrees_of_freedom)

    # Simplified KMO calculation
    inv_corr = np.linalg.pinv(corr)
    partial_corr = -inv_corr / np.sqrt(np.outer(np.diag(inv_corr), np.diag(inv_corr)))
    np.fill_diagonal(partial_corr, 0)
    
    r2 = corr ** 2
    np.fill_diagonal(r2, 0)
    a2 = partial_corr ** 2
    
    kmo_score = float(np.sum(r2) / (np.sum(r2) + np.sum(a2)))
    kmo_score = round(max(0.0, min(kmo_score, 1.0)), 4)

    if kmo_score >= 0.8:
        rating = "Excellent"
    elif kmo_score >= 0.6:
        rating = "Moderate / Suitable"
    else:
        rating = "Poor Suitability"

    return {
        "kmo_score": kmo_score,
        "kmo_rating": rating,
        "bartlett_chi_sq": round(float(statistic), 2),
        "bartlett_p_value": float(np.format_float_scientific(p_value, precision=4)),
        "clustering_suitability": "Suitable" if (kmo_score >= 0.55 and p_value < 0.05) else "Unsuitable",
    }


def run_feature_significance_tests(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Returns a clean DataFrame of T-Test / ANOVA results for classification."""
    if target_col not in df.columns:
        return pd.DataFrame()

    results = []
    target_series = df[target_col].dropna()
    unique_classes = target_series.unique()
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]

    for col in numeric_cols:
        col_data = df[[col, target_col]].dropna()

        if len(unique_classes) == 2:
            group1 = col_data[col_data[target_col] == unique_classes[0]][col]
            group2 = col_data[col_data[target_col] == unique_classes[1]][col]

            if len(group1) > 1 and len(group2) > 1:
                t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
                results.append({
                    "Feature": col,
                    "Test": "Independent T-Test",
                    "Score/Stat": round(float(t_stat), 3),
                    "P-Value": f"{p_val:.4e}",
                    "Significant": "✅ Yes (p < 0.05)" if p_val < 0.05 else "❌ No (Noise)",
                    "Action": "Retain in Model" if p_val < 0.05 else "Candidate for Removal"
                })
        elif len(unique_classes) > 2:
            groups = [col_data[col_data[target_col] == cls][col] for cls in unique_classes]
            valid_groups = [g for g in groups if len(g) > 1]

            if len(valid_groups) > 1:
                f_stat, p_val = stats.f_oneway(*valid_groups)
                results.append({
                    "Feature": col,
                    "Test": "One-Way ANOVA",
                    "Score/Stat": round(float(f_stat), 3),
                    "P-Value": f"{p_val:.4e}",
                    "Significant": "✅ Yes (p < 0.05)" if p_val < 0.05 else "❌ No (Noise)",
                    "Action": "Retain in Model" if p_val < 0.05 else "Candidate for Removal"
                })

    return pd.DataFrame(results)


def extract_latent_factors(df: pd.DataFrame) -> dict:
    """Native PCA factor analysis without buggy third-party libraries."""
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    if numeric_df.shape[1] < 2:
        return {"error": "Need at least 2 numeric variables for Factor Analysis."}

    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(numeric_df)

    pca_full = PCA().fit(scaled_matrix)
    eigenvalues = pca_full.explained_variance_

    optimal_factors = int(np.sum(eigenvalues >= 1.0))
    optimal_factors = max(1, min(optimal_factors, numeric_df.shape[1]))

    pca = PCA(n_components=optimal_factors)
    factor_scores = pca.fit_transform(scaled_matrix)

    factor_col_names = [f"Factor_{i+1}" for i in range(optimal_factors)]
    factor_df = pd.DataFrame(factor_scores, columns=factor_col_names, index=numeric_df.index)

    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loadings_df = pd.DataFrame(loadings, index=numeric_df.columns, columns=factor_col_names).round(3)

    variance_explained = [f"{round(v * 100, 1)}%" for v in pca.explained_variance_ratio_]

    return {
        "num_factors_extracted": optimal_factors,
        "eigenvalues": [round(float(e), 3) for e in eigenvalues[:optimal_factors]],
        "variance_explained": variance_explained,
        "factor_loadings_df": loadings_df,
        "factor_scores_df": factor_df,
    }