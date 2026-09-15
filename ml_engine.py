import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import xgboost as xgb

try:
    from mlxtend.frequent_patterns import apriori, association_rules
except ImportError:
    apriori = None


# ==============================================================================
# ROBUST FEATURE PREPROCESSING & DATETIME EXPANSION HELPER
# ==============================================================================
def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safely converts datetime columns into numeric calendar components,
    encodes categoricals, drops un-indexable high-cardinality text,
    and guards against 'ValueError: No objects to concatenate'.
    """
    if df.empty or df.shape[1] == 0:
        return pd.DataFrame({"constant_bias": [1.0] * len(df)}, index=df.index)

    X = df.copy()

    # 1. Identify and extract datetime columns
    for col in list(X.columns):
        try:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                X[f"{col}_year"] = X[col].dt.year
                X[f"{col}_month"] = X[col].dt.month
                X[f"{col}_day"] = X[col].dt.day
                X[f"{col}_dayofweek"] = X[col].dt.dayofweek
                X = X.drop(columns=[col])
            elif X[col].dtype == "object":
                if any(k in col.lower() for k in ["date", "time", "timestamp", "year", "day"]):
                    parsed = pd.to_datetime(X[col], errors="coerce")
                    if parsed.notnull().mean() > 0.6:
                        X[f"{col}_year"] = parsed.dt.year.fillna(2026).astype(int)
                        X[f"{col}_month"] = parsed.dt.month.fillna(1).astype(int)
                        X[f"{col}_day"] = parsed.dt.day.fillna(1).astype(int)
                        X[f"{col}_dayofweek"] = parsed.dt.dayofweek.fillna(0).astype(int)
                        X = X.drop(columns=[col])
        except Exception:
            pass

    # 2. Drop high-cardinality non-predictive string columns (IDs, hashes, names)
    for col in list(X.columns):
        if X[col].dtype == "object":
            unique_count = X[col].nunique()
            unique_ratio = unique_count / max(len(X), 1)
            if unique_ratio > 0.8 and unique_count > 50:
                X = X.drop(columns=[col])

    # Guard: If no columns remain after dropping, retain original numeric columns or fallback
    if X.shape[1] == 0:
        numeric_fallback = df.select_dtypes(include=["number"])
        if not numeric_fallback.empty:
            X = numeric_fallback.copy()
        else:
            return pd.DataFrame({"constant_bias": [1.0] * len(df)}, index=df.index)

    # 3. Categorical encoding with concat guard
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

    if categorical_cols:
        try:
            X_dummies = pd.get_dummies(X[categorical_cols], drop_first=True)
            if not X_dummies.empty and X_dummies.shape[1] > 0:
                if numeric_cols:
                    X_encoded = pd.concat([X[numeric_cols], X_dummies], axis=1)
                else:
                    X_encoded = X_dummies.copy()
            else:
                X_encoded = X[numeric_cols].copy() if numeric_cols else pd.DataFrame({"constant_bias": [1.0] * len(df)}, index=df.index)
        except Exception:
            X_encoded = X[numeric_cols].copy() if numeric_cols else pd.DataFrame({"constant_bias": [1.0] * len(df)}, index=df.index)
    else:
        X_encoded = X.copy()

    # 4. Cast to float and impute nulls
    X_encoded = X_encoded.apply(pd.to_numeric, errors="coerce")
    X_encoded = X_encoded.fillna(X_encoded.median(numeric_only=True))
    X_encoded = X_encoded.dropna(axis=1, how="all")

    # 5. Final fallback guarantee
    if X_encoded.shape[1] == 0:
        X_encoded["constant_bias"] = 1.0

    return X_encoded


# ==============================================================================
# 70 / 15 / 15 PARTITION SPLIT HELPER
# ==============================================================================
def split_70_15_15(
    X, y=None, random_state: int = 42, stratify: bool = False
) -> tuple:
    strat_y = (
        y if (stratify and y is not None and len(np.unique(y)) > 1) else None
    )
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=strat_y
    )

    strat_temp = (
        y_temp
        if (stratify and y_temp is not None and len(np.unique(y_temp)) > 1)
        else None
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=strat_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ==============================================================================
# 1. CLUSTERING TOURNAMENT
# ==============================================================================
def run_clustering_tournament(df: pd.DataFrame, max_k: int = 6) -> dict:
    clean_features = preprocess_features(df)
    numeric_df = clean_features.select_dtypes(include=["number"]).dropna()

    if numeric_df.empty or len(numeric_df.columns) < 2:
        return {"error": "Clustering requires at least two numeric features."}

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    candidates = []
    upper_k = min(max_k + 1, len(numeric_df))

    for k in range(2, upper_k):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled_data)
        if len(set(labels)) > 1:
            candidates.append({
                "algorithm": f"K-Means (k={k})",
                "family": "Centroid-Based",
                "model_obj": km,
                "labels": labels,
                "silhouette": silhouette_score(scaled_data, labels),
                "davies_bouldin": davies_bouldin_score(scaled_data, labels),
                "calinski_harabasz": calinski_harabasz_score(scaled_data, labels),
                "k": k,
            })

    for linkage in ["ward", "complete", "average"]:
        for k in range(2, min(5, len(numeric_df))):
            agg = AgglomerativeClustering(n_clusters=k, linkage=linkage)
            labels = agg.fit_predict(scaled_data)
            if len(set(labels)) > 1:
                candidates.append({
                    "algorithm": f"Hierarchical [{linkage.capitalize()}] (k={k})",
                    "family": "Connectivity-Based",
                    "model_obj": agg,
                    "labels": labels,
                    "silhouette": silhouette_score(scaled_data, labels),
                    "davies_bouldin": davies_bouldin_score(scaled_data, labels),
                    "calinski_harabasz": calinski_harabasz_score(scaled_data, labels),
                    "k": k,
                })

    if not candidates:
        return {"error": "Unable to compute valid cluster boundaries."}

    candidates = sorted(candidates, key=lambda x: x["silhouette"], reverse=True)
    winner = candidates[0]

    leaderboard = [
        {
            "Rank": f"#{i+1}",
            "Algorithm": c["algorithm"],
            "Family": c["family"],
            "Silhouette Score": round(float(c["silhouette"]), 4),
            "Davies-Bouldin": round(float(c["davies_bouldin"]), 4),
            "Calinski-Harabasz": round(float(c["calinski_harabasz"]), 2),
            "Optimal Clusters": c["k"],
            "Verdict": "🏆 Selected Champion" if i == 0 else "Evaluated",
        }
        for i, c in enumerate(candidates)
    ]

    result_df = numeric_df.copy()
    result_df["Cluster"] = winner["labels"]

    return {
        "task": "Clustering",
        "best_algorithm": winner["algorithm"],
        "best_k": winner["k"],
        "silhouette_score": round(float(winner["silhouette"]), 4),
        "davies_bouldin": round(float(winner["davies_bouldin"]), 4),
        "calinski_harabasz": round(float(winner["calinski_harabasz"]), 2),
        "leaderboard_df": pd.DataFrame(leaderboard),
        "result_df": result_df,
    }


# ==============================================================================
# 2. CLASSIFICATION TOURNAMENT (70 / 15 / 15)
# ==============================================================================
def run_classification_tournament(df: pd.DataFrame, target_col: str) -> dict:
    if target_col not in df.columns:
        return {"error": f"Target '{target_col}' not found."}

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_encoded = preprocess_features(X)

    if y.dtype == "object" or isinstance(y.iloc[0], str):
        y_encoded, classes_list = pd.factorize(y)
        class_labels = [str(c) for c in classes_list]
    else:
        y_encoded = y.values
        class_labels = [str(c) for c in np.unique(y_encoded)]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    X_train, X_val, X_test, y_train, y_val, y_test = split_70_15_15(
        X_scaled, y_encoded, random_state=42, stratify=True
    )

    rf = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    xgb_model = xgb.XGBClassifier(
        eval_metric="logloss", max_depth=5, n_estimators=50, random_state=42, verbosity=0
    )
    log_reg = LogisticRegression(max_iter=500, random_state=42)
    svm = SVC(probability=True, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=min(5, len(X_train)))
    ann = MLPClassifier(
        hidden_layer_sizes=(32, 16), max_iter=150, early_stopping=True, random_state=42
    )

    voting_ensemble = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb_model), ("lr", log_reg)],
        voting="soft",
    )
    stacking_ensemble = StackingClassifier(
        estimators=[("rf", rf), ("svm", svm), ("knn", knn)],
        final_estimator=LogisticRegression(),
    )

    models = {
        "Random Forest": rf,
        "XGBoost Classifier": xgb_model,
        "Logistic Regression": log_reg,
        "Support Vector Machine (SVM)": svm,
        "K-Nearest Neighbors (KNN)": knn,
        "Artificial Neural Network (ANN / Deep MLP)": ann,
        "Ensemble - Soft Voting (RF + XGB + LR)": voting_ensemble,
        "Ensemble - Stacking Classifier": stacking_ensemble,
    }

    candidates = []
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)

            # Benchmark on 15% Validation partition
            val_preds = model.predict(X_val)
            val_acc = accuracy_score(y_val, val_preds)
            val_f1 = f1_score(y_val, val_preds, average="weighted")

            # Holdout verification on 15% Test partition
            test_preds = model.predict(X_test)
            test_acc = accuracy_score(y_test, test_preds)
            test_f1 = f1_score(y_test, test_preds, average="weighted")

            candidates.append({
                "algorithm": name,
                "model": model,
                "val_accuracy": val_acc,
                "val_f1": val_f1,
                "test_accuracy": test_acc,
                "test_f1": test_f1,
                "y_test": y_test,
                "y_pred": test_preds,
            })
        except Exception:
            continue

    if not candidates:
        return {"error": "All classification models failed to converge."}

    candidates = sorted(candidates, key=lambda x: x["val_f1"], reverse=True)
    winner = candidates[0]

    leaderboard = [
        {
            "Rank": f"#{i+1}",
            "Algorithm": c["algorithm"],
            "Val F1 (15%)": round(float(c["val_f1"]), 4),
            "Test F1 (15%)": round(float(c["test_f1"]), 4),
            "Test Accuracy": f"{c['test_accuracy']*100:.2f}%",
            "Verdict": "🏆 Winner" if i == 0 else "Evaluated",
        }
        for i, c in enumerate(candidates)
    ]

    feature_importances = {}
    if hasattr(winner["model"], "feature_importances_"):
        feature_importances = dict(
            zip(
                X_encoded.columns,
                [round(float(v), 4) for v in winner["model"].feature_importances_],
            )
        )
    elif hasattr(winner["model"], "coef_"):
        coef_vals = np.mean(np.abs(winner["model"].coef_), axis=0) if winner["model"].coef_.ndim > 1 else np.abs(winner["model"].coef_)
        feature_importances = dict(
            zip(
                X_encoded.columns,
                [round(float(v), 4) for v in coef_vals],
            )
        )

    return {
        "task": "Classification",
        "best_algorithm": winner["algorithm"],
        "accuracy": round(float(winner["test_accuracy"]), 4),
        "f1_score": round(float(winner["test_f1"]), 4),
        "val_f1_score": round(float(winner["val_f1"]), 4),
        "leaderboard_df": pd.DataFrame(leaderboard),
        "feature_importances": feature_importances,
        "y_test": [int(x) for x in winner["y_test"]],
        "y_pred": [int(x) for x in winner["y_pred"]],
        "class_labels": class_labels,
    }


# ==============================================================================
# 3. REGRESSION TOURNAMENT (70 / 15 / 15 WITH VISUAL METRIC PAYLOADS & FAST CONVERGENCE)
# ==============================================================================
def run_regression_tournament(df: pd.DataFrame, target_col: str) -> dict:
    if target_col not in df.columns:
        return {"error": f"Target '{target_col}' not found."}

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_encoded = preprocess_features(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    X_train, X_val, X_test, y_train, y_val, y_test = split_70_15_15(
        X_scaled, y.values, random_state=42, stratify=False
    )

    # Fast-training model zoo with capped iterations and multi-threading
    models = {
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=50, max_depth=10, n_jobs=-1, random_state=42
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=50, max_depth=4, random_state=42
        ),
        "Ridge Regression (L2)": Ridge(alpha=1.0),
        "Lasso Regression (L1)": Lasso(alpha=0.1, max_iter=1000),
        "Linear Regression (OLS)": LinearRegression(n_jobs=-1),
        "Neural Network Regressor (ANN / MLP)": MLPRegressor(
            hidden_layer_sizes=(32, 16),
            max_iter=150,
            early_stopping=True,
            random_state=42,
        ),
    }

    candidates = []
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)

            # Benchmark on 15% Validation partition
            val_preds = model.predict(X_val)
            val_r2 = r2_score(y_val, val_preds)

            # Holdout verification on 15% Test partition
            test_preds = model.predict(X_test)
            test_r2 = r2_score(y_test, test_preds)
            test_mse = mean_squared_error(y_test, test_preds)
            test_mae = mean_absolute_error(y_test, test_preds)

            candidates.append({
                "algorithm": name,
                "model": model,
                "val_r2": val_r2,
                "test_r2": test_r2,
                "test_mse": test_mse,
                "test_mae": test_mae,
                "y_test": y_test,
                "y_pred": test_preds,
            })
        except Exception:
            continue

    if not candidates:
        return {"error": "Regression models failed to train."}

    candidates = sorted(candidates, key=lambda x: x["val_r2"], reverse=True)
    winner = candidates[0]

    leaderboard = [
        {
            "Rank": f"#{i+1}",
            "Algorithm": c["algorithm"],
            "Val R² (15%)": round(float(c["val_r2"]), 4),
            "Test R² (15%)": round(float(c["test_r2"]), 4),
            "Test MSE": round(float(c["test_mse"]), 4),
            "Test MAE": round(float(c["test_mae"]), 4),
            "Verdict": "🏆 Winner" if i == 0 else "Evaluated",
        }
        for i, c in enumerate(candidates)
    ]

    # Feature Importance with coefficient extraction fallback
    feature_importances = {}
    if hasattr(winner["model"], "feature_importances_"):
        feature_importances = dict(
            zip(
                X_encoded.columns,
                [round(float(v), 4) for v in winner["model"].feature_importances_],
            )
        )
    elif hasattr(winner["model"], "coef_"):
        coef_vals = np.abs(winner["model"].coef_).ravel()
        feature_importances = dict(
            zip(
                X_encoded.columns,
                [round(float(v), 4) for v in coef_vals],
            )
        )

    return {
        "task": "Regression",
        "best_algorithm": winner["algorithm"],
        "r2_score": round(float(winner["test_r2"]), 4),
        "val_r2_score": round(float(winner["val_r2"]), 4),
        "mse": round(float(winner["test_mse"]), 4),
        "mae": round(float(winner["test_mae"]), 4),
        "leaderboard_df": pd.DataFrame(leaderboard),
        "feature_importances": feature_importances,
        "y_test": [float(val) for val in winner["y_test"]],
        "y_pred": [float(val) for val in winner["y_pred"]],
    }


# ==============================================================================
# 4. ASSOCIATION RULE MINING
# ==============================================================================
def run_association_mining(
    df: pd.DataFrame, min_support: float = 0.05, min_threshold: float = 0.6
) -> pd.DataFrame:
    if apriori is None:
        return pd.DataFrame()

    cat_df = df.select_dtypes(include=["object", "category", "bool"])
    if cat_df.empty:
        return pd.DataFrame()

    basket = pd.get_dummies(cat_df)
    try:
        frequent_itemsets = apriori(
            basket, min_support=min_support, use_colnames=True
        )
        if frequent_itemsets.empty:
            return pd.DataFrame()

        rules = association_rules(
            frequent_itemsets, metric="confidence", min_threshold=min_threshold
        )
        if rules.empty:
            return pd.DataFrame()

        clean_rules = pd.DataFrame({
            "Antecedents (If)": rules["antecedents"].apply(
                lambda x: ", ".join(list(x))
            ),
            "Consequents (Then)": rules["consequents"].apply(
                lambda x: ", ".join(list(x))
            ),
            "Support": rules["support"].round(3),
            "Confidence": rules["confidence"].round(3),
            "Lift": rules["lift"].round(3),
        }).sort_values(by="Lift", ascending=False)

        return clean_rules.head(10)
    except Exception:
        return pd.DataFrame()