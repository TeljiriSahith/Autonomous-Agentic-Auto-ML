import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_squared_error,
    silhouette_score, davies_bouldin_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from sklearn.cluster import KMeans, Birch, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture

from llm import get_gemini_client

# =====================================================================
# 1. MODEL REPOSITORIES
# =====================================================================

CLASSIFICATION_MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "NaiveBayes": GaussianNB(),
    # CalibratedClassifierCV eliminates the deprecated probability=True warning in SVC
    "SVC": CalibratedClassifierCV(estimator=SVC(random_state=42)),
    "MLP_NeuralNet": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
    "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
}

REGRESSION_MODELS = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "AdaBoost": AdaBoostRegressor(random_state=42),
    "KNN": KNeighborsRegressor(),
    "SVR": SVR(),
    "MLP_NeuralNet": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42),
    "XGBoost": XGBRegressor(random_state=42),
    "LightGBM": LGBMRegressor(random_state=42, verbose=-1),
    "CatBoost": CatBoostRegressor(verbose=0, random_state=42)
}

CLUSTERING_MODELS = {
    "KMeans": KMeans(n_clusters=3, random_state=42, n_init='auto'),
    "Birch": Birch(n_clusters=3),
    "Agglomerative": AgglomerativeClustering(n_clusters=3),
    "GaussianMixture": GaussianMixture(n_components=3, random_state=42),
    "DBSCAN": DBSCAN(eps=0.5, min_samples=3)
}

# =====================================================================
# 2. TRAINING LOOP
# =====================================================================

def train_all_models(
    cleaned_df: pd.DataFrame, 
    problem_type: str, 
    target_column: Optional[str] = None
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    results_table = {}
    trained_objects = {}

    if problem_type in ["classification", "regression"]:
        if not target_column or target_column not in cleaned_df.columns:
            raise ValueError(f"Target column '{target_column}' missing.")

        X = cleaned_df.drop(columns=[target_column])
        y = cleaned_df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model_dict = CLASSIFICATION_MODELS if problem_type == "classification" else REGRESSION_MODELS

        for name, model in model_dict.items():
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                if problem_type == "classification":
                    acc = round(float(accuracy_score(y_test, preds)), 4)
                    prec = round(float(precision_score(y_test, preds, average='weighted', zero_division=0)), 4)
                    rec = round(float(recall_score(y_test, preds, average='weighted', zero_division=0)), 4)
                    f1 = round(float(f1_score(y_test, preds, average='weighted', zero_division=0)), 4)

                    results_table[name] = {
                        "Accuracy": acc,
                        "Precision": prec,
                        "Recall": rec,
                        "F1_Score": f1
                    }
                else:
                    r2 = round(float(r2_score(y_test, preds)), 4)
                    mse = round(float(mean_squared_error(y_test, preds)), 4)
                    rmse = round(float(np.sqrt(mse)), 4)

                    results_table[name] = {
                        "R2_Score": r2,
                        "MSE": mse,
                        "RMSE": rmse
                    }

                trained_objects[name] = model
                print(f"[Success] Trained {name}")
            except Exception as e:
                print(f"[Skipped] Model {name}: {e}")

    elif problem_type == "clustering":
        X = cleaned_df.copy()
        for name, model in CLUSTERING_MODELS.items():
            try:
                labels = model.fit_predict(X) if hasattr(model, "fit_predict") else model.fit(X).predict(X)
                unique_labels = set(labels) - {-1}
                if 1 < len(unique_labels) < len(X):
                    sil = round(float(silhouette_score(X, labels)), 4)
                    db = round(float(davies_bouldin_score(X, labels)), 4)
                else:
                    sil = -1.0
                    db = 999.0

                results_table[name] = {
                    "Silhouette_Score": sil,
                    "Davies_Bouldin": db,
                    "Clusters_Found": len(unique_labels)
                }
                trained_objects[name] = model
                print(f"[Success] Clustered with {name}")
            except Exception as e:
                print(f"[Skipped] Model {name}: {e}")

    return results_table, trained_objects

# =====================================================================
# 3. DETERMINISTIC WINNER SELECTION (NO HALLUCINATIONS)
# =====================================================================

class ModelVerdict(BaseModel):
    best_model_name: str
    primary_metric_name: str
    primary_metric_value: float
    reasoning: str

def select_best_model_with_llm(
    results_table: dict, 
    problem_type: str, 
    user_goal: str,
    primary_metric: str = "f1"
) -> ModelVerdict:
    """Selects winner deterministically by benchmark metric, then prompts LLM for qualitative reasoning."""
    df_res = pd.DataFrame(results_table).T

    # Map generic names to exact table column names
    metric_map = {
        "f1": "F1_Score",
        "precision": "Precision",
        "recall": "Recall",
        "accuracy": "Accuracy",
        "r2": "R2_Score",
        "rmse": "RMSE",
        "silhouette": "Silhouette_Score"
    }
    target_metric_col = metric_map.get(primary_metric.lower(), "F1_Score")

    if target_metric_col not in df_res.columns:
        target_metric_col = "F1_Score" if problem_type == "classification" else "R2_Score"

    # Deterministic winner selection (higher is better, except RMSE/MSE/Davies_Bouldin)
    if target_metric_col in ["RMSE", "MSE", "Davies_Bouldin"]:
        best_name = str(df_res[target_metric_col].idxmin())
    else:
        best_name = str(df_res[target_metric_col].idxmax())

    best_score = float(df_res.loc[best_name, target_metric_col])

    # LLM commentary prompt
    prompt = f"""
    The winning model selected for this {problem_type} task is '{best_name}' with a {target_metric_col} of {best_score}.
    User Goal: {user_goal}
    Target Metric: {target_metric_col}

    Full Benchmark Table:
    {df_res.to_string()}

    Provide a concise 2-sentence explanation of why {best_name} is best suited for this task given these specific metrics.
    """

    try:
        llm = get_gemini_client()
        response = llm.invoke(prompt)
        reasoning = response.content.strip()
    except Exception:
        reasoning = f"{best_name} achieved the top {target_metric_col} score of {best_score} across all tested candidate models."

    return ModelVerdict(
        best_model_name=best_name,
        primary_metric_name=target_metric_col,
        primary_metric_value=best_score,
        reasoning=reasoning
    )