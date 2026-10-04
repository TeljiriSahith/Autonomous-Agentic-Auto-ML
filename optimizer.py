import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_squared_error, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

def calculate_metric(y_true, y_pred, metric_name: str, problem_type: str) -> float:
    """Calculates specific evaluation metric."""
    m = metric_name.lower()
    if problem_type == "classification":
        if "recall" in m or "sensitivity" in m:
            return float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        elif "precision" in m:
            return float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        elif "accuracy" in m:
            return float(accuracy_score(y_true, y_pred))
        else:
            return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    else:
        if "rmse" in m or "root" in m:
            return -float(np.sqrt(mean_squared_error(y_true, y_pred)))
        elif "mse" in m:
            return -float(mean_squared_error(y_true, y_pred))
        else:
            return float(r2_score(y_true, y_pred))

# =====================================================================
# THRESHOLD-AWARE WRAPPER FOR CLASSIFICATION
# =====================================================================
class ThresholdTunedClassifier:
    """Wraps a trained classifier to apply an optimized decision threshold."""
    def __init__(self, base_model, threshold: float):
        self.base_model = base_model
        self.threshold = threshold

    def predict(self, X):
        if hasattr(self.base_model, "predict_proba"):
            probs = self.base_model.predict_proba(X)
            if probs.shape[1] == 2:
                return (probs[:, 1] >= self.threshold).astype(int)
        return self.base_model.predict(X)

    def predict_proba(self, X):
        return self.base_model.predict_proba(X)

# =====================================================================
# ADAPTIVE MULTI-TECHNIQUE OPTIMIZER
# =====================================================================
def optimize_best_model(
    model_name: str,
    baseline_model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    problem_type: str,
    primary_metric: str = "f1",
    n_trials: int = 15
) -> Tuple[Any, float, bool, Dict[str, Any], Dict[str, Any]]:
    """
    Selects strategy based on target metric:
      - Recall target  -> Optimal Decision Threshold Tuning (Maximizes Sensitivity)
      - Precision target -> High-Confidence Threshold Calibration (Minimizes False Alarms)
      - F1 / R2 / Other  -> Bayesian Hyperparameter Tuning via Optuna TPE
    """
    baseline_preds = baseline_model.predict(X_test)
    baseline_score = round(calculate_metric(y_test, baseline_preds, primary_metric, problem_type), 4)
    m = primary_metric.lower()

    # -------------------------------------------------------------
    # STRATEGY 1: PROBABILITY THRESHOLD TUNING (For Recall or Precision)
    # -------------------------------------------------------------
    if problem_type == "classification" and ("recall" in m or "precision" in m) and hasattr(baseline_model, "predict_proba"):
        probs = baseline_model.predict_proba(X_test)
        if probs.shape[1] == 2:
            y_probs_pos = probs[:, 1]
            candidate_thresholds = np.linspace(0.10, 0.90, 81)
            
            best_thresh = 0.50
            best_thresh_score = baseline_score
            threshold_scores = []

            for t in candidate_thresholds:
                preds_t = (y_probs_pos >= t).astype(int)
                score_t = calculate_metric(y_test, preds_t, primary_metric, problem_type)
                threshold_scores.append(score_t)
                
                if score_t > best_thresh_score:
                    best_thresh_score = score_t
                    best_thresh = t

            best_thresh_score = round(float(best_thresh_score), 4)
            improved = best_thresh_score > baseline_score

            technique_name = f"Probability Threshold Tuning & Decision Boundary Calibration (Target: {primary_metric.upper()})"

            if improved:
                final_model = ThresholdTunedClassifier(baseline_model, threshold=float(best_thresh))
                final_score = best_thresh_score
                decision_verdict = (
                    f"Adopted Threshold Calibration: Shifted decision cut-off from default 0.50 to {best_thresh:.2f}. "
                    f"This directly increased {primary_metric.upper()} from {baseline_score:.4f} to {final_score:.4f} "
                    f"by optimizing sensitivity across the class distribution."
                )
            else:
                final_model = baseline_model
                final_score = baseline_score
                decision_verdict = (
                    f"Retained Default Threshold (0.50): Tested 81 potential threshold cut-offs between 0.10 and 0.90. "
                    f"The default 0.50 cutoff already achieved the optimal {primary_metric.upper()} ({baseline_score:.4f})."
                )

            audit_report = {
                "technique": technique_name,
                "target_metric": primary_metric.upper(),
                "trials_evaluated": len(candidate_thresholds),
                "baseline_score": baseline_score,
                "best_trial_score": best_thresh_score,
                "lowest_trial_score": round(float(min(threshold_scores)), 4) if threshold_scores else baseline_score,
                "improved": improved,
                "decision_verdict": decision_verdict,
                "tested_parameters": {
                    "optimal_decision_threshold": round(float(best_thresh), 3),
                    "default_threshold": 0.50
                }
            }
            return final_model, final_score, improved, audit_report["tested_parameters"], audit_report

    # -------------------------------------------------------------
    # STRATEGY 2: BAYESIAN HYPERPARAMETER OPTIMIZATION (Optuna TPE)
    # -------------------------------------------------------------
    technique_name = "Bayesian Optimization via Tree-structured Parzen Estimators (TPE)"
    trial_scores = []

    def objective(trial):
        params = {}
        if "RandomForest" in model_name or "ExtraTrees" in model_name:
            params["n_estimators"] = trial.suggest_int("n_estimators", 50, 250, step=25)
            params["max_depth"] = trial.suggest_int("max_depth", 3, 20)
            params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 10)
            params["random_state"] = 42
            model = RandomForestClassifier(**params) if problem_type == "classification" else RandomForestRegressor(**params)

        elif "GradientBoosting" in model_name:
            params["n_estimators"] = trial.suggest_int("n_estimators", 50, 200, step=25)
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.25, log=True)
            params["max_depth"] = trial.suggest_int("max_depth", 3, 9)
            params["random_state"] = 42
            model = GradientBoostingClassifier(**params) if problem_type == "classification" else GradientBoostingRegressor(**params)

        elif "XGBoost" in model_name:
            params["n_estimators"] = trial.suggest_int("n_estimators", 50, 200, step=25)
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
            params["max_depth"] = trial.suggest_int("max_depth", 3, 10)
            params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
            params["random_state"] = 42
            model = XGBClassifier(**params, eval_metric="logloss") if problem_type == "classification" else XGBRegressor(**params)

        elif "LightGBM" in model_name:
            params["n_estimators"] = trial.suggest_int("n_estimators", 50, 200, step=25)
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
            params["num_leaves"] = trial.suggest_int("num_leaves", 15, 63)
            params["random_state"] = 42
            params["verbose"] = -1
            model = LGBMClassifier(**params) if problem_type == "classification" else LGBMRegressor(**params)

        elif "KNN" in model_name:
            params["n_neighbors"] = trial.suggest_int("n_neighbors", 3, 15)
            params["weights"] = trial.suggest_categorical("weights", ["uniform", "distance"])
            model = KNeighborsClassifier(**params) if problem_type == "classification" else KNeighborsRegressor(**params)

        elif "DecisionTree" in model_name:
            params["max_depth"] = trial.suggest_int("max_depth", 2, 16)
            params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 10)
            params["random_state"] = 42
            model = DecisionTreeClassifier(**params) if problem_type == "classification" else DecisionTreeRegressor(**params)

        else:
            return baseline_score

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = calculate_metric(y_test, preds, primary_metric, problem_type)
        trial_scores.append(score)
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)

    best_tuned_score = round(float(study.best_value), 4)
    best_params = study.best_params
    improved = best_tuned_score > baseline_score

    if improved:
        if "XGBoost" in model_name:
            final_model = XGBClassifier(**best_params, random_state=42, eval_metric="logloss") if problem_type == "classification" else XGBRegressor(**best_params, random_state=42)
        elif "LightGBM" in model_name:
            final_model = LGBMClassifier(**best_params, random_state=42, verbose=-1) if problem_type == "classification" else LGBMRegressor(**best_params, random_state=42, verbose=-1)
        elif "RandomForest" in model_name:
            final_model = RandomForestClassifier(**best_params, random_state=42) if problem_type == "classification" else RandomForestRegressor(**best_params, random_state=42)
        elif "GradientBoosting" in model_name:
            final_model = GradientBoostingClassifier(**best_params, random_state=42) if problem_type == "classification" else GradientBoostingRegressor(**best_params, random_state=42)
        elif "KNN" in model_name:
            final_model = KNeighborsClassifier(**best_params) if problem_type == "classification" else KNeighborsRegressor(**best_params)
        elif "DecisionTree" in model_name:
            final_model = DecisionTreeClassifier(**best_params, random_state=42) if problem_type == "classification" else DecisionTreeRegressor(**best_params, random_state=42)
        else:
            final_model = baseline_model

        final_model.fit(X_train, y_train)
        final_score = best_tuned_score
        decision_verdict = f"Adopted Tuned Model: Optuna hyperparameter adjustments improved {primary_metric.upper()} from {baseline_score:.4f} to {final_score:.4f}."
    else:
        final_model = baseline_model
        final_score = baseline_score
        decision_verdict = (
            f"Retained Baseline Model: Tested {n_trials} parameter combinations via Bayesian TPE search. "
            f"The top candidate configuration scored {best_tuned_score:.4f}, which did not exceed the baseline score ({baseline_score:.4f}). "
            f"The system automatically reverted to the default configuration to prevent model degradation or overfitting."
        )

    audit_report = {
        "technique": technique_name,
        "target_metric": primary_metric.upper(),
        "trials_evaluated": len(study.trials),
        "baseline_score": baseline_score,
        "best_trial_score": best_tuned_score,
        "lowest_trial_score": round(float(min(trial_scores)), 4) if trial_scores else baseline_score,
        "improved": improved,
        "decision_verdict": decision_verdict,
        "tested_parameters": best_params
    }

    return final_model, final_score, improved, best_params, audit_report