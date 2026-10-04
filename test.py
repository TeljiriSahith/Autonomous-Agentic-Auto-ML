import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from data_analysis import analyze_and_plan
from cleaning_preprocessing import execute_cleaning_pipeline
from model_training import train_all_models, select_best_model_with_llm
from optimizer import optimize_best_model

# -------------------------------------------------------------
# 1. Create a Realistic Synthetic Dataset (60 rows)
# -------------------------------------------------------------
np.random.seed(42)
n_samples = 60

data = {
    "customer_id": [f"CUST_{i:04d}" for i in range(n_samples)],
    "age": np.random.randint(18, 70, size=n_samples),
    "tenure": np.random.randint(1, 10, size=n_samples),
    "balance": np.random.uniform(500.0, 15000.0, size=n_samples).round(2),
    "country": np.random.choice(["Germany", "France", "Spain"], size=n_samples),
}

# Generate churn with mild correlation to balance and age
churn_prob = (data["age"] > 40).astype(int) * 0.4 + (data["balance"] > 7000).astype(int) * 0.4
data["churn"] = (churn_prob + np.random.uniform(0, 0.3, size=n_samples) > 0.5).astype(int)

df_raw = pd.DataFrame(data)
csv_filename = "churn_benchmark_dataset.csv"
df_raw.to_csv(csv_filename, index=False)
print(f"Generated realistic sample dataset: {csv_filename} ({len(df_raw)} rows)")

# -------------------------------------------------------------
# 2. Stage 1: LLM Data Analysis & Plan Generation
# -------------------------------------------------------------
print("\n" + "="*50)
print("STAGE 1: LLM Data Analysis")
print("="*50)
user_goal = "Predict whether a customer will churn based on profile and balance"
plan = analyze_and_plan(csv_filename, user_goal)

print(f"Problem Type: {plan.problem_type}")
print(f"Target Column: {plan.target_column}")
print(f"Primary Metric Selected: {plan.primary_metric}")
print("Action Plan Steps:")
for s in plan.steps:
    print(f"  - Column '{s.column}' -> {s.operation} ({s.reason})")

# -------------------------------------------------------------
# 3. Stage 2: Atomic Cleaning & Preprocessing Execution
# -------------------------------------------------------------
print("\n" + "="*50)
print("STAGE 2: Cleaning & Preprocessing Execution")
print("="*50)
cleaned_df = execute_cleaning_pipeline(df_raw, [s.model_dump() for s in plan.steps])
print(f"Cleaned DataFrame Shape: {cleaned_df.shape}")
print(f"Cleaned Columns: {list(cleaned_df.columns)}")

# -------------------------------------------------------------
# 4. Stage 3: Multi-Model Benchmark Training
# -------------------------------------------------------------
print("\n" + "="*50)
print("STAGE 3: Benchmark Model Training")
print("="*50)
results_table, trained_models = train_all_models(
    cleaned_df=cleaned_df,
    problem_type=plan.problem_type,
    target_column=plan.target_column
)

benchmark_df = pd.DataFrame(results_table).T
print("\nBenchmark Summary:")
if plan.problem_type == "classification":
    print(benchmark_df[["Accuracy", "Precision", "Recall", "F1_Score"]])
elif plan.problem_type == "regression":
    print(benchmark_df[["R2_Score", "MSE", "RMSE"]])
else:
    print(benchmark_df)

# -------------------------------------------------------------
# 5. Stage 4: Winner Selection & LLM Commentary
# -------------------------------------------------------------
print("\n" + "="*50)
print("STAGE 4: Winner Selection")
print("="*50)
verdict = select_best_model_with_llm(
    results_table=results_table,
    problem_type=plan.problem_type,
    user_goal=user_goal,
    primary_metric=plan.primary_metric
)
print(f"Winning Model : {verdict.best_model_name}")
print(f"Target Metric : {verdict.primary_metric_name} = {verdict.primary_metric_value}")
print(f"LLM Reasoning :\n{verdict.reasoning}")

# -------------------------------------------------------------
# 6. Stage 5: Optuna Hyperparameter Optimization & Rollback Check
# -------------------------------------------------------------
print("\n" + "="*50)
print("STAGE 5: Fine-Tuning & Hyperparameter Optimization")
print("="*50)

# Prepare feature matrix and target vector for optimization
X = cleaned_df.drop(columns=[plan.target_column])
y = cleaned_df[plan.target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

winning_baseline_model = trained_models[verdict.best_model_name]

final_model, final_score, improved, best_params = optimize_best_model(
    model_name=verdict.best_model_name,
    baseline_model=winning_baseline_model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    problem_type=plan.problem_type
)

print("\n" + "="*50)
print("FINAL PIPELINE OUTCOME")
print("="*50)
print(f"Selected Model        : {verdict.best_model_name}")
print(f"Final Validation Score: {final_score}")
print(f"Improvement Adopted   : {improved}")
if improved:
    print(f"Best Hyperparameters : {best_params}")
print("Pipeline verification completed successfully!")