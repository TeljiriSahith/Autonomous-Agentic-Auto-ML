import io
import pandas as pd
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from llm import get_gemini_client

class CleaningStep(BaseModel):
    column: str = Field(description="Exact column name")
    operation: str = Field(
        description="Must be one of: 'drop_column', 'impute_mean', 'impute_median', 'encode_one_hot', 'encode_label', 'scale_standard', 'scale_minmax'"
    )
    reason: str = Field(description="Short reason for this operation")

class FullPreprocessingPlan(BaseModel):
    problem_type: Literal["classification", "regression", "clustering"] = Field(
        description="Must be strictly: 'classification', 'regression', or 'clustering'"
    )
    target_column: Optional[str] = Field(description="Name of the target column, or None if clustering")
    primary_metric: str = Field(
        default="f1",
        description="Target evaluation metric: 'f1', 'recall', 'precision', 'accuracy', 'r2', 'rmse', or 'silhouette'"
    )
    steps: List[CleaningStep] = Field(
        default_factory=list,
        description="List of preprocessing steps. MUST NOT be empty."
    )

def generate_heuristic_fallback_steps(df: pd.DataFrame, target_col: Optional[str]) -> List[CleaningStep]:
    """Generates baseline preprocessing steps if the LLM leaves steps empty."""
    steps = []
    for col in df.columns:
        if col == target_col:
            continue
        
        # 1. Unique IDs / High-cardinality text -> drop
        if df[col].dtype == 'object' and (df[col].nunique() / len(df) > 0.5 or 'id' in col.lower()):
            steps.append(CleaningStep(column=col, operation="drop_column", reason="Identifier/high-cardinality feature"))
        # 2. Categorical features -> encode
        elif df[col].dtype == 'object' or (df[col].nunique() < 10 and not pd.api.types.is_numeric_dtype(df[col])):
            steps.append(CleaningStep(column=col, operation="encode_one_hot", reason="Categorical variable encoding"))
        # 3. Numeric continuous features -> scale
        elif pd.api.types.is_numeric_dtype(df[col]):
            steps.append(CleaningStep(column=col, operation="scale_standard", reason="Numerical feature scaling"))
            
    return steps

def analyze_and_plan(csv_path: str, user_goal: str) -> FullPreprocessingPlan:
    """Inspects dataset and determines task type, metric, and column-by-column preprocessing plan."""
    df = pd.read_csv(csv_path)

    buffer = io.StringIO()
    df.info(buf=buffer)
    info_text = buffer.getvalue()
    describe_text = df.describe(include='all').fillna("N/A").to_string()
    head_sample = df.head(3).to_string()

    prompt = f"""
You are an expert Lead Data Scientist. Analyze the dataset and user goal to build a complete preprocessing plan.

User Goal: {user_goal}

--- Dataset info() ---
{info_text}

--- Dataset describe() ---
{describe_text}

--- First 3 Rows ---
{head_sample}

CRITICAL RULES FOR METRIC SELECTION ('primary_metric'):
1. If the user explicitly mentions a metric (e.g. 'precision', 'recall', 'accuracy', 'f1', 'r2', 'rmse'), YOU MUST SELECT THAT METRIC.
2. Healthcare / Medical Diagnosis / Cancer / Heart Disease: -> 'recall'
3. Spam Detection / Phishing: -> 'precision'
4. Customer Churn / Imbalanced Binary Classification: -> 'f1'
5. Regression (Prices, sales, numbers): -> 'r2' or 'rmse'
6. Clustering: -> 'silhouette'

RULES FOR PREPROCESSING STEPS:
- 'steps' MUST NOT BE EMPTY.
- Identifier columns (e.g. 'customer_id', 'id') -> 'drop_column'
- Nominal categorical text -> 'encode_one_hot'
- Continuous numerical columns -> 'scale_standard'
- NEVER drop, scale, or encode the target_column!
"""

    llm = get_gemini_client()
    structured_llm = llm.with_structured_output(FullPreprocessingPlan)
    
    try:
        plan = structured_llm.invoke(prompt)
    except Exception as e:
        print(f"[Notice] LLM structured output parsing error: {e}")
        plan = FullPreprocessingPlan(
            problem_type="classification",
            target_column="churn" if "churn" in df.columns else df.columns[-1],
            primary_metric="f1",
            steps=[]
        )

    # Sanity checks for problem_type
    if plan.problem_type not in ["classification", "regression", "clustering"]:
        plan.problem_type = "classification"

    goal_lower = user_goal.lower()

    # PRIORITY 1: Explicit user metric command in prompt
    if "precision" in goal_lower:
        plan.primary_metric = "precision"
    elif "recall" in goal_lower or "sensitivity" in goal_lower:
        plan.primary_metric = "recall"
    elif "accuracy" in goal_lower:
        plan.primary_metric = "accuracy"
    elif "f1" in goal_lower or "f-score" in goal_lower or "f1-score" in goal_lower:
        plan.primary_metric = "f1"
    elif "rmse" in goal_lower or "root mean squared" in goal_lower:
        plan.primary_metric = "rmse"
    elif "r2" in goal_lower or "r-squared" in goal_lower:
        plan.primary_metric = "r2"

    # PRIORITY 2: Domain-specific heuristics if user didn't name a metric
    elif any(k in goal_lower for k in ["heart", "disease", "cancer", "patient", "diabetes", "medical", "diagnosis", "tumor", "stroke", "illness", "covid"]):
        plan.primary_metric = "recall"
    elif any(k in goal_lower for k in ["spam", "phishing", "fraud accusation"]):
        plan.primary_metric = "precision"
    elif any(k in goal_lower for k in ["churn", "retention", "attrition", "default"]):
        plan.primary_metric = "f1"
    elif not plan.primary_metric:
        plan.primary_metric = "f1" if plan.problem_type == "classification" else "r2"

    # Fallback if LLM returned an empty steps array
    if not plan.steps:
        print("-> LLM returned empty steps; applying automated feature-type fallback pipeline...")
        plan.steps = generate_heuristic_fallback_steps(df, plan.target_column)

    return plan