from typing import TypedDict, Optional, Dict, Any, List
import pandas as pd
from sklearn.model_selection import train_test_split
from langgraph.graph import StateGraph, END

# Import backend modules
from data_analysis import analyze_and_plan, FullPreprocessingPlan
from cleaning_preprocessing import execute_cleaning_pipeline
from model_training import train_all_models, select_best_model_with_llm, ModelVerdict
from optimizer import optimize_best_model


# =====================================================================
# 1. STATE DEFINITION
# =====================================================================

class AgentState(TypedDict):
    csv_path: str
    user_goal: str
    run_optimization: bool

    # Artifacts & Intermediates
    raw_df: Optional[pd.DataFrame]
    plan: Optional[FullPreprocessingPlan]
    cleaned_df: Optional[pd.DataFrame]
    benchmark_results: Optional[Dict[str, Dict[str, float]]]
    trained_models: Optional[Dict[str, Any]]
    verdict: Optional[ModelVerdict]

    # Optimization Artifacts
    final_model: Optional[Any]
    final_score: Optional[float]
    optimization_adopted: Optional[bool]
    best_params: Optional[Dict[str, Any]]
    logs: List[str]


# =====================================================================
# 2. GRAPH NODES
# =====================================================================

def analyze_node(state: AgentState) -> Dict[str, Any]:
    logs = list(state.get("logs", []))
    logs.append(f"[State Machine] Analyzing dataset: {state['csv_path']}")
    
    raw_df = pd.read_csv(state["csv_path"])
    plan = analyze_and_plan(state["csv_path"], state["user_goal"])
    logs.append(f"[State Machine] Identified Problem: {plan.problem_type}, Metric: {plan.primary_metric}")
    
    return {
        "raw_df": raw_df,
        "plan": plan,
        "logs": logs
    }


def preprocess_node(state: AgentState) -> Dict[str, Any]:
    logs = list(state.get("logs", []))
    plan = state["plan"]
    raw_df = state["raw_df"]
    
    logs.append(f"[State Machine] Executing {len(plan.steps)} preprocessing transformations...")
    cleaned_df = execute_cleaning_pipeline(raw_df, [s.model_dump() for s in plan.steps])
    logs.append(f"[State Machine] Cleaned data ready with shape: {cleaned_df.shape}")
    
    return {
        "cleaned_df": cleaned_df,
        "logs": logs
    }


def train_benchmark_node(state: AgentState) -> Dict[str, Any]:
    logs = list(state.get("logs", []))
    plan = state["plan"]
    cleaned_df = state["cleaned_df"]
    
    logs.append("[State Machine] Training candidate benchmark models...")
    results_table, trained_models = train_all_models(
        cleaned_df=cleaned_df,
        problem_type=plan.problem_type,
        target_column=plan.target_column
    )
    logs.append(f"[State Machine] Successfully benchmarked {len(results_table)} algorithms.")
    
    return {
        "benchmark_results": results_table,
        "trained_models": trained_models,
        "logs": logs
    }


def select_model_node(state: AgentState) -> Dict[str, Any]:
    logs = list(state.get("logs", []))
    plan = state["plan"]
    results_table = state["benchmark_results"]
    
    logs.append("[State Machine] Selecting top performing algorithm...")
    verdict = select_best_model_with_llm(
        results_table=results_table,
        problem_type=plan.problem_type,
        user_goal=state["user_goal"],
        primary_metric=plan.primary_metric
    )
    logs.append(f"[State Machine] Winner selected: {verdict.best_model_name} ({verdict.primary_metric_name} = {verdict.primary_metric_value})")
    
    return {
        "verdict": verdict,
        "final_model": state["trained_models"][verdict.best_model_name],
        "final_score": verdict.primary_metric_value,
        "optimization_adopted": False,
        "best_params": {},
        "logs": logs
    }


def optimize_node(state: AgentState) -> Dict[str, Any]:
    logs = list(state.get("logs", []))
    verdict = state["verdict"]
    cleaned_df = state["cleaned_df"]
    plan = state["plan"]
    trained_models = state["trained_models"]
    
    logs.append(f"[State Machine] Initiating Optuna fine-tuning for '{verdict.best_model_name}'...")
    
    # Partition features and target
    X = cleaned_df.drop(columns=[plan.target_column])
    y = cleaned_df[plan.target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    baseline_model = trained_models[verdict.best_model_name]
    final_model, final_score, improved, best_params = optimize_best_model(
        model_name=verdict.best_model_name,
        baseline_model=baseline_model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        problem_type=plan.problem_type
    )
    
    logs.append(f"[State Machine] Optimization outcome - Improved: {improved}, Final Score: {final_score}")
    
    return {
        "final_model": final_model,
        "final_score": final_score,
        "optimization_adopted": improved,
        "best_params": best_params,
        "logs": logs
    }


# =====================================================================
# 3. CONDITIONAL ROUTING & GRAPH COMPILATION
# =====================================================================

def should_optimize_condition(state: AgentState) -> str:
    """Evaluates whether to branch into Optuna optimization or complete execution."""
    if state.get("run_optimization", False):
        return "optimize_node"
    return END


def build_ml_graph():
    """Constructs and compiles the end-to-end LangGraph state machine."""
    workflow = StateGraph(AgentState)

    # Register Nodes
    workflow.add_node("analyze_node", analyze_node)
    workflow.add_node("preprocess_node", preprocess_node)
    workflow.add_node("train_benchmark_node", train_benchmark_node)
    workflow.add_node("select_model_node", select_model_node)
    workflow.add_node("optimize_node", optimize_node)

    # Connect Edges
    workflow.set_entry_point("analyze_node")
    workflow.add_edge("analyze_node", "preprocess_node")
    workflow.add_edge("preprocess_node", "train_benchmark_node")
    workflow.add_edge("train_benchmark_node", "select_model_node")

    # Conditional Branching: Run optimization or terminate
    workflow.add_conditional_edges(
        "select_model_node",
        should_optimize_condition,
        {
            "optimize_node": "optimize_node",
            END: END
        }
    )
    workflow.add_edge("optimize_node", END)

    return workflow.compile()