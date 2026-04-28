import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

# Import custom modules
from src.data_loader import load_and_preprocess_data
from src.metrics import evaluate_objectives, predict_with_group_thresholds

def train_model(model_type, reg, n_trees, depth, min_samples_leaf, class_weight_scale, X_tr, y_tr):
    """
    Step 1 - Controlled Model Training
    """
    class_weight = {0: 1.0, 1: class_weight_scale} if class_weight_scale > 1 else None

    if model_type == 0:
        return LogisticRegression(
            C=1/reg,
            max_iter=300,
            class_weight=class_weight,
            solver="liblinear",
            random_state=42
        ).fit(X_tr, y_tr)
    else:
        return RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=42
        ).fit(X_tr, y_tr)

def evaluate_solution(x, X_train, y_train, s_train, race_groups):
    """
    Step 5 - Cross-Validated Evaluation
    """
    # Unpack decision variables
    model_type = int(round(x[0]))
    reg = x[1]
    n_trees = int(x[2])
    depth = int(x[3])
    min_samples_leaf = int(x[4])
    class_weight_scale = x[5]
    
    # Thresholds for each race group (x[6:])
    thresholds = np.clip(x[6:], 0.3, 0.7)
    thresholds_dict = dict(zip(race_groups, thresholds))

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    f1s, dps, comps = [], [], []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        s_tr, s_val = s_train.iloc[train_idx], s_train.iloc[val_idx]

        model = train_model(
            model_type, reg, n_trees, depth, 
            min_samples_leaf, class_weight_scale, X_tr, y_tr
        )

        y_pred = predict_with_group_thresholds(model, X_val, s_val, thresholds_dict)

        # Reject degenerate predictions
        pos_rate = np.mean(y_pred)
        if pos_rate < 0.05 or pos_rate > 0.95:
            continue

        f1, dp, comp = evaluate_objectives(y_val, y_pred, s_val, model, model_type)
        
        f1s.append(f1)
        dps.append(dp)
        comps.append(comp)

    if len(f1s) == 0:
        return 0.0, 0.0, 1e6 # Penalty for degenerate solutions

    return np.mean(f1s), np.mean(dps), np.mean(comps)

class FairnessMOO(Problem):
    """
    Step 6 - PyMOO Problem Definition
    """
    def __init__(self, X_train, y_train, s_train, race_groups):
        self.X_train = X_train
        self.y_train = y_train
        self.s_train = s_train
        self.race_groups = race_groups
        
        # n_var = 6 (params) + number of race groups (thresholds)
        n_vars = 6 + len(race_groups)
        
        # Define bounds
        xl = np.array([0, 1e-4, 100, 5, 1, 0.5] + [0.3] * len(race_groups))
        xu = np.array([1, 10, 500, 25, 10, 3.0] + [0.7] * len(race_groups))
        
        super().__init__(n_var=n_vars, n_obj=3, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        results = []
        for x in X:
            f1, dp, comp = evaluate_solution(
                x, self.X_train, self.y_train, self.s_train, self.race_groups
            )
            # Pymoo minimizes: maximize F1/DP -> minimize -F1/-DP
            results.append([-f1, -dp, comp])
        out["F"] = np.array(results)

def main():
    print("Loading data...")
    X_train, X_test, y_train, y_test, s_train, s_test = load_and_preprocess_data()
    race_groups = s_train.unique().tolist()
    
    print(f"Detected Race Groups: {race_groups}")

    # Initialize Problem
    problem = FairnessMOO(X_train, y_train, s_train, race_groups)

    # Step 7 - Run NSGA-II
    print("Starting NSGA-II Optimization...")
    algorithm = NSGA2(pop_size=100)
    res = minimize(
        problem,
        algorithm,
        ('n_gen', 80),
        seed=42,
        verbose=True
    )

    # Step 8 - Extract and Save Results
    F = res.F.copy()
    F[:, 0] *= -1  # Back to positive F1
    F[:, 1] *= -1  # Back to positive DP Score (Fairness)
    
    df_pareto = pd.DataFrame(F, columns=["F1", "DP", "Complexity"])
    
    # Save the decision variables (X) alongside the scores (F)
    # This allows you to reconstruct the "best" models later
    X_df = pd.DataFrame(res.X, columns=[f"v{i}" for i in range(res.X.shape[1])])
    results_full = pd.concat([df_pareto, X_df], axis=1)
    
    results_full.to_csv("data/pareto_results.csv", index=False)
    print("\nOptimization Complete. Results saved to data/pareto_results.csv")

if __name__ == "__main__":
    main()