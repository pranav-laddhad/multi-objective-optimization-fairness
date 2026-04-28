import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from pymoo.core.problem import Problem

# Import logic from your metrics file
from src.metrics import evaluate_objectives, predict_with_group_thresholds

def train_candidate_model(model_type, reg, n_trees, depth, min_samples_leaf, class_weight_scale, X_tr, y_tr):
    """
    Handles the specific training logic for both Logistic Regression and Random Forest.
    """
    class_weight = {0: 1.0, 1: class_weight_scale} if class_weight_scale > 1 else None

    if model_type == 0:  # Logistic Regression
        model = LogisticRegression(
            C=1/reg,
            max_iter=300,
            class_weight=class_weight,
            solver="liblinear",
            random_state=42
        )
    else:  # Random Forest
        model = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=42
        )
    
    model.fit(X_tr, y_tr)
    return model

class FairnessMOO(Problem):
    """
    The PyMOO Problem class that wraps the cross-validation and evaluation logic.
    """
    def __init__(self, X_train, y_train, s_train, race_groups):
        self.X_train = X_train
        self.y_train = y_train
        self.s_train = s_train
        self.race_groups = race_groups
        
        # Decision Variables: 
        # [model_type, reg, n_trees, depth, min_samples_leaf, class_weight_scale, ...thresholds...]
        n_vars = 6 + len(race_groups)
        
        # Bounds exactly as per your notebook
        xl = np.array([0, 1e-4, 100, 5, 1, 0.5] + [0.3] * len(race_groups))
        xu = np.array([1, 10, 500, 25, 10, 3.0] + [0.7] * len(race_groups))
        
        super().__init__(n_var=n_vars, n_obj=3, xl=xl, xu=xu)

    def _evaluate_single_solution(self, x):
        """
        Runs the 3-fold stratified CV for a single set of hyperparameters.
        """
        model_type = int(round(x[0]))
        reg = x[1]
        n_trees = int(x[2])
        depth = int(x[3])
        min_samples_leaf = int(x[4])
        class_weight_scale = x[5]
        
        # Extract dynamic thresholds from the decision vector
        thresholds = np.clip(x[6:], 0.3, 0.7)
        thresholds_dict = dict(zip(self.race_groups, thresholds))

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        f1s, dps, comps = [], [], []

        for train_idx, val_idx in skf.split(self.X_train, self.y_train):
            X_tr, X_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_tr, y_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
            s_tr, s_val = self.s_train.iloc[train_idx], self.s_train.iloc[val_idx]

            # Train
            model = train_candidate_model(
                model_type, reg, n_trees, depth, 
                min_samples_leaf, class_weight_scale, X_tr, y_tr
            )

            # Predict using group-specific thresholds
            y_pred = predict_with_group_thresholds(model, X_val, s_val, thresholds_dict)

            # Reject degenerate solutions (too many/few positive predictions)
            pos_rate = np.mean(y_pred)
            if pos_rate < 0.05 or pos_rate > 0.95:
                continue

            # Evaluate (using the modular metrics script)
            f1, dp, comp = evaluate_objectives(y_val, y_pred, s_val, model, model_type)
            
            f1s.append(f1)
            dps.append(dp)
            comps.append(comp)

        # If no fold was valid, return a heavy penalty
        if len(f1s) == 0:
            return -0.0, -0.0, 1e6 

        return np.mean(f1s), np.mean(dps), np.mean(comps)

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Batch evaluation for the population.
        """
        results = []
        for x in X:
            f1, dp, comp = self._evaluate_single_solution(x)
            # PyMOO minimizes everything
            results.append([-f1, -dp, comp])
            
        out["F"] = np.array(results)