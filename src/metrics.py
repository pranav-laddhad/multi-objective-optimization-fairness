import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import euclidean_distances
from fairlearn.metrics import demographic_parity_difference

def calculate_individual_fairness(X, y_pred, sample_size=1000):
    """
    Measures consistency: similar individuals should receive similar outcomes.
    Higher is better (1 - mean difference).
    """
    if len(X) == 0:
        return 0.0
        
    np.random.seed(42)
    idx = np.random.choice(len(X), size=min(sample_size, len(X)), replace=False)
    
    X_sample = X.iloc[idx].values
    y_sample = y_pred[idx]

    dists = euclidean_distances(X_sample)
    np.fill_diagonal(dists, np.inf)

    # Similarity threshold based on the bottom 5th percentile of distances
    threshold = np.percentile(dists, 5)
    similar = dists < threshold

    # Avoid division by zero if no neighbors found
    if not np.any(similar):
        return 1.0

    pred_diff = np.abs(y_sample[:, None] - y_sample[None, :])
    score = 1 - np.mean(pred_diff[similar])

    return score

def get_model_complexity(model, model_type):
    """
    Calculates complexity based on model type.
    model_type 0: Logistic Regression (L1-norm of coefficients)
    model_type 1: Random Forest (Log of total node count)
    """
    if model_type == 0:
        # Logistic Regression complexity
        return np.sum(np.abs(model.coef_))
    else:
        # Random Forest complexity (log-scaled to handle large node counts)
        total_nodes = sum(tree.tree_.node_count for tree in model.estimators_)
        return np.log(total_nodes + 1)

def predict_with_group_thresholds(model, X, sensitive_series, thresholds_dict):
    """
    Applies different probability thresholds for different groups (e.g., Race).
    """
    probs = model.predict_proba(X)[:, 1]
    
    # Map the sensitive attribute of each sample to its specific threshold
    # thresholds_dict should look like {'White': 0.5, 'Black': 0.45, ...}
    sample_thresholds = sensitive_series.map(thresholds_dict).values
    
    return (probs >= sample_thresholds).astype(int)

def evaluate_objectives(y_true, y_pred, sensitive_features, model, model_type, X_dist=None):
    """
    Returns the three primary objectives for the MOO loop.
    Returns: (F1, Demographic Parity, Complexity)
    Note: DP is returned as (1 - diff) so that higher is always better.
    """
    # 1. Performance
    f1 = f1_score(y_true, y_pred)

    # 2. Group Fairness (Higher is better)
    dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
    dp_score = 1 - abs(dp_diff)

    # 3. Complexity
    comp = get_model_complexity(model, model_type)
    
    # Optional: Individual Fairness (if X_dist is provided)
    if X_dist is not None:
        if_score = calculate_individual_fairness(X_dist, y_pred)
        return f1, dp_score, comp, if_score

    return f1, dp_score, comp