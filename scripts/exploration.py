import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import euclidean_distances
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

# Import your custom data loader
from src.data_loader import load_and_preprocess_data

def individual_fairness_score(X, y_pred, sample_size=1000):
    """
    Calculates consistency: similar individuals should receive similar outcomes.
    """
    np.random.seed(42)
    idx = np.random.choice(len(X), size=min(sample_size, len(X)), replace=False)
    
    X_sample = X.iloc[idx].values
    y_sample = y_pred[idx]

    dists = euclidean_distances(X_sample)
    np.fill_diagonal(dists, np.inf)

    # Define 'similarity' as the bottom 5th percentile of distances
    threshold = np.percentile(dists, 5)
    similar = dists < threshold

    # Absolute difference in predictions between similar neighbors
    pred_diff = np.abs(y_sample[:, None] - y_sample[None, :])
    
    # Fairness is 1 minus the mean difference among similar pairs
    score = 1 - np.mean(pred_diff[similar])
    return score

def run_exploration():
    # 1. Load Data
    X_train, X_test, y_train, y_test, s_train, s_test = load_and_preprocess_data()
    
    # 2. Statistical Profiling
    print("\n=== SENSITIVE ATTRIBUTE DISTRIBUTION (RACE) ===")
    print(s_train.value_counts(normalize=True))
    
    # Reconstructing a temporary DF for visualization purposes
    temp_df = X_train.copy()
    temp_df['race_grouped'] = s_train
    temp_df['income'] = y_train

    # Visualization: Income vs Race
    plt.figure(figsize=(8, 5))
    sns.countplot(data=temp_df, x="income", hue="race_grouped")
    plt.title("Income vs Race (Grouped)")
    plt.show()

    # 3. Baseline Model (Sanity Check)
    print("\n--- Running Baseline Model ---")
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    # 4. Compute Metrics
    f1 = f1_score(y_test, y_pred)
    dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=s_test)
    eo_diff = equalized_odds_difference(y_test, y_pred, sensitive_features=s_test)
    
    # Individual Fairness setup
    distance_cols = [col for col in ["age", "education.num", "hours.per.week"] if col in X_test.columns]
    if_score = individual_fairness_score(X_test[distance_cols], y_pred)

    print(f"Baseline F1: {f1:.4f}")
    print(f"Demographic Parity Score (1-abs_diff): {1 - abs(dp_diff):.4f}")
    print(f"Equalized Odds Score (1-abs_diff): {1 - abs(eo_diff):.4f}")
    print(f"Individual Fairness Score: {if_score:.4f}")

    # 5. Group-wise Performance
    print("\n=== GROUP-WISE METRICS (RACE) ===")
    for group in np.unique(s_test):
        idx = (s_test == group)
        print(f"Group: {group:15} | F1: {f1_score(y_test[idx], y_pred[idx]):.4f} | Samples: {np.sum(idx)}")

    # 6. Threshold Trade-off Analysis (The "MOO" Preparation)
    print("\n--- Generating Threshold Trade-offs ---")
    thresholds = np.linspace(0.2, 0.8, 40)
    tradeoff_results = []

    for t in thresholds:
        y_pred_t = (probs >= t).astype(int)
        
        f_t = f1_score(y_test, y_pred_t)
        dp_t = 1 - abs(demographic_parity_difference(y_test, y_pred_t, sensitive_features=s_test))
        eo_t = 1 - abs(equalized_odds_difference(y_test, y_pred_t, sensitive_features=s_test))
        if_t = individual_fairness_score(X_test[distance_cols], y_pred_t)
        
        tradeoff_results.append((f_t, dp_t, eo_t, if_t))

    tradeoff_results = np.array(tradeoff_results)

    # 7. Visualization of Trade-offs
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # F1 vs DP
    axes[0].scatter(tradeoff_results[:,0], tradeoff_results[:,1], color='blue')
    axes[0].set_xlabel("F1 Score")
    axes[0].set_ylabel("Demographic Parity (Higher is Better)")
    axes[0].set_title("F1 vs DP Trade-off")

    # F1 vs EO
    axes[1].scatter(tradeoff_results[:,0], tradeoff_results[:,2], color='green')
    axes[1].set_xlabel("F1 Score")
    axes[1].set_ylabel("Equalized Odds (Higher is Better)")
    axes[1].set_title("F1 vs EO Trade-off")

    # F1 vs IF
    axes[2].scatter(tradeoff_results[:,0], tradeoff_results[:,3], color='red')
    axes[2].set_xlabel("F1 Score")
    axes[2].set_ylabel("Individual Fairness")
    axes[2].set_title("F1 vs Individual Fairness")

    plt.tight_layout()
    plt.show()

    # 8. Correlation Check
    print("\n=== OBJECTIVE CORRELATION (F1, DP, EO, IF) ===")
    corr_matrix = np.corrcoef(tradeoff_results.T)
    print(pd.DataFrame(corr_matrix, 
                       columns=['F1', 'DP', 'EO', 'IF'], 
                       index=['F1', 'DP', 'EO', 'IF']))

if __name__ == "__main__":
    run_exploration()