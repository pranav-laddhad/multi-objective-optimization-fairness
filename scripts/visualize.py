import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

def get_pareto_front(df):
    """
    Extracts the non-dominated points from the results dataframe.
    Assumes F1 and DP are maximized (should be positive) and Complexity is minimized.
    """
    # PyMOO NDS works on minimization. 
    # We negate F1 and DP for the algorithm to find the front correctly.
    F = df[['F1', 'DP', 'Complexity']].values
    F_min = F.copy()
    F_min[:, 0] *= -1 # Maximize F1 -> Minimize -F1
    F_min[:, 1] *= -1 # Maximize DP -> Minimize -DP
    
    nds = NonDominatedSorting().do(F_min, only_non_dominated_front=True)
    return df.iloc[nds]

def plot_2d_tradeoffs(df, df_pareto=None):
    """Generates 2D scatter plots for F1 vs DP and F1 vs Complexity."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # F1 vs DP
    axes[0].scatter(df['F1'], df['DP'], alpha=0.3, label="All Solutions")
    if df_pareto is not None:
        axes[0].scatter(df_pareto['F1'], df_pareto['DP'], color='red', label="Pareto Front")
    axes[0].set_xlabel("F1 Score")
    axes[0].set_ylabel("Demographic Parity (Fairness)")
    axes[0].set_title("F1 vs DP Trade-off")
    axes[0].legend()

    # F1 vs Complexity
    axes[1].scatter(df['F1'], df['Complexity'], alpha=0.3, label="All Solutions")
    if df_pareto is not None:
        axes[1].scatter(df_pareto['F1'], df_pareto['Complexity'], color='red', label="Pareto Front")
    axes[1].set_xlabel("F1 Score")
    axes[1].set_ylabel("Complexity")
    axes[1].set_title("F1 vs Complexity Trade-off")
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

def plot_3d_pareto(df_pareto, color_by='F1', selected_idx=None):
    """Generates a 3D visualization of the Pareto Front."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Main Scatter
    sc = ax.scatter(
        df_pareto['F1'], 
        df_pareto['DP'], 
        df_pareto['Complexity'], 
        c=df_pareto[color_by], 
        cmap='viridis', 
        s=50, 
        alpha=0.8
    )

    # Highlight a selected solution (e.g., a Knee point)
    if selected_idx is not None:
        point = df_pareto.iloc[selected_idx]
        ax.scatter(
            point['F1'], point['DP'], point['Complexity'], 
            color='red', s=200, edgecolors='black', label='Selected Solution'
        )
        ax.legend()

    ax.set_xlabel("F1 Score")
    ax.set_ylabel("DP (Fairness)")
    ax.set_zlabel("Complexity")
    plt.colorbar(sc, label=color_by)
    plt.title(f"3D Pareto Front (Colored by {color_by})")
    plt.show()

def plot_knee_points(df_pareto, top_n=3):
    """Visualizes the top-N candidate solutions (Knee points)."""
    # Simple heuristic: sort by F1 and pick spread
    df_sorted = df_pareto.sort_values("F1", ascending=False)
    
    # Selecting indices for visual spread
    indices = np.linspace(0, len(df_sorted)-1, top_n).astype(int)
    candidates = df_sorted.iloc[indices]

    plt.figure(figsize=(8, 6))
    plt.scatter(df_pareto['F1'], df_pareto['DP'], alpha=0.5, label="Pareto Front")
    
    colors = ['red', 'orange', 'green', 'purple', 'brown']
    for i, (idx, row) in enumerate(candidates.iterrows()):
        plt.scatter(row['F1'], row['DP'], color=colors[i % len(colors)], s=150, 
                    label=f"Candidate {i+1} (Comp: {row['Complexity']:.2f})")

    plt.xlabel("F1 Score")
    plt.ylabel("Demographic Parity")
    plt.title("Knee Point Analysis")
    plt.legend()
    plt.show()

def main():
    try:
        df = pd.read_csv("data/pareto_results.csv")
    except FileNotFoundError:
        print("Error: data/pareto_results.csv not found. Please run run_optimization.py first.")
        return

    # 1. Identify Pareto Front
    print("Extracting Pareto Front...")
    df_pareto = get_pareto_front(df)
    print(f"Total Solutions: {len(df)} | Pareto Points: {len(df_pareto)}")

    # 2. Run All Visualizations
    print("Generating 2D Trade-off Plots...")
    plot_2d_tradeoffs(df, df_pareto)

    print("Generating 3D Pareto Front...")
    plot_3d_pareto(df_pareto, color_by='F1')

    print("Generating Knee Point Visualization...")
    plot_knee_points(df_pareto)

if __name__ == "__main__":
    main()