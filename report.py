import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load your full grid search CSV
results_df = pd.read_csv('full_grid_search_results.csv')

# Remove duplicates (if any) on parameters before pivot
results_df = results_df.drop_duplicates(subset=['Stop Loss %', 'Profit Target %'])

metrics = {
    'Total Return %': 'Total Return %',
    'Total Trades': 'Total Trades',
    'Win Rate %': 'Win Rate %',
    'Profit Factor': 'Profit Factor',
    'ML Accuracy': 'ML Accuracy',
    'ML Precision': 'ML Precision',
    'ML Recall': 'ML Recall'
}

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.flatten()

for i, (title, col) in enumerate(metrics.items()):
    # Use pivot_table to handle any remaining duplicates gracefully
    pivot_table = results_df.pivot_table(
        index='Stop Loss %',
        columns='Profit Target %',
        values=col,
        aggfunc='mean'
    )
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt='.2f' if results_df[col].dtype == float else 'd',
        cmap='YlGnBu',
        ax=axes[i],
        cbar=True
    )
    axes[i].set_title(title)
    axes[i].set_xlabel('Profit Target %')
    axes[i].set_ylabel('Stop Loss %')

# Hide unused axes if any
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
