import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the results
with open('data/model_results.pkl', 'rb') as f:
    all_results = pickle.load(f)


summary_data = []
for result in all_results:
    summary_data.append({
        'Model': result['model_name'],
        'Accuracy': result['accuracy'],
        'F1-Weighted': result['f1_weighted'],
        'F1-Macro': result['f1_macro'],
        'AUC Score': result['auc_score']
    })

summary_df = pd.DataFrame(summary_data).sort_values('F1-Weighted', ascending=False)
print(summary_df)

#Performance Metrics Bar Chart:
plt.figure(figsize=(12, 6))
metrics = ['Accuracy', 'F1-Weighted', 'F1-Macro', 'AUC Score']
x = np.arange(len(summary_df))
width = 0.2

for i, metric in enumerate(metrics):
    plt.bar(x + i*width, summary_df[metric], width, label=metric, alpha=0.8)

plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(x + width*1.5, summary_df['Model'], rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Confusion Matrix Visualization:
def plot_confusion_matrix_from_results(results, class_names):
    """Plot confusion matrices for all models"""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

    if n_models == 1:
        axes = [axes]

    for ax, result in zip(axes, results):
        cm = result['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(result['model_name'])
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')

    plt.tight_layout()
    plt.show()

class_names = ["1","2","3","4","5"]
plot_confusion_matrix_from_results(all_results, class_names)

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# ROC Curve Comparison (for multi-class):
def plot_roc_curves(results, y_true, class_names):
    """Plot ROC curves for all models"""
    plt.figure(figsize=(10, 8))

    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=np.arange(len(class_names)))

    for result in results:
        y_pred_proba = result['y_pred_proba']

        # Compute micro-average ROC curve and ROC area
        fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{result['model_name']} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Micro-average ROC Curve Comparison')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()

# 1. Load your test labels
data = np.load('data/preprocessed_data.npz')
y_test = data['y_test']
plot_roc_curves(all_results, y_test, class_names)


def plot_model_metrics(results):
    """Plot overall metrics (accuracy, F1-weighted, etc.) across models"""
    metrics = ['accuracy', 'f1_weighted', 'f1_macro', 'auc_score']
    metric_names = ['Accuracy', 'F1-Weighted', 'F1-Macro', 'AUC Score']

    model_names = [result['model_name'] for result in results]
    n_models = len(model_names)

    plt.figure(figsize=(12, 6))
    x = np.arange(n_models)
    width = 0.2

    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        values = [result[metric] for result in results]
        plt.bar(x + i * width, values, width, label=metric_name, alpha=0.8)

    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x + width * 1.5, model_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_model_metrics(all_results)