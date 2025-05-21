import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import pandas as pd

def plot_curves(y_true, y_prob, model_name):
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(f'{model_name}_roc_curve.png')
    plt.close()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.savefig(f'{model_name}_pr_curve.png')
    plt.close()

def plot_combined_roc(roc_curves):
    plt.figure(figsize=(8, 6))
    for model_name, (fpr, tpr, roc_auc) in roc_curves.items():
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.savefig('combined_roc_curve.png')
    plt.show()
    plt.close()

def plot_feature_importance(feat_imp: pd.Series, model_name: str, top_n: int = None, save_path: str = None):
    """
    Generate and save a bar plot of feature importances.

    Parameters:
    - feat_imp: pd.Series of importances (index=feature names).
    - model_name: identifier to label the plot title and filename.
    - top_n: if specified, plot only the top_n features by importance.
    - save_path: custom filepath to save the figure (defaults to '<model_name>_feature_importance.png').
    """
    # Optionally top N features
    if top_n is not None:
        feat_imp = feat_imp.nlargest(top_n)

    plt.figure(figsize=(10, 6))
    feat_imp.sort_values().plot.barh()
    plt.title(f'{model_name} Feature Importances')
    plt.xlabel('Importance')
    plt.tight_layout()

    # Determine filename
    filename = save_path or f'{model_name}_feature_importance.png'
    plt.savefig(filename)
    plt.close()
    print(f"Saved feature importance plot: {filename}")

    
def print_confusion_matrix(tp, tn, fp, fn):
    print("Confusion Matrix:")
    print(f"{'':<20}{'Predicted Negative':<20}{'Predicted Positive':<20}")
    print(f"{'Actual Negative':<20}{tn:<20}{fp:<20}")
    print(f"{'Actual Positive':<20}{fn:<20}{tp:<20}")
