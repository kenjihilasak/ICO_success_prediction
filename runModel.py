# run_model.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import make_scorer, roc_auc_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    make_scorer, roc_auc_score, f1_score,
    precision_recall_curve, roc_curve, auc, confusion_matrix
)

# Import preprocessor and RowCounter from pipeline
from pipeline import preprocessor, RowCounter

# Import custom plotting functions
from plotUtils import plot_curves, plot_combined_roc, plot_feature_importance, print_confusion_matrix

# ---- Constants ----
DATA_PATH = 'df_model.xlsx'
RESULTS_PATH = 'model_results.csv'
roc_curves = {}

# ---- Models and parameter grids ----
models = {
    'LogisticRegression': LogisticRegression(solver='liblinear', random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', verbosity=0,random_state=42)
}

param_grids = {
    'LogisticRegression': {'model__C': [0.1, 1.0, 10.0]},
    'RandomForest': {'model__n_estimators': [100, 200], 'model__max_depth': [None, 5, 10]},
    'XGBoost': {'model__n_estimators': [100, 200], 'model__learning_rate': [0.01, 0.1]}
}

# ---- TimeSeries split ----
ts_cv = TimeSeriesSplit(n_splits=5)

# Scoring metrics
scoring = {'auc': 'roc_auc', 'f1': make_scorer(f1_score)}

def main():
    # Load data
    df = pd.read_excel(DATA_PATH, engine='openpyxl')
    X = df.drop('success', axis=1)
    y = df['success']

    results = []

    for name, estimator in models.items():
        print(f"\n=== {name} ===")
        pipe = Pipeline([
            #('count_initial', RowCounter(f'{name} initial')),
            ('preprocessor', preprocessor),
            #('count_after', RowCounter(f'{name} after_prep')),
            ('model', estimator)
        ])

        grid = GridSearchCV(
            pipe,
            param_grid=param_grids[name],
            cv=ts_cv,
            scoring=scoring,
            refit='auc',
            return_train_score=False
        )
        grid.fit(X, y)

        # Best estimator and predictions
        best = grid.best_estimator_
        y_prob = best.predict_proba(X)[:, 1]
        y_pred = best.predict(X)

        # Compute metrics
        auc_score = roc_auc_score(y, y_prob)
        f1 = f1_score(y, y_pred)
        cm = confusion_matrix(y, y_pred).ravel()
        tn, fp, fn, tp = cm
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        # Precision-Recall AUC
        precision_vals, recall_vals, _ = precision_recall_curve(y, y_prob)
        pr_auc = auc(recall_vals, precision_vals)


        # Plot curves
        plot_curves(y, y_prob, name)

        # Feature importance / coefficients
        if name in ['RandomForest', 'XGBoost']:
            perm_imp = permutation_importance(best, X, y, n_repeats=10, random_state=42)
            feat_imp = pd.Series(perm_imp.importances_mean, index=X.columns)
            feat_imp.sort_values(ascending=False).to_csv(f'{name}_feature_importance.csv')
            # Plot feature importance
            plot_feature_importance(feat_imp, model_name=name, top_n=10)
        else:  # Logistic Regression
            feat_imp = pd.Series(best.named_steps['model'].coef_[0], index=X.columns)
            feat_imp = feat_imp.abs().sort_values(ascending=False)
            feat_imp.to_csv(f'{name}_coefficients.csv')
            # Plot feature importance (coefficients magnitude)
            plot_feature_importance(feat_imp, model_name=name, top_n=10)

        # Save results
        results.append({
            'model': name,
            'best_params': grid.best_params_,
            'roc_auc': auc_score,
            'pr_auc': pr_auc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp
        })

        # ROC curve data for combined plot
        fpr, tpr, _ = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)
        roc_curves[name] = (fpr, tpr, roc_auc)

        print(f"Best AUC: {auc_score:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print_confusion_matrix(tp, tn, fp, fn)
    
    # Plot all ROC curves together
    plot_combined_roc(roc_curves)

    # Export summary
    pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
    print(f"\nAll results saved to {RESULTS_PATH}")

if __name__ == '__main__':
    main()
