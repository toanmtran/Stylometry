import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import classification_report, confusion_matrix

def run_rigorous_train(seed_path):
    print(f"Seed: {seed_path}")
    data = np.load(seed_path)
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    groups_train = data['groups_train']

    # 1. Hyperparameter domains
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'gamma': [0.0001, 0.001, 0.01, 0.1]
    }

    # 2. GroupKFold
    gkf = GroupKFold(n_splits=5)

    # 3. Grid Search
    grid_search = GridSearchCV(
        estimator=SVC(kernel='rbf', random_state=42),
        param_grid=param_grid,
        cv=gkf,
        scoring='accuracy',
        return_train_score=True,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train, groups=groups_train)

    # Phase 1
    results_df = pd.DataFrame(grid_search.cv_results_)

    # Make copy
    report_phase1 = results_df[[
        'param_C', 'param_gamma',
        'mean_train_score', 'mean_test_score', 'std_test_score'
    ]].copy()

    report_phase1.rename(columns={
        'param_C': 'C',
        'param_gamma': 'Gamma',
        'mean_train_score': 'Train Acc (Mean)',
        'mean_test_score': 'Val Acc (Mean)',
        'std_test_score': 'Val Std Dev'
    }, inplace=True)

    report_phase1['Gap'] = report_phase1['Train Acc (Mean)'] - report_phase1['Val Acc (Mean)']
    report_phase1 = report_phase1.sort_values(by='Val Acc (Mean)', ascending=False)

    print("Cross Validation Results:")
    print(report_phase1.to_string(index=False))

    # csv to report
    report_phase1.to_csv(f"phase1_results_{seed_path.split('.')[0]}.csv", index=False)

    # Performance on test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print(f"Best hyperparameters: C={grid_search.best_params_['C']}, Gamma={grid_search.best_params_['gamma']}")

    print("Result on test set")
    print(classification_report(y_test, y_pred, target_names=['Different Author', 'Same Author']))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    return report_phase1, grid_search.best_params_

if __name__ == "__main__":
    run_rigorous_train('processed_data_seed_21.npz')