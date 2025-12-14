import numpy as np
from config import Config

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score
from auto_sklearn2.classifier import AutoSklearnClassifier

# ==== Classification ====
def evaluate(X: np.ndarray, y: np.ndarray, cfg: Config) -> dict[str, float]:
    results = {}

    cv = StratifiedKFold(n_splits=cfg.kfolds, shuffle=True, random_state=cfg.random_state)

    # kNN Classifier
    knn_clf = SKPipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ])
    acc_knn = cross_val_score(knn_clf, X, y, cv=cv, scoring='accuracy').mean()
    results['knn_accuracy'] = acc_knn

    # SVM RBF Classifier
    svm_clf = SKPipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=1.0, gamma='scale'))
    ])
    acc_svm = cross_val_score(svm_clf, X, y, cv=cv, scoring='accuracy').mean()
    results['svm_accuracy'] = acc_svm

    y = np.array(y)
    for name, clf in [('knn', knn_clf), ('svm', svm_clf)]:
        accs, sens, specs = [], [], []
        
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            accs.append(accuracy_score(y_test, y_pred))

            sens.append(recall_score(y_test, y_pred, pos_label=1, average='weighted'))

            specs.append(recall_score(y_test, y_pred, pos_label=0, average='weighted'))

        results[f'{name}_accuracy'] = np.mean(accs)
        results[f'{name}_sensitivity'] = np.mean(sens)
        results[f'{name}_specificity'] = np.mean(specs)



    return results

def evaluate_auto(X: np.ndarray, y: np.ndarray, cfg: Config) -> dict[str, float]:
    results = {}

    cv = StratifiedKFold(n_splits=cfg.kfolds, shuffle=True, random_state=cfg.random_state)

    auto_clf = AutoSklearnClassifier(
        #time_left_for_this_task=cfg.autosklearn_time,
        time_limit=cfg.autosklearn_per_run,
        #memory_limit=cfg.autosklearn_memory,
        #seed=cfg.random_state,
        #resampling_strategy=cv,
        #resampling_strategy_arguments={'folds' : cfg.kfolds}
    )

    acc, sens, specs = [], [], []
    y = np.array(y)

    for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            auto_clf.fit(X_train, y_train)
            y_pred = auto_clf.predict(X_test)

            acc.append(accuracy_score(y_test, y_pred))

            sens.append(recall_score(y_test, y_pred, pos_label=1, average='weighted'))

            specs.append(recall_score(y_test, y_pred, pos_label=0, average='weighted'))

    results[f'auto_accuracy'] = np.mean(acc)
    results[f'auto_sensitivity'] = np.mean(sens)
    results[f'auto_specificity'] = np.mean(specs)


    return results

def evaluate_auto_official(X_train: np.ndarray, y_train: np.ndarray,
                                 X_test: np.ndarray, y_test: np.ndarray,
                                 cfg: Config) -> dict[str, float]:
    results = {}

    # === Cross Validation solo sul train ufficiale ===
    cv = StratifiedKFold(n_splits=cfg.kfolds, shuffle=True, random_state=cfg.random_state)

    cv_acc, cv_sens, cv_spec = [], [], []
    y_train = np.array(y_train)

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        auto_clf = AutoSklearnClassifier(time_limit=cfg.autosklearn_time)
        auto_clf.fit(X_tr, y_tr)
        y_pred_val = auto_clf.predict(X_val)

        cv_acc.append(accuracy_score(y_val, y_pred_val))
        cv_sens.append(recall_score(y_val, y_pred_val, pos_label=1, average='weighted'))
        cv_spec.append(recall_score(y_val, y_pred_val, pos_label=0, average='weighted'))

    results['auto_accuracy_intra_train_cv'] = np.mean(cv_acc)
    results['auto_sensitivity_intra_train_cv'] = np.mean(cv_sens)
    results['auto_specificity_intra_train_cv'] = np.mean(cv_spec)

    # === Fit finale sul train completo ===
    final_clf = AutoSklearnClassifier(time_limit=cfg.autosklearn_time)
    final_clf.fit(X_train, y_train)
    y_pred_test = final_clf.predict(X_test)

    results['auto_accuracy_official_split'] = accuracy_score(y_test, y_pred_test)
    results['auto_sensitivity_official_split'] = recall_score(y_test, y_pred_test, pos_label=1, average='weighted')
    results['auto_specificity_official_split'] = recall_score(y_test, y_pred_test, pos_label=0, average='weighted')

    return results

