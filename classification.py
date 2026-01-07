import numpy as np
from config import Config

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score
from auto_sklearn2.classifier import AutoSklearnClassifier

# ==== Classification ====
import numpy as np
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score



def evaluate(X_train: np.ndarray, y_train: np.ndarray,
             X_test: np.ndarray, y_test: np.ndarray,
             cfg: Config) -> dict[str, float]:
    results = {}

    # === Label Processing ===
    mask_train = y_train != "Unknown"
    X_train, y_train = X_train[mask_train], y_train[mask_train]

    mask_test = y_test != "Unknown"
    X_test, y_test = X_test[mask_test], y_test[mask_test]

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)


    def icbhi_metrics(y_true, y_pred):
        # classi: 0 = Normal, 1 = Anomalous
        P_anom = np.sum((y_true == 1) & (y_pred == 1))
        N_anom = np.sum(y_true == 1)
        P_norm = np.sum((y_true == 0) & (y_pred == 0))
        N_norm = np.sum(y_true == 0)

        Se = P_anom / N_anom if N_anom > 0 else 0.0
        Sp = P_norm / N_norm if N_norm > 0 else 0.0
        Score = (Se + Sp) / 2
        return Se, Sp, Score


    # === kNN Classifier ===
    knn_clf = SKPipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=7))
    ])
    knn_clf.fit(X_train, y_train)
    y_pred_knn = knn_clf.predict(X_test)

    Se_knn, Sp_knn, Score_knn = icbhi_metrics(y_test, y_pred_knn)
    #results['knn_accuracy'] = accuracy_score(y_test, y_pred_knn)
    results['knn_sensitivity'] = Se_knn * 100
    results['knn_specificity'] = Sp_knn * 100
    results['knn_score'] = Score_knn * 100

    # === SVM RBF Classifier ===
    svm_clf = SKPipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=0.1, gamma='scale', class_weight='balanced'))
    ])
    svm_clf.fit(X_train, y_train)
    y_pred_svm = svm_clf.predict(X_test)

    Se_svm, Sp_svm, Score_svm = icbhi_metrics(y_test, y_pred_svm)
    #results['svm_accuracy'] = accuracy_score(y_test, y_pred_svm)
    results['svm_sensitivity'] = Se_svm * 100
    results['svm_specificity'] = Sp_svm * 100
    results['svm_score'] = Score_svm * 100

    return results

def evaluate_auto(X: np.ndarray, y: np.ndarray, cfg: Config) -> dict[str, float]:
    results = {}

    # === Cross Validation ===
    cv = StratifiedKFold(n_splits=cfg.kfolds, shuffle=True, random_state=cfg.random_state)

    # === Automatic Classification ===
    auto_clf = AutoSklearnClassifier(
        time_limit=cfg.autosklearn_per_run,
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

    cv_acc, cv_sens, cv_spec = [], [], []
    y_train = np.array(y_train)

    # === Automatic Classification ===
    auto_clf = AutoSklearnClassifier(time_limit=cfg.autosklearn_time, random_state=cfg.random_state)

    scaler = StandardScaler()
    scaler.fit(X_train, y_train)
    y_pred = auto_clf.predict(X_test)

    results['auto_accuracy_official_split'] = auto_sklearn.score(X_test, y_test)

    return results