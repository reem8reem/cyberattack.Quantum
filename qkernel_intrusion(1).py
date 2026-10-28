# Quantum kernel SVM vs classical RBF SVM for healthcare network intrusion detection
#
# HOW TO USE IN COLAB: paste each "# %%" block into its own cell (or upload this
# file and run it). Run the install line once, in its own cell:
#     !pip -q install pennylane scikit-learn pandas
#
# DEMO MODE: with CSV_PATH = None the script uses synthetic data, only to check that
# the pipeline runs. NEVER report demo-mode numbers as results.

# %% 1. Config  (edit this block only)
CSV_PATH = "nsl.csv"                 # e.g. "/content/CICIoMT2024.csv" or "/content/NSL_KDD.csv"
LABEL_COL = "label"             # name of the column holding attack / normal labels
BENIGN_VALUES = {"normal", "benign", "0"}   # label values (lowercase) meaning "not an attack"
N_PER_CLASS = 300               # samples drawn per class (balanced)
N_QUBITS = 4                    # = number of PCA features = number of qubits
SEEDS = [0, 1, 2]               # repeat the experiment to report mean +/- std

# %% 2. Imports
import time
import numpy as np
import pandas as pd
import pennylane as qml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)


# %% 3. Data loading  (swap datasets by changing the config above)
def load_data(seed):
    if CSV_PATH is None:  # demo mode only
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=4000, n_features=20, n_informative=8,
                                   weights=[0.7, 0.3], random_state=seed)
        return pd.DataFrame(X), pd.Series(y)
    df = pd.read_csv(CSV_PATH)
    y = (~df[LABEL_COL].astype(str).str.strip().str.lower().isin(BENIGN_VALUES)).astype(int)
    X = df.drop(columns=[LABEL_COL]).select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    X = X.loc[:, X.nunique() > 1]          # drop constant columns
    return X, y


def balanced_sample(X, y, n_per_class, seed):
    rng = np.random.RandomState(seed)
    idx = []
    for c in (0, 1):
        ci = np.where(y.values == c)[0]
        idx.extend(rng.choice(ci, size=min(n_per_class, len(ci)), replace=False))
    idx = np.array(idx)
    return X.iloc[idx].values, y.values[idx]


# %% 4. Quantum kernel (fidelity kernel, exact statevector simulation)
dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev)
def encode(x):
    qml.IQPEmbedding(x, wires=range(N_QUBITS), n_repeats=2)
    return qml.state()


def states(Z):
    return np.array([np.array(encode(z)) for z in Z])


def quantum_kernel(A, B):
    # k(a, b) = |<phi(a)|phi(b)>|^2
    return np.abs(A.conj() @ B.T) ** 2


# %% 5. Metrics
def metrics(y_true, y_pred, score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, score),
        "false_alarm_rate": fp / (fp + tn),
    }


# %% 6. One full experiment (identical preprocessing for both models)
def run(seed):
    X, y = load_data(seed)
    Xs, ys = balanced_sample(X, y, N_PER_CLASS, seed)
    Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=0.3, stratify=ys, random_state=seed)

    # Fit all preprocessing on TRAIN only, then apply to test
    sc = StandardScaler().fit(Xtr)
    pca = PCA(N_QUBITS, random_state=seed).fit(sc.transform(Xtr))
    Ztr = pca.transform(sc.transform(Xtr))
    Zte = pca.transform(sc.transform(Xte))
    mm = MinMaxScaler((0, np.pi)).fit(Ztr)
    Ztr = mm.transform(Ztr)
    Zte = np.clip(mm.transform(Zte), 0, np.pi)

    # Classical baseline: RBF-kernel SVM, tuned with 3-fold CV
    t = time.time()
    gc = GridSearchCV(SVC(kernel="rbf"),
                      {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.1, 1, 10]},
                      cv=3, scoring="f1").fit(Ztr, ytr)
    c_time = time.time() - t
    c_res = metrics(yte, gc.predict(Zte), gc.decision_function(Zte))
    c_res.update(model="classical_rbf_svm", seed=seed, train_time_s=c_time,
                 best_params=str(gc.best_params_))

    # Quantum model: same SVM, quantum kernel, same C grid
    t = time.time()
    Str, Ste = states(Ztr), states(Zte)
    Ktr, Kte = quantum_kernel(Str, Str), quantum_kernel(Ste, Str)
    k_time = time.time() - t
    t = time.time()
    gq = GridSearchCV(SVC(kernel="precomputed"), {"C": [0.1, 1, 10, 100]},
                      cv=3, scoring="f1").fit(Ktr, ytr)
    q_time = time.time() - t + k_time
    q_res = metrics(yte, gq.predict(Kte), gq.decision_function(Kte))
    q_res.update(model="quantum_kernel_svm", seed=seed, train_time_s=q_time,
                 best_params=str(gq.best_params_))
    return [c_res, q_res], Ztr


# %% 7. Run everything and summarize
rows = []
for s in SEEDS:
    r, Ztr_last = run(s)
    rows.extend(r)
results = pd.DataFrame(rows)
results.to_csv("results_raw.csv", index=False)

cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "false_alarm_rate", "train_time_s"]
summary = results.groupby("model")[cols].agg(["mean", "std"]).round(3)
print(summary.T)
summary.to_csv("results_summary.csv")

# Quantum resource metrics for the README
n_train = len(Ztr_last)
n_test = int(round(n_train * 0.3 / 0.7))
try:
    depth = qml.specs(encode)(Ztr_last[0])["resources"].depth
except Exception:
    depth = "n/a (check qml.specs output manually)"
print("\nQuantum resources")
print("qubits:", N_QUBITS)
print("circuit depth:", depth)
print("trainable parameters: 0 (fixed feature map; only the SVM is trained)")
print("kernel entries needed on hardware (train + test):",
      n_train * (n_train - 1) // 2 + n_test * n_train)
