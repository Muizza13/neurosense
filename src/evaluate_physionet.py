import sys; sys.path.insert(0,".")
import numpy as np, json
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, cross_val_predict
from src.physionet import feature_names

d = np.load("data/processed/physionet_features.npz")
X, y, g = d["X"], d["y"], d["g"]
names = feature_names()
def pipe(m): return Pipeline([("s",StandardScaler()),("m",m)])
RNG=42

# ---------- WITHIN-SUBJECT (trial-level 5-fold CV per subject) ----------
print("=== WITHIN-SUBJECT (5-fold trial CV per subject) ===")
print(f"{'model':13s}{'mean balAcc':>13s}{'mean AUC':>11s}{'>chance':>9s}")
within = {}
for nm, mk in [("LogReg", lambda: LogisticRegression(max_iter=2000,C=0.5)),
               ("RandomForest", lambda: RandomForestClassifier(n_estimators=300,random_state=RNG))]:
    baccs, aucs = [], []
    for s in np.unique(g):
        Xs, ys = X[g==s], y[g==s]
        skf = StratifiedKFold(5, shuffle=True, random_state=RNG)
        proba = cross_val_predict(pipe(mk()), Xs, ys, cv=skf, method="predict_proba")[:,1]
        baccs.append(balanced_accuracy_score(ys, (proba>=0.5).astype(int)))
        aucs.append(roc_auc_score(ys, proba))
    baccs, aucs = np.array(baccs), np.array(aucs)
    within[nm] = dict(balacc_mean=baccs.mean(), balacc_std=baccs.std(),
                      auc_mean=aucs.mean(), auc_std=aucs.std(),
                      n_above_chance=int((baccs>0.5).sum()))
    print(f"{nm:13s}{baccs.mean():.3f}+/-{baccs.std():.3f}{aucs.mean():>8.3f}{'':3s}{within[nm]['n_above_chance']}/10")

# ---------- CROSS-SUBJECT (leave-one-subject-out) ----------
print("\n=== CROSS-SUBJECT (leave-one-subject-out) ===")
logo = LeaveOneGroupOut()
cross = {}
for nm, mk in [("LogReg", lambda: LogisticRegression(max_iter=2000,C=0.5)),
               ("RandomForest", lambda: RandomForestClassifier(n_estimators=300,random_state=RNG))]:
    proba = cross_val_predict(pipe(mk()), X, y, cv=logo, groups=g, method="predict_proba")[:,1]
    bac = balanced_accuracy_score(y, (proba>=0.5).astype(int)); auc = roc_auc_score(y, proba)
    cross[nm] = dict(balacc=bac, auc=auc)
    print(f"{nm:13s} balAcc={bac:.3f}  AUC={auc:.3f}")

# ---------- DESCRIPTIVE: which features does the model lean on? ----------
print("\n=== Top features (|standardized LogReg coef|, descriptive) ===")
from sklearn.linear_model import LogisticRegression as LR
p = pipe(LR(max_iter=2000, C=0.5)).fit(X, y)
coef = p.named_steps["m"].coef_[0]
order = np.argsort(np.abs(coef))[::-1][:8]
for i in order:
    print(f"  {names[i]:10s} coef={coef[i]:+.3f}")

json.dump({"within_subject":within,"cross_subject":cross,
           "top_features":[names[i] for i in order]},
          open("reports/physionet_results.json","w"), indent=2)
print("\nsaved reports/physionet_results.json")
